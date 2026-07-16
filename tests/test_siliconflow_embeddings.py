"""SiliconFlowEmbeddingProvider 单元测试。

所有 HTTP 调用都被 mock；测试中使用的 Key 全部是明显的假 Key
（``test-secret-never-log``），不会被记录到日志中。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from rag.siliconflow_embeddings import (  # noqa: E402
    EmbeddingAuthError,
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingResponseFormatError,
    EmbeddingServerError,
    EmbeddingTimeoutError,
    SiliconFlowEmbeddingProvider,
)


# 假 Key：明显是测试用途，确保不会被误认为真实 Key
FAKE_KEY = "test-secret-never-log"
DIM = 1024
BATCH_SIZE = 4  # 本测试中显式配置的 batch_size；默认值仍是 16


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _new_settings(**overrides) -> Settings:
    """构造一个不读 .env 的 Settings。"""
    with mock.patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _new_provider(
    settings: Settings | None = None,
    session: requests.Session | None = None,
    batch_size: int = 16,
) -> SiliconFlowEmbeddingProvider:
    s = settings or _new_settings(
        siliconflow_api_key=FAKE_KEY,
        siliconflow_embedding_batch_size=batch_size,
    )
    return SiliconFlowEmbeddingProvider(settings=s, session=session or requests.Session())


def _make_payload(texts: list[str], dim: int = DIM, shuffle: bool = False) -> dict:
    """生成 SiliconFlow embeddings 响应的 payload 模拟数据。

    :param shuffle: 若为 True，则把 data 列表顺序打乱，以验证 index 恢复。
    """
    items = [{"index": i, "embedding": _make_vector(i, dim)} for i in range(len(texts))]
    if shuffle:
        import random

        random.seed(42)
        random.shuffle(items)
    return {"data": items, "model": "Qwen/Qwen3-Embedding-4B", "usage": {"total_tokens": 0}}


def _make_vector(seed: int, dim: int = DIM) -> list[float]:
    """生成确定性的 dim 维向量；值域在 [-1, 1]。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=dim).tolist()


def _fake_response(
    payload: dict | Exception,
    status_code: int = 200,
) -> requests.Response:
    """构造 ``requests.Response`` 模拟对象。"""
    resp = requests.Response()
    resp.status_code = status_code
    resp._content = (  # type: ignore[attr-defined]
        b"" if isinstance(payload, Exception) else _json_dumps(payload)
    )
    if isinstance(payload, Exception):
        resp.reason = "Mock Error"  # type: ignore[attr-defined]
        # 让 resp.json() 抛同样异常
        def _raise_json(*_a, **_k):  # noqa: ANN001
            raise payload

        resp.json = _raise_json  # type: ignore[method-assign]
    else:
        resp.encoding = "utf-8"
    return resp


def _json_dumps(obj) -> bytes:
    import json

    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


class _MockAdapter(requests.adapters.HTTPAdapter):
    """通过 ``session.mount`` 注入；按 ``url`` 返回预编排的响应序列。"""

    def __init__(self, responses: list):
        super().__init__()
        self._responses = list(responses)

    def send(self, request, **kwargs):  # noqa: ANN001 - requests API
        if not self._responses:
            raise AssertionError("MockAdapter：没有更多预设响应。")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _attach(session: requests.Session, responses: list) -> None:
    session.mount("http://", _MockAdapter(responses))
    session.mount("https://", _MockAdapter(responses))


# ---------------------------------------------------------------------------
# 1. 单条文本成功
# ---------------------------------------------------------------------------
def test_embed_single_text_success():
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(["你好"]))])
    p = _new_provider(session=session)

    arr = p.embed_documents(["你好"])
    assert arr.shape == (1, DIM)
    assert arr.dtype == np.float32
    # L2 范数接近 1
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. 多条文本成功
# ---------------------------------------------------------------------------
def test_embed_multiple_texts_success():
    texts = ["文本A", "文本B", "文本C", "文本D", "文本E"]
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(texts))])
    p = _new_provider(session=session)

    arr = p.embed_documents(texts)
    assert arr.shape == (5, DIM)
    assert arr.dtype == np.float32
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. 批次拆分正确
# ---------------------------------------------------------------------------
def test_batch_splitting_serial_calls():
    """输入 10 条 + batch_size=4 → 应发 3 次请求（4+4+2）。"""
    texts = [f"文本-{i}" for i in range(10)]
    # 预设 3 个响应
    responses = [
        _fake_response(_make_payload(texts[0:4])),
        _fake_response(_make_payload(texts[4:8])),
        _fake_response(_make_payload(texts[8:10])),
    ]
    session = requests.Session()
    adapter = _MockAdapter(responses)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    p = _new_provider(session=session, batch_size=4)

    arr = p.embed_documents(texts)
    assert arr.shape == (10, DIM)
    # 3 个响应都应被消费
    assert len(adapter._responses) == 0  # type: ignore[attr-defined]


def test_batch_size_one_still_works():
    texts = ["A", "B", "C"]
    responses = [_fake_response(_make_payload([t])) for t in texts]
    session = requests.Session()
    adapter = _MockAdapter(responses)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    p = _new_provider(
        session=session,
        settings=_new_settings(
            siliconflow_api_key=FAKE_KEY,
            siliconflow_embedding_batch_size=1,
        ),
    )
    arr = p.embed_documents(texts)
    assert arr.shape == (3, DIM)


# ---------------------------------------------------------------------------
# 4. 返回顺序按 index 恢复
# ---------------------------------------------------------------------------
def test_order_recovered_by_index():
    """服务端乱序返回（data 列表 index=2,0,1），结果应按输入顺序 0,1,2 排列。"""
    texts = ["T0", "T1", "T2"]
    payload = _make_payload(texts, shuffle=True)
    # 服务端只打乱了 data 列表顺序，index 字段仍是原始位置
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    arr = p.embed_documents(texts)
    expected_idx0 = _make_vector(0)
    # 第一行应等于 _make_vector(0)（原始 index=0）
    assert np.allclose(arr[0], np.asarray(expected_idx0, dtype=np.float32) / np.linalg.norm(expected_idx0), atol=1e-4)
    # 第三行对应 index=2
    expected_idx2 = _make_vector(2)
    assert np.allclose(arr[2], np.asarray(expected_idx2, dtype=np.float32) / np.linalg.norm(expected_idx2), atol=1e-4)


# ---------------------------------------------------------------------------
# 5. shape 正确
# ---------------------------------------------------------------------------
def test_output_shape_matches_input_count():
    texts = [f"X{i}" for i in range(7)]
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(texts))])
    p = _new_provider(session=session)

    arr = p.embed_documents(texts)
    assert arr.shape == (7, DIM)


# ---------------------------------------------------------------------------
# 6. dtype = float32
# ---------------------------------------------------------------------------
def test_dtype_is_float32():
    texts = ["A"]
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(texts))])
    p = _new_provider(session=session)

    arr = p.embed_documents(texts)
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# 7. L2 范数接近 1
# ---------------------------------------------------------------------------
def test_l2_norm_unit():
    texts = ["A", "B", "C"]
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(texts))])
    p = _new_provider(session=session)

    arr = p.embed_documents(texts)
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. 空输入异常
# ---------------------------------------------------------------------------
def test_empty_input_raises():
    p = _new_provider()
    with pytest.raises(EmbeddingError):
        p.embed_documents([])


# ---------------------------------------------------------------------------
# 9. 返回数量不一致异常
# ---------------------------------------------------------------------------
def test_count_mismatch_raises():
    # 输入 3 条，但响应 data 只有 2 条
    payload = {"data": [{"index": 0, "embedding": _make_vector(0)},
                        {"index": 1, "embedding": _make_vector(1)}]}
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingResponseFormatError) as ei:
        p.embed_documents(["A", "B", "C"])
    assert "不一致" in str(ei.value)


# ---------------------------------------------------------------------------
# 10. 返回维度错误异常
# ---------------------------------------------------------------------------
def test_dimension_mismatch_raises():
    # 返回的 embedding 维度为 8，与 dim=1024 不一致
    payload = {"data": [{"index": 0, "embedding": [0.1] * 8}]}
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingResponseFormatError):
        p.embed_documents(["A"])


# ---------------------------------------------------------------------------
# 11. 返回包含 NaN/Inf 异常
# ---------------------------------------------------------------------------
def test_nan_or_inf_raises():
    # 含 NaN
    vec = _make_vector(0)
    vec[10] = float("nan")
    payload = {"data": [{"index": 0, "embedding": vec}]}
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingResponseFormatError):
        p.embed_documents(["A"])


def test_inf_raises():
    vec = _make_vector(0)
    vec[5] = float("inf")
    payload = {"data": [{"index": 0, "embedding": vec}]}
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingResponseFormatError):
        p.embed_documents(["A"])


# ---------------------------------------------------------------------------
# 12. 缺少 data 字段异常
# ---------------------------------------------------------------------------
def test_missing_data_field_raises():
    payload = {"model": "Qwen/Qwen3-Embedding-4B"}  # 没有 data
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingResponseFormatError):
        p.embed_documents(["A"])


def test_missing_embedding_field_raises():
    payload = {"data": [{"index": 0}]}  # 没有 embedding
    session = requests.Session()
    _attach(session, [_fake_response(payload)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingResponseFormatError):
        p.embed_documents(["A"])


# ---------------------------------------------------------------------------
# 13. 401 不重试
# ---------------------------------------------------------------------------
def test_401_no_retry():
    session = requests.Session()
    # 只准备 1 个 401 响应；如果发生重试，第二个响应会触发断言失败
    _attach(session, [_fake_response({}, status_code=401)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingAuthError):
        p.embed_documents(["A"])


# ---------------------------------------------------------------------------
# 14. 403 不重试
# ---------------------------------------------------------------------------
def test_403_no_retry():
    session = requests.Session()
    _attach(session, [_fake_response({}, status_code=403)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingAuthError):
        p.embed_documents(["A"])


# ---------------------------------------------------------------------------
# 15. 429 重试后成功
# ---------------------------------------------------------------------------
def test_429_retry_then_success():
    session = requests.Session()
    _attach(
        session,
        [
            _fake_response({}, status_code=429),
            _fake_response(_make_payload(["A"])),
        ],
    )
    p = _new_provider(session=session)

    arr = p.embed_documents(["A"])
    assert arr.shape == (1, DIM)


# ---------------------------------------------------------------------------
# 16. 500 重试后成功
# ---------------------------------------------------------------------------
def test_500_retry_then_success():
    session = requests.Session()
    _attach(
        session,
        [
            _fake_response({}, status_code=500),
            _fake_response(_make_payload(["A"])),
        ],
    )
    p = _new_provider(session=session)

    arr = p.embed_documents(["A"])
    assert arr.shape == (1, DIM)


# ---------------------------------------------------------------------------
# 17. Timeout 重试后成功
# ---------------------------------------------------------------------------
def test_timeout_retry_then_success(monkeypatch):
    """第一次抛 requests.Timeout，第二次成功。"""
    # 用 Session 的 send 抛 Timeout 异常
    call_count = {"n": 0}

    def fake_send(self, request, **kwargs):  # noqa: ANN001
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.Timeout("simulated timeout")
        # 第二次返回正常响应
        return _fake_response(_make_payload(["A"]))

    session = requests.Session()
    monkeypatch.setattr(requests.Session, "send", fake_send)
    p = _new_provider(session=session)

    arr = p.embed_documents(["A"])
    assert arr.shape == (1, DIM)
    # 必须至少重试一次
    assert call_count["n"] >= 2


def test_connection_error_retry_then_success(monkeypatch):
    call_count = {"n": 0}

    def fake_send(self, request, **kwargs):  # noqa: ANN001
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise requests.ConnectionError("simulated connection error")
        return _fake_response(_make_payload(["A"]))

    session = requests.Session()
    monkeypatch.setattr(requests.Session, "send", fake_send)
    p = _new_provider(session=session)

    arr = p.embed_documents(["A"])
    assert arr.shape == (1, DIM)


# ---------------------------------------------------------------------------
# 18. 连续失败达到上限后抛错
# ---------------------------------------------------------------------------
def test_429_exhausts_retries_raises():
    """连续 4 次 429：3 次重试都失败，最终抛 EmbeddingRateLimitError。"""
    session = requests.Session()
    # 第 1 次 + 2 次重试 = 3 次全部失败 + 第 4 次本不应消费
    _attach(
        session,
        [
            _fake_response({}, status_code=429),
            _fake_response({}, status_code=429),
            _fake_response({}, status_code=429),
            _fake_response({}, status_code=429),
        ],
    )
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingRateLimitError):
        p.embed_documents(["A"])


def test_500_exhausts_retries_raises():
    session = requests.Session()
    _attach(
        session,
        [
            _fake_response({}, status_code=500),
            _fake_response({}, status_code=500),
            _fake_response({}, status_code=500),
        ],
    )
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingServerError):
        p.embed_documents(["A"])


def test_timeout_exhausts_retries_raises(monkeypatch):
    def fake_send(self, request, **kwargs):  # noqa: ANN001
        raise requests.Timeout("always timeout")

    session = requests.Session()
    monkeypatch.setattr(requests.Session, "send", fake_send)
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingTimeoutError):
        p.embed_documents(["A"])


# ---------------------------------------------------------------------------
# 19. 未配置 Key 时明确报错
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_configuration_error():
    # 不注入 SILICONFLOW_API_KEY
    s = _new_settings()  # 不传 key
    p = SiliconFlowEmbeddingProvider(settings=s, session=requests.Session())

    with pytest.raises(EmbeddingConfigurationError) as ei:
        p.embed_documents(["A"])
    # 错误消息必须明确指出环境变量名
    assert "SILICONFLOW_API_KEY" in str(ei.value)


def test_missing_api_key_embed_query_raises():
    s = _new_settings()
    p = SiliconFlowEmbeddingProvider(settings=s, session=requests.Session())

    with pytest.raises(EmbeddingConfigurationError):
        p.embed_query("A")


# ---------------------------------------------------------------------------
# 20. 日志中不含测试 Key
# ---------------------------------------------------------------------------
class _LogCapture:
    def __init__(self):
        self.records: list[str] = []

    def __call__(self, msg: str) -> None:
        self.records.append(msg)


def test_logs_do_not_contain_api_key(caplog):
    """即使发生错误，日志记录中也绝不出现测试 Key。"""
    caplog.set_level(logging.DEBUG)

    # 故意构造一个 401，让 provider 记录状态码与「未授权」字样
    session = requests.Session()
    _attach(session, [_fake_response({}, status_code=401)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingAuthError):
        p.embed_documents(["A"])

    # caplog.text 包含所有 handler 输出
    all_log_text = caplog.text
    assert FAKE_KEY not in all_log_text, (
        f"测试 Key 出现在日志中！\n---- 日志内容 ---\n{all_log_text}"
    )
    # 同时也不应该出现 "Bearer test-secret..." 之类的拼接
    assert "Bearer test-secret" not in all_log_text


def test_logs_do_not_contain_key_on_success(caplog):
    """成功路径的日志也不应包含测试 Key。"""
    caplog.set_level(logging.DEBUG)

    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(["A"]))])
    p = _new_provider(session=session)
    p.embed_documents(["A"])

    assert FAKE_KEY not in caplog.text


def test_logs_do_not_contain_key_on_retry(caplog):
    """重试场景的日志也不应包含测试 Key。"""
    caplog.set_level(logging.DEBUG)

    session = requests.Session()
    _attach(
        session,
        [
            _fake_response({}, status_code=429),
            _fake_response(_make_payload(["A"])),
        ],
    )
    p = _new_provider(session=session)
    p.embed_documents(["A"])

    assert FAKE_KEY not in caplog.text


# ===========================================================================
# 补充：embed_query
# ===========================================================================
def test_embed_query_returns_shape_1_dim():
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload(["Q"]))])
    p = _new_provider(session=session)

    arr = p.embed_query("Q")
    assert arr.shape == (1, DIM)
    assert arr.dtype == np.float32


def test_embed_query_empty_string_still_calls_api():
    """空字符串允许（调用 API，由服务端判断）。但不允许 list 输入。"""
    session = requests.Session()
    _attach(session, [_fake_response(_make_payload([""]))])
    p = _new_provider(session=session)

    arr = p.embed_query("")
    assert arr.shape == (1, DIM)


# ===========================================================================
# 补充：4xx 不重试 + 边界
# ===========================================================================
def test_400_no_retry():
    session = requests.Session()
    _attach(session, [_fake_response({}, status_code=400)])
    p = _new_provider(session=session)

    with pytest.raises(EmbeddingError) as ei:
        p.embed_documents(["A"])
    # 不能是限流/服务端/鉴权错误的子类
    assert not isinstance(ei.value, EmbeddingAuthError)
    assert not isinstance(ei.value, EmbeddingRateLimitError)
    assert not isinstance(ei.value, EmbeddingServerError)


def test_endpoint_construction_no_double_v1():
    """base_url 已含 /v1 时，endpoint 不应出现重复 /v1。"""
    s = _new_settings(
        siliconflow_api_key=FAKE_KEY,
        siliconflow_base_url="https://api.siliconflow.cn/v1",
    )
    p = SiliconFlowEmbeddingProvider(settings=s, session=requests.Session())
    assert p._build_endpoint() == "https://api.siliconflow.cn/v1/embeddings"  # type: ignore[attr-defined]

    # 即便 base_url 已含 /embeddings，也不应再追加
    s2 = _new_settings(
        siliconflow_api_key=FAKE_KEY,
        siliconflow_base_url="https://api.siliconflow.cn/v1/embeddings",
    )
    p2 = SiliconFlowEmbeddingProvider(settings=s2, session=requests.Session())
    assert p2._build_endpoint() == "https://api.siliconflow.cn/v1/embeddings"  # type: ignore[attr-defined]


def test_provider_metadata():
    p = _new_provider(batch_size=BATCH_SIZE)
    assert p.model_name == "Qwen/Qwen3-Embedding-4B"
    assert p.dimensions == DIM
    assert p.batch_size == BATCH_SIZE


def test_non_string_input_raises():
    p = _new_provider()
    with pytest.raises(EmbeddingError):
        p.embed_documents(["A", 123])  # type: ignore[list-item]