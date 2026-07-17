"""MiniMax LLM 客户端单元测试。

所有 ``openai.OpenAI`` 调用都被 mock；测试中使用的 Key 全部是明显的假 Key
（``test-minimax-secret-never-log``），不会被记录到日志中。

测试范围：

- 非流式 / 流式成功路径
- 流式跳过 None / 空 content
- messages 校验（空 / 非法 role / 非字符串 content）
- 缺 Key 报错
- 异常映射（Auth / RateLimit / Timeout / Connection / 5xx / 其它 4xx）
- 参数传递（model / temperature / max_tokens）
- 日志中不出现 Key 与完整 Prompt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402

from rag.llm_client import (  # noqa: E402
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMServerError,
    LLMTimeoutError,
    MiniMaxLLMClient,
    ReasoningContentFilter,
    sanitize_model_content,
)


# 假 Key：明显是测试用途，确保不会被误认为真实 Key
FAKE_KEY = "test-minimax-secret-never-log"

# 一个不敏感的完整 Prompt 文本，用于检测日志泄漏
SAMPLE_PROMPT = "用户问题：什么是阿莫西林？\n可用的来源片段：<source id=\"S1\">阿莫西林</source>"
SAMPLE_ANSWER = "阿莫西林是一种抗菌药物 [S1]。"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _new_settings(**overrides) -> Settings:
    """构造一个不读 .env 的 Settings。"""
    with mock.patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _new_client(
    settings: Settings | None = None,
    *,
    mock_create: mock.Mock | None = None,
) -> tuple[MiniMaxLLMClient, mock.Mock]:
    """构造一个注入 mock ``client.chat.completions.create`` 的 LLMClient。"""
    s = settings or _new_settings(minimax_api_key=FAKE_KEY)
    fake_openai = mock.MagicMock(spec=OpenAI)
    if mock_create is not None:
        fake_openai.chat.completions.create = mock_create  # type: ignore[method-assign]
    client = MiniMaxLLMClient(settings=s, client=fake_openai)
    return client, fake_openai


def _make_completion(content: str = SAMPLE_ANSWER) -> ChatCompletion:
    """构造非流式 ChatCompletion 模拟对象。"""
    return ChatCompletion(
        id="cmpl-test",
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        created=0,
        model="MiniMax-M3",
        object="chat.completion",
    )


def _make_chunk(
    content: str | None = None,
    *,
    role: str | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    """构造流式 ChatCompletionChunk 模拟对象。"""
    return ChatCompletionChunk(
        id="cmpl-test",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
        created=0,
        model="MiniMax-M3",
        object="chat.completion.chunk",
    )


def _make_status_error(status_code: int, message: str = "mock") -> APIStatusError:
    """构造一个 ``APIStatusError`` 实例（需要真实的 httpx.Response）。"""
    resp = httpx.Response(
        status_code,
        request=httpx.Request("POST", "http://test.invalid/v1/chat/completions"),
        json={"error": message},
    )
    return APIStatusError(message=message, response=resp, body=None)


# ---------------------------------------------------------------------------
# 1. 非流式成功
# ---------------------------------------------------------------------------
def test_complete_success():
    create_mock = mock.Mock(return_value=_make_completion("你好世界"))
    client, openai_mock = _new_client(mock_create=create_mock)

    out = client.complete(
        [
            {"role": "system", "content": "你是一个助手。"},
            {"role": "user", "content": "你好"},
        ]
    )
    assert out == "你好世界"
    # 请求参数正确
    kwargs = create_mock.call_args.kwargs
    assert kwargs["model"] == "MiniMax-M3"
    assert kwargs["stream"] is False
    assert kwargs["messages"][1]["content"] == "你好"


# ---------------------------------------------------------------------------
# 2. 流式成功
# ---------------------------------------------------------------------------
def test_stream_success():
    chunks = [
        _make_chunk("你"),
        _make_chunk("好"),
        _make_chunk("世界"),
    ]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)

    pieces = list(client.stream([{"role": "user", "content": "hi"}]))
    assert pieces == ["你", "好", "世界"]
    # 验证 stream=True 被传递
    assert create_mock.call_args.kwargs["stream"] is True


# ---------------------------------------------------------------------------
# 3. 流式多个 token 顺序正确
# ---------------------------------------------------------------------------
def test_stream_token_order():
    tokens = ["一", "二", "三", "四", "五"]
    chunks = [_make_chunk(t) for t in tokens]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)

    out = list(client.stream([{"role": "user", "content": "q"}]))
    assert out == tokens
    assert "".join(out) == "一二三四五"


# ---------------------------------------------------------------------------
# 4. delta.content 为 None 时跳过
# ---------------------------------------------------------------------------
def test_stream_skip_none_content():
    chunks = [
        _make_chunk("你"),
        _make_chunk(None),  # role chunk 或 end-of-stream 段
        _make_chunk("好"),
    ]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)

    out = list(client.stream([{"role": "user", "content": "q"}]))
    assert out == ["你", "好"]


# ---------------------------------------------------------------------------
# 5. choices 为空时跳过
# ---------------------------------------------------------------------------
def test_stream_skip_empty_choices():
    # 模拟 OpenAI SDK 在 usage-only chunk 中返回空 choices
    empty_chunk = SimpleNamespace(choices=[])
    normal = _make_chunk("ok")
    create_mock = mock.Mock(return_value=iter([empty_chunk, normal]))
    client, _ = _new_client(mock_create=create_mock)

    out = list(client.stream([{"role": "user", "content": "q"}]))
    assert out == ["ok"]


def test_stream_skip_none_choices_attr():
    # choices 属性本身缺失
    broken = SimpleNamespace(spec=[])  # 没有任何属性
    normal = _make_chunk("ok")
    create_mock = mock.Mock(return_value=iter([broken, normal]))
    client, _ = _new_client(mock_create=create_mock)

    out = list(client.stream([{"role": "user", "content": "q"}]))
    assert out == ["ok"]


# ---------------------------------------------------------------------------
# 6. messages 为空时报错
# ---------------------------------------------------------------------------
def test_empty_messages_raises():
    client, _ = _new_client()
    with pytest.raises(LLMResponseFormatError) as ei:
        client.complete([])
    assert "不能为空" in str(ei.value)


def test_empty_messages_stream_raises():
    client, _ = _new_client()
    with pytest.raises(LLMResponseFormatError):
        list(client.stream([]))


# ---------------------------------------------------------------------------
# 7. role 非法时报错
# ---------------------------------------------------------------------------
def test_invalid_role_raises():
    client, _ = _new_client()
    with pytest.raises(LLMResponseFormatError) as ei:
        client.complete([{"role": "admin", "content": "x"}])
    assert "role" in str(ei.value)


# ---------------------------------------------------------------------------
# 8. content 不是字符串时报错
# ---------------------------------------------------------------------------
def test_non_string_content_raises():
    client, _ = _new_client()
    with pytest.raises(LLMResponseFormatError) as ei:
        client.complete([{"role": "user", "content": ["a", "b"]}])  # type: ignore[list-item]
    assert "content" in str(ei.value)


# ---------------------------------------------------------------------------
# 9. 未配置 Key 时明确报错
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_on_complete():
    create_mock = mock.Mock()
    s = _new_settings()  # 没有 minimax_api_key
    client, _ = _new_client(settings=s, mock_create=create_mock)
    with pytest.raises(LLMConfigurationError) as ei:
        client.complete([{"role": "user", "content": "hi"}])
    assert "MINIMAX_API_KEY" in str(ei.value)
    # 缺 Key 时不能真的发出请求
    create_mock.assert_not_called()


def test_missing_api_key_raises_on_stream():
    s = _new_settings()
    client, _ = _new_client(settings=s, mock_create=mock.Mock())
    with pytest.raises(LLMConfigurationError):
        list(client.stream([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# 10. 模型名称正确传递
# ---------------------------------------------------------------------------
def test_model_name_passed():
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(mock_create=create_mock)
    client.complete([{"role": "user", "content": "q"}])
    assert create_mock.call_args.kwargs["model"] == "MiniMax-M3"
    assert client.model_name == "MiniMax-M3"
    assert client.model == "MiniMax-M3"


def test_model_name_from_settings():
    s = _new_settings(minimax_api_key=FAKE_KEY, minimax_model="Custom-Model-9")
    client, _ = _new_client(settings=s, mock_create=mock.Mock())
    assert client.model_name == "Custom-Model-9"


# ---------------------------------------------------------------------------
# 11. temperature 正确传递
# ---------------------------------------------------------------------------
def test_temperature_override_passed():
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(mock_create=create_mock)
    client.complete(
        [{"role": "user", "content": "q"}],
        temperature=0.7,
    )
    assert create_mock.call_args.kwargs["temperature"] == 0.7


def test_temperature_default_from_settings():
    s = _new_settings(minimax_api_key=FAKE_KEY, minimax_temperature=0.5)
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(settings=s, mock_create=create_mock)
    client.complete([{"role": "user", "content": "q"}])
    assert create_mock.call_args.kwargs["temperature"] == 0.5


# ---------------------------------------------------------------------------
# 12. max_tokens 为 None 时不传参
# ---------------------------------------------------------------------------
def test_max_tokens_none_not_passed():
    s = _new_settings(minimax_api_key=FAKE_KEY, minimax_max_tokens=None)
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(settings=s, mock_create=create_mock)
    client.complete([{"role": "user", "content": "q"}])
    assert "max_tokens" not in create_mock.call_args.kwargs


def test_max_tokens_none_stream_not_passed():
    s = _new_settings(minimax_api_key=FAKE_KEY, minimax_max_tokens=None)
    create_mock = mock.Mock(return_value=iter([_make_chunk("ok")]))
    client, _ = _new_client(settings=s, mock_create=create_mock)
    list(client.stream([{"role": "user", "content": "q"}]))
    assert "max_tokens" not in create_mock.call_args.kwargs


# ---------------------------------------------------------------------------
# 13. max_tokens 有值时正确传递
# ---------------------------------------------------------------------------
def test_max_tokens_passed():
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(mock_create=create_mock)
    client.complete([{"role": "user", "content": "q"}], max_tokens=512)
    assert create_mock.call_args.kwargs["max_tokens"] == 512


def test_max_tokens_default_from_settings():
    s = _new_settings(minimax_api_key=FAKE_KEY, minimax_max_tokens=256)
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(settings=s, mock_create=create_mock)
    client.complete([{"role": "user", "content": "q"}])
    assert create_mock.call_args.kwargs["max_tokens"] == 256


# ---------------------------------------------------------------------------
# 14. 非流式 choices 为空
# ---------------------------------------------------------------------------
def test_complete_empty_choices_raises():
    resp = ChatCompletion(
        id="x",
        choices=[],
        created=0,
        model="MiniMax-M3",
        object="chat.completion",
    )
    create_mock = mock.Mock(return_value=resp)
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMResponseFormatError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    assert "choices" in str(ei.value)


# ---------------------------------------------------------------------------
# 15. 非流式 content 为空
# ---------------------------------------------------------------------------
def test_complete_empty_content_raises():
    resp = _make_completion("")
    create_mock = mock.Mock(return_value=resp)
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMResponseFormatError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    assert "为空" in str(ei.value)


def test_complete_none_content_raises():
    resp = ChatCompletion(
        id="x",
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=None),
                finish_reason="stop",
            )
        ],
        created=0,
        model="MiniMax-M3",
        object="chat.completion",
    )
    create_mock = mock.Mock(return_value=resp)
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMResponseFormatError):
        client.complete([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------------------
# 16. AuthenticationError 映射
# ---------------------------------------------------------------------------
def test_authentication_error_mapped():
    resp = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    create_mock = mock.Mock(side_effect=AuthenticationError(message="x", response=resp, body=None))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMAuthenticationError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    # 错误中不出现 Key 字符
    assert FAKE_KEY not in str(ei.value)
    # 错误中提示是鉴权失败
    assert "鉴权" in str(ei.value)


# ---------------------------------------------------------------------------
# 17. RateLimitError 映射
# ---------------------------------------------------------------------------
def test_rate_limit_error_mapped():
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    create_mock = mock.Mock(side_effect=RateLimitError(message="x", response=resp, body=None))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMRateLimitError):
        client.complete([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------------------
# 18. APITimeoutError 映射
# ---------------------------------------------------------------------------
def test_timeout_error_mapped():
    create_mock = mock.Mock(side_effect=APITimeoutError(request=httpx.Request("POST", "http://x")))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMTimeoutError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    assert "超时" in str(ei.value)


# ---------------------------------------------------------------------------
# 19. APIConnectionError 映射
# ---------------------------------------------------------------------------
def test_connection_error_mapped():
    create_mock = mock.Mock(side_effect=APIConnectionError(request=httpx.Request("POST", "http://x")))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMConnectionError):
        client.complete([{"role": "user", "content": "q"}])


# ---------------------------------------------------------------------------
# 20. 5xx APIStatusError 映射
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_5xx_status_mapped_to_server_error(status_code: int):
    create_mock = mock.Mock(side_effect=_make_status_error(status_code, "boom"))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMServerError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    assert str(status_code) in str(ei.value)


# ---------------------------------------------------------------------------
# 21. 其他 APIStatusError 映射为通用 LLMError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_other_status_mapped_to_llm_error(status_code: int):
    create_mock = mock.Mock(side_effect=_make_status_error(status_code, "bad"))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    # 必须是 LLMError 但不能是特殊子类
    assert not isinstance(ei.value, LLMServerError)
    assert not isinstance(ei.value, LLMAuthenticationError)
    assert not isinstance(ei.value, LLMRateLimitError)
    assert str(status_code) in str(ei.value)


# ---------------------------------------------------------------------------
# 22. 流式中途异常映射
# ---------------------------------------------------------------------------
def test_stream_midway_exception_mapped():
    """流式迭代过程中抛出 APITimeoutError，应转为 LLMTimeoutError。"""

    def gen():
        yield _make_chunk("你")
        yield _make_chunk("好")
        raise APITimeoutError(request=httpx.Request("POST", "http://x"))

    create_mock = mock.Mock(return_value=gen())
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMTimeoutError):
        list(client.stream([{"role": "user", "content": "q"}]))


def test_stream_midway_auth_error_mapped():
    def gen():
        yield _make_chunk("你")
        resp = httpx.Response(401, request=httpx.Request("POST", "http://x"))
        raise AuthenticationError(message="x", response=resp, body=None)

    create_mock = mock.Mock(return_value=gen())
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMAuthenticationError):
        list(client.stream([{"role": "user", "content": "q"}]))


def test_stream_initial_exception_mapped():
    """create() 自身抛 APITimeoutError。"""
    create_mock = mock.Mock(side_effect=APITimeoutError(request=httpx.Request("POST", "http://x")))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMTimeoutError):
        list(client.stream([{"role": "user", "content": "q"}]))


# ---------------------------------------------------------------------------
# 23. 日志中不出现测试 Key
# ---------------------------------------------------------------------------
def test_logs_no_test_key_on_success(caplog):
    caplog.set_level(logging.DEBUG)

    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(mock_create=create_mock)
    client.complete(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
    )

    assert FAKE_KEY not in caplog.text


def test_logs_no_test_key_on_auth_error(caplog):
    caplog.set_level(logging.DEBUG)
    resp = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    create_mock = mock.Mock(side_effect=AuthenticationError(message="x", response=resp, body=None))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMAuthenticationError):
        client.complete([{"role": "user", "content": "q"}])
    assert FAKE_KEY not in caplog.text


def test_logs_no_test_key_on_stream(caplog):
    caplog.set_level(logging.DEBUG)
    create_mock = mock.Mock(return_value=iter([_make_chunk("ok")]))
    client, _ = _new_client(mock_create=create_mock)
    list(client.stream([{"role": "user", "content": "q"}]))
    assert FAKE_KEY not in caplog.text


# ---------------------------------------------------------------------------
# 24. 日志中不出现完整 Prompt
# ---------------------------------------------------------------------------
def test_logs_no_full_prompt_in_output(caplog):
    caplog.set_level(logging.DEBUG)

    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(mock_create=create_mock)
    client.complete(
        [
            {"role": "system", "content": SAMPLE_PROMPT},
            {"role": "user", "content": "什么是阿莫西林？"},
        ]
    )
    # 日志里不能出现完整 prompt 文本
    assert SAMPLE_PROMPT not in caplog.text
    # 也应看不到回答内容
    assert SAMPLE_ANSWER not in caplog.text


def test_logs_no_full_prompt_in_stream(caplog):
    caplog.set_level(logging.DEBUG)
    create_mock = mock.Mock(return_value=iter([_make_chunk("ok")]))
    client, _ = _new_client(mock_create=create_mock)
    list(
        client.stream(
            [
                {"role": "system", "content": SAMPLE_PROMPT},
                {"role": "user", "content": "什么是阿莫西林？"},
            ]
        )
    )
    assert SAMPLE_PROMPT not in caplog.text


# ---------------------------------------------------------------------------
# 25. 异常链保留（raise from）
# ---------------------------------------------------------------------------
def test_exception_chaining_preserved():
    """原始 SDK 异常应通过 ``__cause__`` 保留。"""
    resp = httpx.Response(401, request=httpx.Request("POST", "http://x"))
    original = AuthenticationError(message="x", response=resp, body=None)
    create_mock = mock.Mock(side_effect=original)
    client, _ = _new_client(mock_create=create_mock)

    with pytest.raises(LLMAuthenticationError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    assert ei.value.__cause__ is original


# ---------------------------------------------------------------------------
# 26. 各种合法 role
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_valid_roles_accepted(role: str):
    create_mock = mock.Mock(return_value=_make_completion("ok"))
    client, _ = _new_client(mock_create=create_mock)
    out = client.complete([{"role": role, "content": "x"}])
    assert out == "ok"


# ---------------------------------------------------------------------------
# 27. content 为空字符串（user 显式传入）允许，但 complete() 仍要求非空响应
# ---------------------------------------------------------------------------
def test_empty_string_content_message_allowed():
    create_mock = mock.Mock(return_value=_make_completion("resp"))
    client, _ = _new_client(mock_create=create_mock)
    # 允许空 content
    out = client.complete(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": ""},
        ]
    )
    assert out == "resp"


# ---------------------------------------------------------------------------
# 28. 流式遇到空 content 字符串时跳过
# ---------------------------------------------------------------------------
def test_stream_skip_empty_string_content():
    chunks = [
        _make_chunk("你"),
        _make_chunk(""),
        _make_chunk("好"),
    ]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)

    out = list(client.stream([{"role": "user", "content": "q"}]))
    assert out == ["你", "好"]


# ---------------------------------------------------------------------------
# 29. 默认构造（不传 client）会读 Settings
# ---------------------------------------------------------------------------
def test_default_construct_with_key_creates_openai_client():
    """Mock OpenAI 构造，验证参数（base_url / api_key / timeout / max_retries）。"""
    s = _new_settings(
        minimax_api_key=FAKE_KEY,
        minimax_base_url="http://example/v1",
        minimax_timeout=30.0,
        minimax_max_retries=5,
    )
    with mock.patch("rag.llm_client.OpenAI") as openai_cls:
        client = MiniMaxLLMClient(settings=s)
    openai_cls.assert_called_once_with(
        api_key=FAKE_KEY,
        base_url="http://example/v1",
        timeout=30.0,
        max_retries=5,
    )
    assert client.model_name == "MiniMax-M3"


def test_default_construct_without_key_raises_configuration_error():
    s = _new_settings()  # 没有 key
    with pytest.raises(LLMConfigurationError) as ei:
        MiniMaxLLMClient(settings=s)
    assert "MINIMAX_API_KEY" in str(ei.value)


# ---------------------------------------------------------------------------
# 30. OpenAI 构造阶段 ImportError（典型：未安装 socksio）
# ---------------------------------------------------------------------------
def test_default_construct_import_error_propagates_safely():
    """当 OpenAI() 构造抛 ImportError（如缺少 socksio）时，客户端原样传播。

    客户端**不**包装 ImportError —— 这让上层（如连接测试脚本）能针对
    代理依赖缺失给出专门的提示。
    """
    s = _new_settings(minimax_api_key=FAKE_KEY)
    import_error = ImportError(
        "Using SOCKS proxy, but the 'socksio' package is not installed."
    )
    with mock.patch("rag.llm_client.OpenAI", side_effect=import_error):
        with pytest.raises(ImportError) as ei:
            MiniMaxLLMClient(settings=s)
    # 异常消息中不包含 Key
    assert FAKE_KEY not in str(ei.value)


def test_default_construct_import_error_does_not_log_key(caplog):
    """ImportError 路径下日志中也不应出现 Key。"""
    caplog.set_level(logging.DEBUG)
    s = _new_settings(minimax_api_key=FAKE_KEY)
    import_error = ImportError("simulated socksio missing")
    with mock.patch("rag.llm_client.OpenAI", side_effect=import_error):
        with pytest.raises(ImportError):
            MiniMaxLLMClient(settings=s)
    # 日志里不能出现测试 Key
    assert FAKE_KEY not in caplog.text


# ---------------------------------------------------------------------------
# 31. 注入 client 时不触发 OpenAI() 构造
# ---------------------------------------------------------------------------
def test_injected_client_bypass_construct():
    """注入 client 时，OpenAI() 不被调用，构造不会触发 ImportError。"""
    s = _new_settings(minimax_api_key=FAKE_KEY)
    fake_openai = mock.MagicMock(spec=OpenAI)
    with mock.patch("rag.llm_client.OpenAI") as openai_cls:
        client = MiniMaxLLMClient(settings=s, client=fake_openai)
    openai_cls.assert_not_called()
    assert client._client is fake_openai  # type: ignore[attr-defined]


# ===========================================================================
# ReasoningContentFilter 单元测试
# ===========================================================================
def _drive_filter(chunks: list[str]) -> str:
    """用一组 chunk 驱动过滤器，拼接所有 yield + finish 的输出。"""
    flt = ReasoningContentFilter()
    parts: list[str] = []
    for c in chunks:
        parts.extend(flt.feed(c))
    parts.extend(flt.finish())
    return "".join(parts)


# ---------------------------------------------------------------------------
# 1. 普通答案保持不变
# ---------------------------------------------------------------------------
def test_filter_plain_text_unchanged():
    assert _drive_filter(["hello world"]) == "hello world"
    assert _drive_filter(["你好世界"]) == "你好世界"
    assert _drive_filter(["", "hello"]) == "hello"


# ---------------------------------------------------------------------------
# 2. 完整 <think>...</think> 被移除
# ---------------------------------------------------------------------------
def test_filter_strip_complete_think_block():
    out = _drive_filter(["<think>some reasoning</think>answer"])
    assert out == "answer"
    assert "reasoning" not in out
    assert "<think>" not in out
    assert "</think>" not in out


# ---------------------------------------------------------------------------
# 3. 完整 <analysis>...</analysis> 被移除
# ---------------------------------------------------------------------------
def test_filter_strip_complete_analysis_block():
    out = _drive_filter(["<analysis>deep analysis</analysis>final"])
    assert out == "final"
    assert "analysis" not in out.lower() or "analysis" in "final".lower()  # ignore case
    assert "<analysis>" not in out
    assert "</analysis>" not in out


# ---------------------------------------------------------------------------
# 4. 推理标签位于答案开头
# ---------------------------------------------------------------------------
def test_filter_think_at_start():
    out = _drive_filter(["<think>hidden</think>visible"])
    assert out == "visible"


# ---------------------------------------------------------------------------
# 5. 推理标签位于答案中间
# ---------------------------------------------------------------------------
def test_filter_think_in_middle():
    out = _drive_filter(["before<think>hidden</think>after"])
    assert out == "beforeafter"


# ---------------------------------------------------------------------------
# 6. 多个推理区块
# ---------------------------------------------------------------------------
def test_filter_multiple_blocks():
    out = _drive_filter(
        [
            "<think>a</think>",
            "X",
            "<analysis>b</analysis>",
            "Y",
            "<think>c</think>Z",
        ]
    )
    assert out == "XYZ"


# ---------------------------------------------------------------------------
# 7. 标签跨多个流式 chunk
# ---------------------------------------------------------------------------
def test_filter_tag_split_across_chunks():
    out = _drive_filter(["<think>reas", "oning</think>a", "nswer"])
    assert out == "answer"


# ---------------------------------------------------------------------------
# 8. 开始标签本身跨 chunk
# ---------------------------------------------------------------------------
def test_filter_open_tag_split_across_chunks():
    out = _drive_filter(["<", "thin", "k>reasoning</think>", "real"])
    assert out == "real"


# ---------------------------------------------------------------------------
# 9. 结束标签本身跨 chunk
# ---------------------------------------------------------------------------
def test_filter_close_tag_split_across_chunks():
    out = _drive_filter(["<think>reasoning</", "thin", "k>real"])
    assert out == "real"


# ---------------------------------------------------------------------------
# 10. 未闭合标签不会泄漏
# ---------------------------------------------------------------------------
def test_filter_unclosed_tag_discarded():
    out = _drive_filter(["<think>never closed", "more reasoning"])
    assert out == ""
    assert "reasoning" not in out


def test_filter_unclosed_open_only():
    out = _drive_filter(["<think>only open no close"])
    assert out == ""


def test_filter_partial_close_tag_at_end_discarded():
    out = _drive_filter(["<think>r</think>", "a<think>b</", "thin"])
    assert out == "a"  # 最后未闭合部分丢弃


# ---------------------------------------------------------------------------
# 11. reasoning_content 独立字段被忽略（非流式）
# ---------------------------------------------------------------------------
def test_non_stream_ignores_reasoning_field():
    """message.reasoning 存在时，complete() 不应使用其内容。"""
    # 构造一个带 reasoning 字段的 fake message
    msg = ChatCompletionMessage(
        role="assistant",
        content="<answer>用户可见答案</answer>",
    )
    # 模拟 reasoning 字段（ChatCompletionMessage 不一定支持，但通过 setattr 注入）
    object.__setattr__(msg, "reasoning", "内部推理：这是思考内容 SECRET_REASONING")
    resp = ChatCompletion(
        id="cmpl-x",
        choices=[
            CompletionChoice(
                index=0,
                message=msg,
                finish_reason="stop",
            )
        ],
        created=0,
        model="MiniMax-M3",
        object="chat.completion",
    )
    create_mock = mock.Mock(return_value=resp)
    client, _ = _new_client(mock_create=create_mock)
    out = client.complete([{"role": "user", "content": "q"}])
    # 内部推理内容不能出现在返回中
    assert "SECRET_REASONING" not in out
    assert "内部推理" not in out


# ---------------------------------------------------------------------------
# 12. 清理后为空时抛异常
# ---------------------------------------------------------------------------
def test_non_stream_only_think_raises_format_error():
    """非流式 content 只包含 <think>...</think>，complete() 应抛 LLMResponseFormatError。"""
    resp = _make_completion("<think>all hidden</think>")
    create_mock = mock.Mock(return_value=resp)
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMResponseFormatError) as ei:
        client.complete([{"role": "user", "content": "q"}])
    assert "为空" in str(ei.value)


def test_stream_only_think_raises_format_error():
    """流式 content 只包含 <think>...</think>，stream() 应抛 LLMResponseFormatError。"""
    chunks = [
        _make_chunk("<think>"),
        _make_chunk("hidden"),
        _make_chunk("</think>"),
    ]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMResponseFormatError):
        list(client.stream([{"role": "user", "content": "q"}]))


def test_stream_unclosed_think_raises_format_error():
    """流式未闭合的 think，stream() 抛出（实际被捕获到空时抛）。"""
    chunks = [
        _make_chunk("<think>hidden"),
        _make_chunk("more"),
    ]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)
    with pytest.raises(LLMResponseFormatError):
        list(client.stream([{"role": "user", "content": "q"}]))


# ---------------------------------------------------------------------------
# 13. 合法普通文本中不相关的尖括号内容不被误删
# ---------------------------------------------------------------------------
def test_filter_unrelated_angle_brackets_preserved():
    out = _drive_filter(["1 < 2 and 3 > 2"])
    assert out == "1 < 2 and 3 > 2"


def test_filter_html_like_text_preserved():
    out = _drive_filter(["the <a> tag and <b/> tag"])
    assert out == "the <a> tag and <b/> tag"


def test_filter_pseudo_tag_not_real_preserved():
    """<thinker> 不是 <think>（缺 >），应保留。"""
    out = _drive_filter(["<thinker>foo</thinker>"])
    assert out == "<thinker>foo</thinker>"


# ---------------------------------------------------------------------------
# 14. 日志和测试输出不包含被过滤的思考内容
# ---------------------------------------------------------------------------
def test_logs_dont_contain_filtered_reasoning(caplog):
    caplog.set_level(logging.DEBUG)
    secret = "DO_NOT_LOG_SECRET_REASONING"
    resp = _make_completion(f"<think>{secret}</think>real answer")
    create_mock = mock.Mock(return_value=resp)
    client, _ = _new_client(mock_create=create_mock)
    out = client.complete([{"role": "user", "content": "q"}])
    assert out == "real answer"
    assert secret not in caplog.text
    assert secret not in out


def test_stream_logs_dont_contain_filtered_reasoning(caplog):
    caplog.set_level(logging.DEBUG)
    secret = "STREAM_SECRET_REASONING"
    chunks = [
        _make_chunk("<think>"),
        _make_chunk(secret),
        _make_chunk("</think>"),
        _make_chunk("visible"),
    ]
    create_mock = mock.Mock(return_value=iter(chunks))
    client, _ = _new_client(mock_create=create_mock)
    out = list(client.stream([{"role": "user", "content": "q"}]))
    assert "".join(out) == "visible"
    assert secret not in caplog.text


# ---------------------------------------------------------------------------
# 15. 非流式 sanitize_model_content 单点测试
# ---------------------------------------------------------------------------
def test_sanitize_function_basic():
    assert sanitize_model_content("hello") == "hello"
    assert sanitize_model_content("<think>r</think>ans") == "ans"
    assert sanitize_model_content("<think>r") == ""


def test_sanitize_function_non_string_raises():
    with pytest.raises(LLMResponseFormatError):
        sanitize_model_content(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 16. 过滤器状态查询
# ---------------------------------------------------------------------------
def test_filter_state_property():
    flt = ReasoningContentFilter()
    assert flt.state == "NORMAL"
    flt.feed("<think>x")
    assert flt.state == "IN_THINK"
    flt.feed("</think>")
    assert flt.state == "NORMAL"
    flt.feed("<analysis>x")
    assert flt.state == "IN_ANALYSIS"
    flt.feed("</analysis>")
    assert flt.state == "NORMAL"


# ---------------------------------------------------------------------------
# 17. 多次 finish 调用幂等
# ---------------------------------------------------------------------------
def test_filter_finish_idempotent():
    """finish 第一次返回残余缓冲，第二次返回 []（已 flush）。这是预期语义。"""
    flt = ReasoningContentFilter()
    # 喂入会触发 hold 的内容（含 '<' 但未形成完整标签）
    flt.feed("a<")
    a = flt.finish()
    b = flt.finish()
    # 第一次 finish 把 hold 中的残余返回；第二次返回空（已清空）
    assert a == ["<"]
    assert b == []


# ---------------------------------------------------------------------------
# 18. feed 非字符串抛 TypeError
# ---------------------------------------------------------------------------
def test_filter_feed_rejects_non_string():
    flt = ReasoningContentFilter()
    with pytest.raises(TypeError):
        flt.feed(123)  # type: ignore[arg-type]
