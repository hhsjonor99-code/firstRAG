"""SiliconFlow Embedding 提供商实现。

封装 `https://api.siliconflow.cn/v1/embeddings` 接口，调用
``Qwen/Qwen3-Embedding-4B``（默认）将文本编码为 ``(N, 1024)`` 的 float32 向量。

关键约束：

1. **不记录 Key**：日志中只记录「Authorization 头存在/缺失」与 HTTP 状态码，
   不出现 Key 字符；全局 ``APIKeyRedactionFilter`` 兜底。
2. **批量调用 + 顺序恢复**：按 ``batch_size`` 切分串行请求；服务端可能乱序
   返回，因此必须按 ``data[i].index`` 还原成输入顺序。
3. **异常分级**：401/403 不重试立即抛 :class:`EmbeddingAuthError`；429、5xx
   与 Timeout/ConnectionError 通过 tenacity 重试；其它 4xx 立即抛
   :class:`EmbeddingError`。
4. **延迟 Key 校验**：`__init__` 不立即要求 Key；调用 ``embed_*`` 时才通过
   :meth:`config.settings.Settings.require_siliconflow_key` 校验。
5. **向量归一化**：所有向量做 L2 归一化，便于余弦相似度检索。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional
from urllib.parse import urljoin

import numpy as np
import requests
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.logging import get_logger
from config.settings import Settings, get_settings

from .embedding_provider import (
    EmbeddingError,
    validate_non_empty_texts,
)


# ---------------------------------------------------------------------------
# 异常层次
# ---------------------------------------------------------------------------
class EmbeddingAuthError(EmbeddingError):
    """鉴权失败（HTTP 401 / 403）。不重试。"""


class EmbeddingRateLimitError(EmbeddingError):
    """触发限流（HTTP 429）。可重试。"""


class EmbeddingTimeoutError(EmbeddingError):
    """请求超时或连接错误。可重试。"""


class EmbeddingServerError(EmbeddingError):
    """服务端错误（HTTP 5xx）。可重试。"""


class EmbeddingResponseFormatError(EmbeddingError):
    """响应 JSON 格式不符合预期。"""


class EmbeddingConfigurationError(EmbeddingError):
    """配置错误（如缺 Key）。"""


# ---------------------------------------------------------------------------
# 实现
# ---------------------------------------------------------------------------
class SiliconFlowEmbeddingProvider:
    """SiliconFlow Embedding 提供商实现。

    :param settings: 全局配置；缺省为 :func:`config.settings.get_settings`。
    :param session: 可选 ``requests.Session``，便于测试注入 mock adapter。
    """

    # 可重试异常：429 / 5xx / Timeout / ConnectionError
    _RETRYABLE_EXCEPTIONS = (
        EmbeddingRateLimitError,
        EmbeddingServerError,
        EmbeddingTimeoutError,
        requests.Timeout,
        requests.ConnectionError,
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._session = session or requests.Session()
        self._log = get_logger("rag.siliconflow_embeddings")
        # 防止并发调用混用同一个 Provider 时日志交错 —— 但仍允许并发调用，
        # 只是各自记录各自的耗时。

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._settings.siliconflow_embedding_model

    @property
    def dimensions(self) -> int:
        return int(self._settings.siliconflow_embedding_dimensions)

    @property
    def batch_size(self) -> int:
        return max(1, int(self._settings.siliconflow_embedding_batch_size))

    @property
    def timeout(self) -> int:
        return max(1, int(self._settings.siliconflow_timeout))

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """批量将多条文本编码为向量。

        :param texts: 待编码文本列表；**不允许为空**。
        :returns: ``shape == (len(texts), self.dimensions)``、``dtype ==
            np.float32``、每行 L2 范数 ≈ 1 的二维数组。
        """
        materialized = validate_non_empty_texts(texts)
        if not all(isinstance(t, str) for t in materialized):
            raise EmbeddingError("Embedding 输入包含非 str 元素。")

        key = self._require_api_key()
        endpoint = self._build_endpoint()
        batch_size = self.batch_size

        all_vectors: list[np.ndarray] = []
        text_count = len(materialized)
        batch_count = (text_count + batch_size - 1) // batch_size

        self._log.info(
            "Embedding 请求开始：model=%s dim=%d 输入=%d 批次=%d batch_size=%d",
            self.model_name,
            self.dimensions,
            text_count,
            batch_count,
            batch_size,
        )

        for start in range(0, text_count, batch_size):
            batch = materialized[start : start + batch_size]
            batch_index = start // batch_size
            self._log.debug(
                "Embedding 批次 %d/%d：size=%d", batch_index + 1, batch_count, len(batch)
            )
            vectors = self._embed_one_batch(
                key=key,
                endpoint=endpoint,
                batch=batch,
                batch_index=batch_index,
                batch_count=batch_count,
            )
            all_vectors.append(vectors)

        result = np.vstack(all_vectors).astype(np.float32, copy=False)

        # 最终一致性校验（即便每批都校验过，仍做一次兜底）
        if result.shape != (text_count, self.dimensions):
            raise EmbeddingResponseFormatError(
                f"最终向量 shape={result.shape} 与期望 (N={text_count}, "
                f"dim={self.dimensions}) 不一致。"
            )
        if not np.isfinite(result).all():
            raise EmbeddingResponseFormatError("向量包含 NaN 或 Inf。")

        # L2 归一化（每行）
        self._l2_normalize_inplace(result)

        self._log.info(
            "Embedding 请求完成：输入=%d 返回 shape=%s dtype=%s",
            text_count,
            tuple(result.shape),
            result.dtype,
        )
        return result

    def embed_query(self, text: str) -> np.ndarray:
        """单条查询向量化，``shape == (1, dimensions)``."""
        if not isinstance(text, str):
            raise EmbeddingError("embed_query 输入必须为 str。")
        # 复用 embed_documents，保持 batch 处理、日志与归一化逻辑一致
        arr = self.embed_documents([text])
        if arr.shape != (1, self.dimensions):
            raise EmbeddingResponseFormatError(
                f"embed_query 返回 shape={arr.shape}，期望 (1, {self.dimensions})。"
            )
        return arr

    # ------------------------------------------------------------------
    # 内部：单批请求
    # ------------------------------------------------------------------
    def _embed_one_batch(
        self,
        *,
        key: str,
        endpoint: str,
        batch: list[str],
        batch_index: int,
        batch_count: int,
    ) -> np.ndarray:
        """发起一次 embeddings 请求（含重试）；返回 ``(len(batch), dim)`` float32 数组。"""
        payload = {
            "model": self.model_name,
            "input": batch,
            "encoding_format": "float",
            "dimensions": self.dimensions,
        }
        # 头中只携带 Bearer；不构造任何包含 Key 字符的额外日志字段。
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        # tenacity 重试：最多 3 次，指数退避 0.5s/1s/2s（最多 4s）
        retryer = Retrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception_type(self._RETRYABLE_EXCEPTIONS),
        )

        last_exc: Optional[BaseException] = None
        try:
            for attempt in retryer:
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        self._log.debug(
                            "Embedding 重试：批次 %d/%d 第 %d 次",
                            batch_index + 1,
                            batch_count,
                            attempt.retry_state.attempt_number,
                        )
                    response = self._post_with_timeout(endpoint, headers, payload)
                    self._handle_status(response, batch_index, batch_count, len(batch))
                    vectors = self._parse_response(
                        response, expected_count=len(batch), batch_index=batch_index
                    )
                    return vectors
        except RetryError as exc:  # tenacity 兜底；reraise=True 时不会再触发
            last_exc = exc

        # 理论上来不到这里（reraise=True）；但为类型安全保留兜底。
        if last_exc is not None:
            raise EmbeddingError(f"Embedding 重试失败：{last_exc}") from last_exc
        raise EmbeddingError("Embedding 重试失败：未知原因。")

    def _post_with_timeout(
        self, endpoint: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> requests.Response:
        """包装 ``requests.post``，把 Timeout / ConnectionError 转为统一异常。"""
        try:
            return self._session.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise EmbeddingTimeoutError(
                f"Embedding 请求超时（>{self.timeout}s）。"
            ) from exc
        except requests.ConnectionError as exc:
            raise EmbeddingTimeoutError(
                f"Embedding 连接错误：{type(exc).__name__}"
            ) from exc

    def _handle_status(
        self,
        response: requests.Response,
        batch_index: int,
        batch_count: int,
        expected_count: int,
    ) -> None:
        """根据 HTTP 状态码决定抛错或继续。"""
        status = response.status_code
        if status in (401, 403):
            self._log.warning(
                "Embedding 鉴权失败：批次 %d/%d status=%d",
                batch_index + 1,
                batch_count,
                status,
            )
            raise EmbeddingAuthError(
                f"Embedding 鉴权失败（HTTP {status}）：请检查 SILICONFLOW_API_KEY。"
            )
        if status == 429:
            self._log.warning(
                "Embedding 触发限流：批次 %d/%d status=429，将重试",
                batch_index + 1,
                batch_count,
            )
            raise EmbeddingRateLimitError("Embedding 触发限流（HTTP 429）。")
        if status in (500, 502, 503, 504):
            self._log.warning(
                "Embedding 服务端错误：批次 %d/%d status=%d，将重试",
                batch_index + 1,
                batch_count,
                status,
            )
            raise EmbeddingServerError(f"Embedding 服务端错误（HTTP {status}）。")
        if 400 <= status < 500:
            # 其它 4xx：不重试，抛通用 EmbeddingError
            self._log.warning(
                "Embedding 客户端错误：批次 %d/%d status=%d body_len=%d",
                batch_index + 1,
                batch_count,
                status,
                len(response.text or ""),
            )
            raise EmbeddingError(
                f"Embedding 请求被拒绝（HTTP {status}）。"
            )
        # 2xx 通过；其它（1xx/3xx）按异常处理
        if not (200 <= status < 300):
            raise EmbeddingError(f"Embedding 收到非预期状态码：HTTP {status}。")

    def _parse_response(
        self,
        response: requests.Response,
        *,
        expected_count: int,
        batch_index: int,
    ) -> np.ndarray:
        """解析 embeddings 响应，按 ``data[i].index`` 恢复顺序。"""
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingResponseFormatError(
                f"Embedding 响应不是合法 JSON：{type(exc).__name__}。"
            ) from exc

        if not isinstance(data, dict):
            raise EmbeddingResponseFormatError(
                f"Embedding 响应顶层不是 object，得到 {type(data).__name__}。"
            )

        items = data.get("data")
        if not isinstance(items, list):
            raise EmbeddingResponseFormatError("Embedding 响应缺少 'data' 列表字段。")
        if len(items) != expected_count:
            raise EmbeddingResponseFormatError(
                f"Embedding 返回数量 {len(items)} 与输入数量 {expected_count} 不一致。"
            )

        # 按 index 排序恢复顺序
        ordered_vectors: list[Optional[list[float]]] = [None] * expected_count  # type: ignore[list-item]
        for item in items:
            if not isinstance(item, dict):
                raise EmbeddingResponseFormatError(
                    "Embedding 响应 data 项不是 object。"
                )
            idx = item.get("index")
            emb = item.get("embedding")
            if not isinstance(idx, int):
                raise EmbeddingResponseFormatError(
                    "Embedding 响应缺少合法 'index' 字段。"
                )
            if not isinstance(emb, list):
                raise EmbeddingResponseFormatError(
                    "Embedding 响应缺少合法 'embedding' 列表字段。"
                )
            if not (0 <= idx < expected_count):
                raise EmbeddingResponseFormatError(
                    f"Embedding 响应 index={idx} 超出范围 [0, {expected_count})。"
                )
            if ordered_vectors[idx] is not None:
                raise EmbeddingResponseFormatError(
                    f"Embedding 响应 index={idx} 重复。"
                )
            ordered_vectors[idx] = emb

        if any(v is None for v in ordered_vectors):
            raise EmbeddingResponseFormatError(
                "Embedding 响应未能覆盖全部 index。"
            )

        # 维度校验
        dim = self.dimensions
        for i, emb in enumerate(ordered_vectors):  # type: ignore[assignment]
            assert emb is not None
            if len(emb) != dim:
                raise EmbeddingResponseFormatError(
                    f"Embedding 第 {i} 条维度 {len(emb)} 与期望 {dim} 不一致。"
                )

        # 构造 numpy 数组
        arr = np.asarray(ordered_vectors, dtype=np.float32)

        # 数值合法性校验
        if not np.isfinite(arr).all():
            raise EmbeddingResponseFormatError("Embedding 返回向量包含 NaN 或 Inf。")

        # L2 归一化
        self._l2_normalize_inplace(arr)

        self._log.debug(
            "Embedding 批次 %d 完成：count=%d dim=%d dtype=%s",
            batch_index + 1,
            arr.shape[0],
            arr.shape[1],
            arr.dtype,
        )
        return arr

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _require_api_key(self) -> str:
        """调用时延迟校验 API Key。"""
        try:
            return self._settings.require_siliconflow_key()
        except Exception as exc:  # MissingAPIKeyError 是 RuntimeError 子类
            raise EmbeddingConfigurationError(str(exc)) from exc

    def _build_endpoint(self) -> str:
        """拼接 ``base_url + /embeddings``，避免重复 /v1 或重复斜杠。"""
        base = (self._settings.siliconflow_base_url or "").rstrip("/")
        # 如果 base 已经以 /embeddings 结尾，则不再追加
        if base.endswith("/embeddings"):
            return base
        return urljoin(base + "/", "embeddings")

    @staticmethod
    def _l2_normalize_inplace(arr: np.ndarray) -> None:
        """对二维 float32 数组逐行做 L2 归一化；原地修改。

        全零行归一化后仍为全零（避免产生 NaN）。
        """
        if arr.ndim != 2:
            raise EmbeddingError(f"归一化要求 2D 数组，得到 ndim={arr.ndim}。")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # 用 np.where 防止 0 除产生 NaN；零向量保持为 0
        safe_norms = np.where(norms == 0, 1.0, norms)
        arr /= safe_norms
        # 显式把零向量行归零
        if np.any(norms == 0):
            arr *= (norms != 0).astype(arr.dtype)


# 一个简单的线程级重入锁占位，便于未来扩展（当前未使用）
_ = threading.Lock  # noqa: F841