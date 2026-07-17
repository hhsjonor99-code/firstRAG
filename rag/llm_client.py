"""MiniMax OpenAI 兼容 LLM 客户端。

封装 :class:`openai.OpenAI`，通过 ``base_url`` 指向 ``https://api.minimaxi.com/v1``，
使用 OpenAI 兼容的 ``chat.completions.create`` 接口调用 ``MiniMax-M3``。

设计原则：

1. **延迟 Key 校验**：`Settings()` 缺 Key 时仍可正常加载；只有实际
   :class:`MiniMaxLLMClient` 构造或调用 ``complete`` / ``stream`` 时才通过
   :meth:`config.settings.Settings.require_minimax_key` 校验。
2. **OpenAI 兼容**：第一版仅使用标准 OpenAI 兼容接口；
   **不**传 ``reasoning_split`` 或其它 MiniMax 私有扩展字段。
3. **流式 / 非流式**：
   - ``complete`` 返回完整 ``content``；空内容抛 :class:`LLMResponseFormatError`。
   - ``stream`` 是惰性生成器：逐 chunk 读取 ``delta.content``，跳过 ``None``
     与空字符串；正常结束即停止；中途异常转换为对应项目异常。
4. **消息校验**：``messages`` 不能为空；每条消息必须包含合法
   ``role ∈ {"system", "user", "assistant"}`` 与字符串 ``content``。
5. **PromptBuilder 解耦**：本模块不构造 RAG Prompt，也不在内部拼接任何
   业务上下文；调用方负责先用 :class:`rag.prompt_builder.PromptBuilder`
   构造好 messages 再传入。
6. **日志安全**：只记录模型名 / 消息数量 / 输入字符总数 / 是否流式 /
   耗时 / HTTP 或 SDK 错误类型；**不**记录任何 Key、请求头、完整响应体、
   完整 Prompt 或回答内容。``APIKeyRedactionFilter`` 兜底。

异常层次（继承关系）：

::

    LLMError
    ├── LLMConfigurationError
    ├── LLMAuthenticationError
    ├── LLMRateLimitError
    ├── LLMTimeoutError
    ├── LLMConnectionError
    ├── LLMServerError
    └── LLMResponseFormatError

OpenAI SDK 异常映射：

- ``openai.AuthenticationError``  → :class:`LLMAuthenticationError`
- ``openai.RateLimitError``      → :class:`LLMRateLimitError`
- ``openai.APITimeoutError``     → :class:`LLMTimeoutError`
- ``openai.APIConnectionError``  → :class:`LLMConnectionError`
- ``openai.APIStatusError``：
    - 5xx                       → :class:`LLMServerError`
    - 其它状态                  → :class:`LLMError`
- 响应 ``choices`` 为空 /
  ``message.content`` 为空或结构异常 → :class:`LLMResponseFormatError`

推理内容隔离（v1.1）：

- 已知部分模型会返回两类推理内容：
    1. **独立字段**（如 ``message.reasoning`` / ``delta.reasoning``）：本客户端
       **不**读取这些字段，调用方拿到的只有面向用户的 ``content``。
    2. **嵌入式标签**（``<think>...</think>`` / ``<analysis>...</analysis>``）：
       通过 :class:`ReasoningContentFilter` 在客户端内部过滤。
- ``complete()`` 与 ``stream()`` 共享同一过滤规则；
  流式过滤器是有状态的状态机，正确处理跨 chunk 的标签切分，
  不会把未闭合的推理内容泄漏给调用方。
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from config.logging import get_logger
from config.settings import MissingAPIKeyError, Settings, get_settings


# ---------------------------------------------------------------------------
# 异常层次
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """LLM 调用相关错误的基类。"""


class LLMConfigurationError(LLMError):
    """配置错误（典型：未设置 ``MINIMAX_API_KEY``）。"""


class LLMAuthenticationError(LLMError):
    """鉴权失败（HTTP 401 / 403）。"""


class LLMRateLimitError(LLMError):
    """触发限流（HTTP 429）。"""


class LLMTimeoutError(LLMError):
    """请求超时。"""


class LLMConnectionError(LLMError):
    """网络连接错误。"""


class LLMServerError(LLMError):
    """服务端错误（HTTP 5xx）。"""


class LLMResponseFormatError(LLMError):
    """响应结构不符合预期（如 choices 为空、content 为空）。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    """校验 messages 列表，返回规范化结果（浅拷贝，元素为内置 dict）。

    - 不为 ``None``，必须为 ``list``，且非空。
    - 每条元素必须为 ``dict``，含 ``role`` 与 ``content``。
    - ``role`` ∈ ``{"system", "user", "assistant"}``。
    - ``content`` 必须是 ``str``（允许空字符串，但要求显式存在）。
    """
    if messages is None:
        raise LLMResponseFormatError("messages 不能为 None。")
    if not isinstance(messages, list):
        raise LLMResponseFormatError(
            f"messages 必须是 list 类型，得到 {type(messages).__name__}。"
        )
    if len(messages) == 0:
        raise LLMResponseFormatError("messages 列表不能为空。")

    validated: list[dict[str, str]] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise LLMResponseFormatError(
                f"messages[{idx}] 必须是 dict，得到 {type(msg).__name__}。"
            )
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in _ALLOWED_ROLES:
            raise LLMResponseFormatError(
                f"messages[{idx}].role 必须是 "
                f"{sorted(_ALLOWED_ROLES)} 之一，得到 {role!r}。"
            )
        if not isinstance(content, str):
            raise LLMResponseFormatError(
                f"messages[{idx}].content 必须是 str 类型，"
                f"得到 {type(content).__name__}。"
            )
        validated.append({"role": role, "content": content})
    return validated


# ---------------------------------------------------------------------------
# 推理内容过滤
# ---------------------------------------------------------------------------
# 支持的嵌入式推理标签：开标签 -> 闭标签
_REASONING_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<analysis>", "</analysis>"),
)
# 持有缓冲长度：开标签最长长度 - 1（用于处理跨 chunk 切分）
_MAX_OPEN_TAG_LEN = max(len(p[0]) for p in _REASONING_TAG_PAIRS)
# 持有缓冲长度：闭标签最长长度 - 1
_MAX_CLOSE_TAG_LEN = max(len(p[1]) for p in _REASONING_TAG_PAIRS)

_STATE_NORMAL = "NORMAL"
_STATE_IN_THINK = "IN_THINK"
_STATE_IN_ANALYSIS = "IN_ANALYSIS"


class ReasoningContentFilter:
    """有状态的流式推理内容过滤器。

    用于从模型 ``content`` 中过滤掉 ``<think>...</think>`` 与
    ``<analysis>...</analysis>`` 等成对出现的标签，**仅**保留面向用户的答案。

    行为规则：

    - 普通文本立即 ``yield``（通过 :meth:`feed` 返回的 ``list[str]``）。
    - 遇到 ``<think>`` 切换到「丢弃」状态，直到 ``</think>`` 恢复输出。
    - 标签可能跨越多个 chunk：过滤器持有最末 ``max_tag_len - 1`` 个字符作为
      「可能未完成的标签前缀」，绝不会把不完整的标签或思考内容短暂
      ``yield`` 出去。
    - 状态机结束（:meth:`finish`）时若仍处于丢弃状态，未闭合的推理区
      **被静默丢弃**；若处于 ``NORMAL``，剩余缓冲会作为最终答案片段返回。
    - 不会缓存完整回答后再一次性返回；普通文本即时产出。
    - 不依赖任何正则匹配；只做精确的成对标签匹配。
    """

    __slots__ = ("_state", "_buf")

    def __init__(self) -> None:
        self._state: str = _STATE_NORMAL
        self._buf: str = ""

    @property
    def state(self) -> str:
        """当前状态：``NORMAL`` / ``IN_THINK`` / ``IN_ANALYSIS``。"""
        return self._state

    def feed(self, chunk: str) -> list[str]:
        """喂入一段 ``content`` 片段，返回可对外 ``yield`` 的安全文本列表。

        :param chunk: 来自 ``delta.content`` 的字符串片段；允许为 ``""``。
        :returns: 顺序产出的、面向用户的纯文本片段（不含任何 ``<think>`` /
            ``<analysis>`` 标签或内部思考内容）。
        """
        if not isinstance(chunk, str):
            raise TypeError(f"feed() 接收的 chunk 必须是 str，得到 {type(chunk).__name__}。")
        if not chunk:
            return []
        self._buf += chunk
        out: list[str] = []

        # 在 NORMAL 与 IN_* 之间循环，直到无法再前进
        while True:
            if self._state == _STATE_NORMAL:
                # 找最早出现的开标签
                earliest_pos = -1
                earliest_tag = ""
                for open_tag, _ in _REASONING_TAG_PAIRS:
                    idx = self._buf.find(open_tag)
                    if idx != -1 and (earliest_pos == -1 or idx < earliest_pos):
                        earliest_pos = idx
                        earliest_tag = open_tag
                if earliest_pos != -1:
                    # 把标签前的安全文本输出
                    if earliest_pos > 0:
                        out.append(self._buf[:earliest_pos])
                    # 消费开标签，切换到丢弃状态
                    self._buf = self._buf[earliest_pos + len(earliest_tag):]
                    self._state = (
                        _STATE_IN_THINK if earliest_tag == "<think>" else _STATE_IN_ANALYSIS
                    )
                    # 继续，可能还有更多内容
                    continue
                # 没有完整标签：找最早的 '<'。若完全没有 '<'，buffer 一定安全，立即 yield
                lt_pos = self._buf.find("<")
                if lt_pos == -1:
                    if self._buf:
                        out.append(self._buf)
                        self._buf = ""
                    break
                # 有 '<' 但还没形成完整标签：yield '<' 之前的内容
                if lt_pos > 0:
                    out.append(self._buf[:lt_pos])
                self._buf = self._buf[lt_pos:]
                # 现在 buffer 以 '<' 开头。如果太长（> max_open_tag_len），
                # 保留最后 max_open_tag_len - 1 字符作为「可能未完成前缀」
                if len(self._buf) > _MAX_OPEN_TAG_LEN:
                    hold_len = _MAX_OPEN_TAG_LEN - 1
                    out.append(self._buf[: len(self._buf) - hold_len])
                    self._buf = self._buf[-hold_len:]
                break
            else:
                # IN_THINK / IN_ANALYSIS：找对应的闭标签
                close_tag = (
                    "</think>" if self._state == _STATE_IN_THINK else "</analysis>"
                )
                idx = self._buf.find(close_tag)
                if idx != -1:
                    # 消费到闭标签末尾，恢复 NORMAL
                    self._buf = self._buf[idx + len(close_tag):]
                    self._state = _STATE_NORMAL
                    continue
                # 未找到：留 hold_len 个字符
                hold_len = len(close_tag) - 1
                if len(self._buf) > hold_len:
                    self._buf = self._buf[-hold_len:]
                break

        return out

    def finish(self) -> list[str]:
        """流式结束时调用，吐出剩余的安全文本（若有）。

        - 若状态为 ``NORMAL``：剩余缓冲作为最终片段返回。
        - 若状态仍为 ``IN_THINK`` / ``IN_ANALYSIS``：未闭合的推理区**丢弃**。
        """
        out: list[str] = []
        if self._state == _STATE_NORMAL and self._buf:
            out.append(self._buf)
        # 其它状态：丢弃残余缓冲
        self._buf = ""
        self._state = _STATE_NORMAL
        return out


def sanitize_model_content(content: str) -> str:
    """对非流式 ``message.content`` 做单次推理内容清理。

    与 :class:`ReasoningContentFilter` 使用相同的过滤规则。

    :returns: 清理后的用户可见内容。
    """
    if not isinstance(content, str):
        raise LLMResponseFormatError(
            f"sanitize 输入必须是 str，得到 {type(content).__name__}。"
        )
    flt = ReasoningContentFilter()
    parts: list[str] = flt.feed(content)
    parts.extend(flt.finish())
    return "".join(parts)


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
class MiniMaxLLMClient:
    """MiniMax OpenAI 兼容 LLM 客户端。

    :param settings: 全局配置；缺省为 :func:`config.settings.get_settings`。
    :param client: 可选 :class:`openai.OpenAI` 客户端，便于测试注入 mock。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._log = get_logger("rag.llm_client")
        if client is not None:
            # 测试场景：外部注入；构造时不强制要求 Key。
            self._client = client
        else:
            # 真实场景：构造时延迟校验 Key；缺 Key 时抛 LLMConfigurationError。
            key = self._require_api_key()
            self._client = OpenAI(
                api_key=key,
                base_url=self._settings.minimax_base_url,
                timeout=float(self._settings.minimax_timeout),
                max_retries=int(self._settings.minimax_max_retries),
            )

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._settings.minimax_model

    @property
    def model(self) -> str:
        """与 :attr:`model_name` 等价；保留便于不同调用方习惯。"""
        return self.model_name

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """非流式调用，返回模型 ``content`` 字符串。

        :param messages: 已构造好的 messages 列表。
        :param temperature: 覆盖默认温度；``None`` 表示使用 Settings 默认值。
        :param max_tokens: 覆盖默认最大 token；``None`` 表示不传给 API。
        :raises LLMError: 全部错误的基类。
        """
        validated = _validate_messages(messages)
        # 使用注入的 client 时也要保证 Key 已配置（用于真实调用）
        self._ensure_key_for_injected_client()

        temp = self._resolve_temperature(temperature)
        max_tok = max_tokens if max_tokens is not None else self._settings.minimax_max_tokens

        total_chars = sum(len(m["content"]) for m in validated)
        self._log.info(
            "LLM 非流式请求开始：model=%s 消息数=%d 输入字符=%d",
            self.model_name,
            len(validated),
            total_chars,
        )
        started = time.perf_counter()
        try:
            kwargs = self._build_request_kwargs(
                messages=validated,
                stream=False,
                temperature=temp,
                max_tokens=max_tok,
            )
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            mapped = self._map_exception(exc)
            self._log_failure(mapped, elapsed)
            raise mapped from exc

        elapsed = time.perf_counter() - started
        content = self._extract_non_stream_content(response)
        # 推理内容隔离：剥离嵌入式 <think> / <analysis> 标签
        cleaned = sanitize_model_content(content)
        if not cleaned:
            # 即便 raw content 非空，过滤后可能为空（模型只输出推理或未闭合）
            raise LLMResponseFormatError(
                "LLM 响应在过滤推理内容后为空（仅包含 <think> / <analysis> 等标签）。"
            )
        self._log.info(
            "LLM 非流式请求完成：model=%s 耗时=%.3fs",
            self.model_name,
            elapsed,
        )
        return cleaned

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """流式调用，返回字符串片段的惰性迭代器。

        - 逐 chunk 读取 ``choices[0].delta.content``。
        - 跳过 ``choices`` 为空 / ``delta.content`` 为 ``None`` 或空字符串的 chunk。
        - 正常结束即停止（**不**额外发送结束信号）。
        - 中途异常转换为项目异常，迭代器立即终止。
        """
        validated = _validate_messages(messages)
        self._ensure_key_for_injected_client()

        temp = self._resolve_temperature(temperature)
        max_tok = max_tokens if max_tokens is not None else self._settings.minimax_max_tokens

        total_chars = sum(len(m["content"]) for m in validated)
        self._log.info(
            "LLM 流式请求开始：model=%s 消息数=%d 输入字符=%d",
            self.model_name,
            len(validated),
            total_chars,
        )
        started = time.perf_counter()
        try:
            kwargs = self._build_request_kwargs(
                messages=validated,
                stream=True,
                temperature=temp,
                max_tokens=max_tok,
            )
            response_iter = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            mapped = self._map_exception(exc)
            self._log_failure(mapped, elapsed)
            raise mapped from exc

        # 返回生成器：调用方 for-loop 逐段获取
        return self._iter_stream_chunks(response_iter, started)

    # ------------------------------------------------------------------
    # 内部：流式 chunk 迭代
    # ------------------------------------------------------------------
    def _iter_stream_chunks(
        self,
        response_iter: Any,
        started: float,
    ) -> Iterator[str]:
        """惰性产出 ``delta.content``；中途异常映射为项目异常。

        推理内容隔离：
        - 通过 :class:`ReasoningContentFilter` 过滤 ``<think>...</think>`` 与
          ``<analysis>...</analysis>`` 标签；
        - 普通文本立即 yield，标签 / 思考内容永不出现在 yield 序列中；
        - 流结束时若过滤器未闭合推理区，安全丢弃。
        - 整个流 yield 的内容若最终为空，抛 :class:`LLMResponseFormatError`。
        """
        flt = ReasoningContentFilter()
        total_chars = 0
        try:
            for chunk in response_iter:
                if not getattr(chunk, "choices", None):
                    # 某些 chunk（首/末/usage 段）可能没有 choices；安全跳过
                    continue
                try:
                    delta = chunk.choices[0].delta
                except (IndexError, AttributeError, TypeError):
                    continue
                content = getattr(delta, "content", None)
                if content is None:
                    continue
                if not isinstance(content, str):
                    # 异常结构：直接报格式错误
                    raise LLMResponseFormatError(
                        f"流式 delta.content 不是 str，得到 {type(content).__name__}。"
                    )
                if content == "":
                    continue
                for safe_text in flt.feed(content):
                    total_chars += len(safe_text)
                    yield safe_text
            # 流结束：flush 残余缓冲
            for safe_text in flt.finish():
                total_chars += len(safe_text)
                yield safe_text
        except LLMError:
            # 已经是项目异常（含上面 raise 出来的）；原样抛出
            raise
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - started
            mapped = self._map_exception(exc)
            self._log_failure(mapped, elapsed)
            raise mapped from exc
        # 流式结束但未产生任何面向用户的内容
        if total_chars == 0:
            raise LLMResponseFormatError(
                "LLM 流式响应在过滤推理内容后为空（仅包含 <think> / <analysis> 等标签）。"
            )

    # ------------------------------------------------------------------
    # 内部：请求构造 / 参数解析
    # ------------------------------------------------------------------
    def _build_request_kwargs(
        self,
        *,
        messages: list[dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        # 第一版：不传 reasoning_split 或其它 MiniMax 私有扩展字段。
        return kwargs

    def _resolve_temperature(self, override: Optional[float]) -> Optional[float]:
        if override is None:
            default = self._settings.minimax_temperature
            if default is None:
                return None
            return float(default)
        return float(override)

    def _ensure_key_for_injected_client(self) -> None:
        """如果 client 是外部注入的（测试场景），仍要确保 Settings 有 Key。

        这样 ``complete`` / ``stream`` 在没有 Key 时也能给出统一错误，
        而不是发出"无 Key"请求后才失败。
        """
        if not self._settings.has_minimax_key():
            raise LLMConfigurationError(
                "缺少环境变量 MINIMAX_API_KEY；"
                "请先在 .env 中设置，或通过环境变量注入。"
            )

    # ------------------------------------------------------------------
    # 内部：非流式响应解析
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_non_stream_content(response: Any) -> str:
        """从非流式响应中提取 ``choices[0].message.content``；失败时抛 LLMResponseFormatError。"""
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMResponseFormatError("LLM 响应缺少 choices。")
        try:
            first = choices[0]
        except IndexError as exc:
            raise LLMResponseFormatError("LLM 响应 choices 为空。") from exc

        message = getattr(first, "message", None)
        if message is None:
            raise LLMResponseFormatError("LLM 响应缺少 message。")

        content = getattr(message, "content", None)
        if content is None:
            raise LLMResponseFormatError("LLM 响应 message.content 为 None。")
        if not isinstance(content, str):
            raise LLMResponseFormatError(
                f"LLM 响应 message.content 不是 str，得到 {type(content).__name__}。"
            )
        if content == "":
            raise LLMResponseFormatError("LLM 响应 message.content 为空字符串。")
        return content

    # ------------------------------------------------------------------
    # 内部：异常映射
    # ------------------------------------------------------------------
    @staticmethod
    def _map_exception(exc: BaseException) -> LLMError:
        """把 OpenAI SDK 异常映射为项目异常；其它异常包装为通用 LLMError。"""
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, AuthenticationError):
            return LLMAuthenticationError("LLM 鉴权失败：请检查 MINIMAX_API_KEY。")
        if isinstance(exc, RateLimitError):
            return LLMRateLimitError("LLM 触发限流（HTTP 429）。")
        if isinstance(exc, APITimeoutError):
            return LLMTimeoutError("LLM 请求超时。")
        if isinstance(exc, APIConnectionError):
            return LLMConnectionError("LLM 连接错误。")
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if isinstance(status, int) and 500 <= status < 600:
                return LLMServerError(f"LLM 服务端错误（HTTP {status}）。")
            return LLMError(f"LLM 请求被拒绝（HTTP {status}）。")
        # 其它未分类异常：保持原类型，但包装为 LLMError 子类以便上层统一处理。
        # 这里返回通用 LLMError，并通过 from exc 保留原始链。
        return LLMError(f"LLM 调用未预期异常：{type(exc).__name__}")

    # ------------------------------------------------------------------
    # 内部：Key 校验 / 日志
    # ------------------------------------------------------------------
    def _require_api_key(self) -> str:
        try:
            return self._settings.require_minimax_key()
        except MissingAPIKeyError as exc:
            raise LLMConfigurationError(str(exc)) from exc

    def _log_failure(self, exc: LLMError, elapsed: float) -> None:
        """记录失败日志；只记录错误类型与耗时，不含 Key / 完整响应。"""
        self._log.warning(
            "LLM 请求失败：model=%s 错误类型=%s 耗时=%.3fs",
            self.model_name,
            type(exc).__name__,
            elapsed,
        )


# 显式 re-export 公共符号
__all__ = [
    "MiniMaxLLMClient",
    "ReasoningContentFilter",
    "sanitize_model_content",
    "LLMError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMServerError",
    "LLMResponseFormatError",
]
