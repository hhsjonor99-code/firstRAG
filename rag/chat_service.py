"""firstRAG 聊天 / RAG 问答服务（业务编排层）。

职责：

1. 接收 ``query`` + 可选 ``history``。
2. 多轮改写：仅当存在历史时调用 LLM 改写为独立问题；无历史直接用原 query。
3. 调用 :class:`Retriever` 检索 Top-K 文档片段。
4. 拼接 RAG Prompt，调用 LLM（OpenAI 兼容客户端，已内部 sanitize reasoning）。
5. 校验 / 过滤非法引用编号；最终答案与 citations 一起返回。
6. 支持非流式 :meth:`ask` 与流式 :meth:`stream`。
7. 流式产出 :class:`ChatStreamEvent` 序列。

公开 API：

- :meth:`ask(query, history)` -> :class:`ChatMessage`
- :meth:`stream(query, history)` -> ``Iterator[ChatStreamEvent]``
- :meth:`rewrite_query(query, history)` -> ``str``

异常层次：

- :class:`ChatServiceError` —— 基类
- :class:`QueryValidationError` —— query 为空或非法
- :class:`QueryRewriteError` —— 改写阶段失败
- :class:`ChatGenerationError` —— 回答生成阶段失败

安全约束：

- 不在日志中输出 query、Prompt、文档正文、完整回答、API Key。
- LLM 客户端已负责过滤 MiniMax ``<think>`` / ``<analysis>`` 等；
  ChatService 不再重新引入这些内容。
- citations 完全由 Retriever 提供；不接受 LLM 自报的文件名 / 页码 / 段落号。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterator, Optional

from config.logging import get_logger
from config.settings import Settings
from config.settings import get_settings as _get_settings

from .llm_client import (
    LLMError,
    LLMResponseFormatError,
    MiniMaxLLMClient,
)
from .models import (
    CHAT_EVENT_DONE,
    CHAT_EVENT_ERROR,
    CHAT_EVENT_REWRITE,
    CHAT_EVENT_SOURCES,
    CHAT_EVENT_TOKEN,
    ChatMessage,
    ChatStreamEvent,
    RetrievedChunk,
)
from .prompt_builder import PromptBuilder
from .retriever import Retriever, RetrieverError

# 固定拒答语（与 PromptBuilder 中的固定话术保持一致）
NO_EVIDENCE_REPLY = "当前知识库中没有找到足够依据。"

# 改写结果的合理性保护
_REWRITE_REJECT_TOO_LONG = 600
_REWRITE_ANSWER_HINT_RE = re.compile(
    r"^(根据|综上|综上所述|根据以上|综合以上|基于以上|根据上文|根据以上信息|根据上文信息|"
    r"答案是|回答是|答复是|可以回答|可以确定|可以肯定)",
)


# ---------------------------------------------------------------------------
# 异常层次
# ---------------------------------------------------------------------------
class ChatServiceError(RuntimeError):
    """聊天服务错误基类。"""


class QueryValidationError(ChatServiceError):
    """query 为空 / 非法。"""


class QueryRewriteError(ChatServiceError):
    """改写阶段失败（LLM 改写错误等）。"""


class ChatGenerationError(ChatServiceError):
    """回答生成阶段失败（LLM 错误、空响应、引用全部非法等）。"""


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------
class ChatService:
    """RAG 聊天服务。

    :param retriever: 检索服务。
    :param prompt_builder: Prompt 构造器。
    :param llm_client: LLM 客户端（OpenAI 兼容）。
    :param settings: 全局配置。
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: MiniMaxLLMClient,
        settings: Optional[Settings] = None,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm_client
        self._settings = settings or _get_settings()
        self._log = get_logger("rag.chat_service")

    # ------------------------------------------------------------------
    # 公共 API：改写 / 非流式 / 流式
    # ------------------------------------------------------------------
    def rewrite_query(
        self,
        query: str,
        history: list[ChatMessage],
    ) -> str:
        """把可能依赖上文的问题改写为独立问题。

        行为：

        - 无历史 → 直接返回原 query（不调用 LLM）。
        - 有历史 → 调用 ``PromptBuilder.build_rewrite_messages`` + ``llm_client.complete``。
        - 改写为空 / 过长 / 出现明显回答式开头 → 回退原 query。
        - LLM 错误 → 抛 :class:`QueryRewriteError`。
        """
        clean_query = self._validate_query(query)
        if not history:
            return clean_query

        messages = self._prompt_builder.build_rewrite_messages(
            query=clean_query, history=history
        )
        try:
            rewritten = self._llm.complete(messages)
        except LLMError as exc:
            raise QueryRewriteError(
                f"问题改写失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise QueryRewriteError(
                f"问题改写失败（未预期异常）：{type(exc).__name__}。"
            ) from exc

        if not isinstance(rewritten, str):
            raise QueryRewriteError("改写结果不是字符串类型。")
        cleaned = rewritten.strip()
        if not cleaned:
            return clean_query
        if len(cleaned) > _REWRITE_REJECT_TOO_LONG:
            self._log.warning(
                "改写结果过长（%d 字符），回退原 query。", len(cleaned)
            )
            return clean_query
        if _REWRITE_ANSWER_HINT_RE.match(cleaned):
            self._log.warning("改写结果疑似回答式开头，回退原 query。")
            return clean_query
        return cleaned

    def ask(
        self,
        query: str,
        history: Optional[list[ChatMessage]] = None,
    ) -> ChatMessage:
        """非流式问答：返回完整 :class:`ChatMessage`。

        流程：

        1. 校验 query。
        2. 改写（无历史时直接用原 query）。
        3. 检索。
        4. 无结果 → 固定拒答。
        5. 有结果 → 拼 Prompt → 调 LLM → 清理非法引用。
        6. 构造 ChatMessage（含 citations、metadata）。
        """
        clean_query = self._validate_query(query)
        history_list = list(history) if history else []
        standalone_query = self.rewrite_query(clean_query, history_list)
        retrieved = self._retrieve(standalone_query)
        if not retrieved:
            return self._make_no_evidence_message(clean_query, standalone_query)
        return self._generate_answer(
            user_query=clean_query,
            standalone_query=standalone_query,
            retrieved=retrieved,
        )

    def stream(
        self,
        query: str,
        history: Optional[list[ChatMessage]] = None,
    ) -> Iterator[ChatStreamEvent]:
        """流式问答：产出 :class:`ChatStreamEvent` 序列。

        事件顺序（典型）：

        1. ``rewrite`` —— 仅当改写发生且与原 query 不同时
        2. ``sources`` —— 检索完成
        3. ``token`` —— 每段 LLM 输出（已 sanitize reasoning）
        4. ``done`` —— 流式结束，携带最终 ChatMessage

        无检索结果时：

        - 仍会产出 ``sources``（citations 为空）；
        - 紧接着产出 ``token`` 事件（固定拒答语片段）；
        - 最后 ``done`` 携带固定 ChatMessage。

        中途异常 → 抛 :class:`ChatGenerationError`（统一策略：不产出 error
        事件后终止，而是直接抛异常；让上层（UI / 测试）能 try/except 捕获）。
        """
        clean_query = self._validate_query(query)
        history_list = list(history) if history else []
        # 1. 改写
        try:
            standalone_query = self.rewrite_query(clean_query, history_list)
        except QueryRewriteError as exc:
            raise ChatGenerationError(str(exc)) from exc
        if standalone_query != clean_query:
            yield ChatStreamEvent(
                event_type=CHAT_EVENT_REWRITE,
                content=standalone_query,
                metadata={"original_query_len": len(clean_query)},
            )
        # 2. 检索
        try:
            retrieved = self._retrieve(standalone_query)
        except (RetrieverError, Exception) as exc:  # noqa: BLE001
            raise ChatGenerationError(
                f"检索失败：{type(exc).__name__}。"
            ) from exc
        # 3. sources 事件
        yield ChatStreamEvent(
            event_type=CHAT_EVENT_SOURCES,
            citations=list(retrieved),
            metadata={
                "retrieval_count": len(retrieved),
                "standalone_query": standalone_query,
            },
        )

        # 4. 无结果：固定拒答
        if not retrieved:
            yield ChatStreamEvent(
                event_type=CHAT_EVENT_TOKEN,
                content=NO_EVIDENCE_REPLY,
            )
            final_msg = self._make_no_evidence_message(clean_query, standalone_query)
            yield ChatStreamEvent(
                event_type=CHAT_EVENT_DONE,
                message=final_msg,
                metadata={"retrieval_count": 0},
            )
            return

        # 5. 有结果：拼 Prompt → 流式调用 LLM
        messages = self._prompt_builder.build_answer_messages(
            query=standalone_query, retrieved_chunks=retrieved
        )
        accumulated: list[str] = []
        try:
            for token in self._llm.stream(messages):
                if not token:
                    continue
                accumulated.append(token)
                yield ChatStreamEvent(
                    event_type=CHAT_EVENT_TOKEN,
                    content=token,
                )
        except LLMError as exc:
            raise ChatGenerationError(
                f"LLM 流式调用失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ChatGenerationError(
                f"LLM 流式调用失败（未预期异常）：{type(exc).__name__}。"
            ) from exc

        # 6. 流结束：清理非法引用，构造最终 ChatMessage
        raw_answer = "".join(accumulated)
        cleaned, illegal = self._prompt_builder.sanitize_invalid_citations(
            raw_answer, retrieved
        )
        if not cleaned:
            raise ChatGenerationError(
                "LLM 流式响应在清理引用后为空。"
            )
        if illegal:
            self._log.warning(
                "LLM 引用了未提供的编号：%s，已过滤。", illegal
            )
        final_msg = self._make_assistant_message(
            user_query=clean_query,
            standalone_query=standalone_query,
            retrieved=retrieved,
            content=cleaned,
            illegal_citations=illegal,
        )
        yield ChatStreamEvent(
            event_type=CHAT_EVENT_DONE,
            message=final_msg,
            metadata={
                "retrieval_count": len(retrieved),
                "illegal_citations": illegal,
                "raw_answer_len": len(raw_answer),
                "cleaned_answer_len": len(cleaned),
            },
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _validate_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise QueryValidationError("query 必须是 str 类型。")
        cleaned = query.strip()
        if not cleaned:
            raise QueryValidationError("query 不能为空。")
        return cleaned

    def _retrieve(self, standalone_query: str) -> list[RetrievedChunk]:
        try:
            return self._retriever.retrieve(standalone_query)
        except RetrieverError as exc:
            raise ChatGenerationError(
                f"检索失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ChatGenerationError(
                f"检索失败（未预期异常）：{type(exc).__name__}。"
            ) from exc

    def _generate_answer(
        self,
        user_query: str,
        standalone_query: str,
        retrieved: list[RetrievedChunk],
    ) -> ChatMessage:
        messages = self._prompt_builder.build_answer_messages(
            query=standalone_query, retrieved_chunks=retrieved
        )
        try:
            raw_answer = self._llm.complete(messages)
        except LLMError as exc:
            raise ChatGenerationError(
                f"LLM 回答失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ChatGenerationError(
                f"LLM 回答失败（未预期异常）：{type(exc).__name__}。"
            ) from exc
        if not isinstance(raw_answer, str) or not raw_answer:
            raise ChatGenerationError("LLM 响应为空。")
        cleaned, illegal = self._prompt_builder.sanitize_invalid_citations(
            raw_answer, retrieved
        )
        if not cleaned:
            raise ChatGenerationError("LLM 响应在清理引用后为空。")
        if illegal:
            self._log.warning(
                "LLM 引用了未提供的编号：%s，已过滤。", illegal
            )
        return self._make_assistant_message(
            user_query=user_query,
            standalone_query=standalone_query,
            retrieved=retrieved,
            content=cleaned,
            illegal_citations=illegal,
        )

    def _make_assistant_message(
        self,
        user_query: str,
        standalone_query: str,
        retrieved: list[RetrievedChunk],
        content: str,
        illegal_citations: Optional[list[str]] = None,
    ) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=content,
            citations=list(retrieved),
            created_at=datetime.now(timezone.utc),
            metadata=self._build_metadata(
                user_query=user_query,
                standalone_query=standalone_query,
                retrieved=retrieved,
                illegal_citations=illegal_citations,
            ),
        )

    def _make_no_evidence_message(
        self,
        user_query: str,
        standalone_query: str,
    ) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=NO_EVIDENCE_REPLY,
            citations=[],
            created_at=datetime.now(timezone.utc),
            metadata=self._build_metadata(
                user_query=user_query,
                standalone_query=standalone_query,
                retrieved=[],
                illegal_citations=None,
            ),
        )

    @staticmethod
    def _build_metadata(
        user_query: str,
        standalone_query: str,
        retrieved: list[RetrievedChunk],
        illegal_citations: Optional[list[str]],
    ) -> dict:
        meta: dict = {
            "standalone_query": standalone_query,
            "retrieval_count": len(retrieved),
        }
        if illegal_citations:
            meta["illegal_citations"] = list(illegal_citations)
        return meta


# 显式 re-export
__all__ = [
    "ChatService",
    "ChatServiceError",
    "QueryValidationError",
    "QueryRewriteError",
    "ChatGenerationError",
    "NO_EVIDENCE_REPLY",
]
