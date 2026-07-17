"""测试 / 调试脚本共享的假实现（Fake）。

**仅用于单元测试与本地调试**；**不**用于生产路径。

提供：

- :class:`FakeEmbeddingProvider` —— 满足 :class:`rag.embedding_provider.EmbeddingProvider` 协议
- :class:`FakeLLMClient` —— 满足 :class:`rag.llm_client.MiniMaxLLMClient` 公开 API
- :class:`StreamingFakeLLMClient` —— 支持 ``stream()`` 的 FakeLLMClient
- :class:`FakeRetriever` —— 满足 :class:`rag.retriever.Retriever` 公开 API

这些对象：

- 不调用任何远程 API；
- 不读取 / 写入磁盘；
- 不记录或打印任何 Key、Prompt、文档正文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np

from config.settings import Settings
from rag.embedding_provider import EmbeddingError
from rag.llm_client import LLMError
from rag.models import ChatMessage, ChatStreamEvent, DocumentChunk, RetrievedChunk


# ---------------------------------------------------------------------------
# FakeEmbeddingProvider
# ---------------------------------------------------------------------------
class FakeEmbeddingProvider:
    """确定性假 Embedding。

    同一文本每次返回**相同**的伪向量（基于 SHA1 哈希生成的 0-1 float）。
    向量经过 L2 归一化，使得余弦相似度等价于点积。

    :param dimensions: 输出维度；缺省从 Settings 读取。
    :param settings: 读取 ``siliconflow_embedding_dimensions`` 的配置。
    :param fail: 若提供，则 ``embed_*`` 调用时抛 :class:`EmbeddingError`。
    """

    def __init__(
        self,
        dimensions: Optional[int] = None,
        settings: Optional[Settings] = None,
        fail: Optional[Exception] = None,
    ) -> None:
        s = settings or Settings(_env_file=None)  # type: ignore[call-arg]
        self._dimensions = int(dimensions) if dimensions is not None else int(
            s.siliconflow_embedding_dimensions
        )
        self._fail = fail
        self.call_count = 0
        self.last_texts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("FakeEmbeddingProvider 输入文本列表为空。")
        for t in texts:
            if not isinstance(t, str):
                raise EmbeddingError("FakeEmbeddingProvider 输入包含非 str 元素。")
        if self._fail is not None:
            self.call_count += 1
            self.last_texts = list(texts)
            raise self._fail
        self.call_count += 1
        self.last_texts = list(texts)
        arr = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i] = self._fake_vector(t)
        # L2 归一化
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe = np.where(norms == 0, 1.0, norms)
        arr = arr / safe
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])

    def _fake_vector(self, text: str) -> np.ndarray:
        """根据文本生成确定性向量。"""
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        # 重复填充到 dimensions 个字节
        buf = (h * ((self._dimensions // len(h)) + 1))[: self._dimensions]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        return arr


# ---------------------------------------------------------------------------
# FakeLLMClient
# ---------------------------------------------------------------------------
@dataclass
class FakeLLMClient:
    """非流式假 LLM 客户端。

    通过 :attr:`responses` 字典映射 ``messages`` 哈希 → 返回文本；或在
    :attr:`default_response` 时返回固定文本。

    可注入 :attr:`raise_on_complete` 让 ``complete()`` 抛指定异常。
    """

    default_response: str = "FakeLLM 的回答。"
    responses: dict[str, str] = field(default_factory=dict)
    raise_on_complete: Optional[BaseException] = None
    call_count: int = 0
    last_messages: list[dict[str, str]] = field(default_factory=list)
    last_temperature: Optional[float] = None
    last_max_tokens: Optional[int] = None

    @property
    def model_name(self) -> str:
        return "fake-llm"

    @property
    def model(self) -> str:
        return self.model_name

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.call_count += 1
        self.last_messages = [dict(m) for m in messages]
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        if self.raise_on_complete is not None:
            raise self.raise_on_complete
        key = self._make_key(messages)
        if key in self.responses:
            return self.responses[key]
        return self.default_response

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """默认流式：把 ``complete()`` 结果按 1 字符一段 yield。"""
        text = self.complete(messages, temperature, max_tokens)
        for ch in text:
            yield ch

    @staticmethod
    def _make_key(messages: list[dict[str, str]]) -> str:
        """用 messages 拼接产生稳定 key。"""
        return "|".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in messages)


@dataclass
class StreamingFakeLLMClient(FakeLLMClient):
    """支持更复杂流式行为（按 token 列表）的 FakeLLMClient。"""

    stream_chunks: list[str] = field(default_factory=list)
    """流式返回的 token 列表；与 :attr:`default_response` 互斥。"""

    def stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        self.call_count += 1
        self.last_messages = [dict(m) for m in messages]
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        if self.raise_on_complete is not None:
            raise self.raise_on_complete
        if self.stream_chunks:
            for c in self.stream_chunks:
                yield c
            return
        for ch in self.complete(messages, temperature, max_tokens):
            yield ch


# ---------------------------------------------------------------------------
# FakeRetriever
# ---------------------------------------------------------------------------
class FakeRetriever:
    """假 Retriever，返回固定的 RetrievedChunk 列表。"""

    def __init__(self, results: Optional[list[RetrievedChunk]] = None) -> None:
        self._results = list(results) if results else []
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_top_k: Optional[int] = None
        self.raise_on_retrieve: Optional[BaseException] = None

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        self.call_count += 1
        self.last_query = query
        self.last_top_k = top_k
        if self.raise_on_retrieve is not None:
            raise self.raise_on_retrieve
        # 截到 top_k
        if top_k is not None and top_k > 0:
            return self._results[:top_k]
        return list(self._results)

    def set_results(self, results: list[RetrievedChunk]) -> None:
        self._results = list(results)


def make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    source_name: str = "fake.txt",
    score: float = 0.9,
    citation_id: str = "S1",
) -> RetrievedChunk:
    """便捷构造 RetrievedChunk。"""
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        source_name=source_name,
        chunk_index=0,
    )
    return RetrievedChunk(chunk=chunk, score=score, citation_id=citation_id)


# ---------------------------------------------------------------------------
# FakeChatService
# ---------------------------------------------------------------------------
class FakeChatService:
    """满足 :class:`rag.chat_service.ChatService.stream` 协议的假服务。

    用于 UI 控制器的回归测试。**不**调用任何远程 API。

    :param events: 预设的 :class:`ChatStreamEvent` 序列；将按顺序 yield。
    :param raise_on_stream: 若提供，则 ``stream()`` 立即抛出该异常，
        不会 yield 任何事件。
    :param final_message: 若提供，会被记录到 ``last_final_message``，
        便于测试在 controller 之外校验。
    """

    def __init__(
        self,
        events: Optional[list[ChatStreamEvent]] = None,
        raise_on_stream: Optional[BaseException] = None,
        final_message: Optional[ChatMessage] = None,
    ) -> None:
        self._events: list[ChatStreamEvent] = list(events) if events else []
        self._raise = raise_on_stream
        self._final_message = final_message
        self.call_count = 0
        self.last_query: Optional[str] = None
        self.last_history: Optional[list[ChatMessage]] = None

    def stream(
        self,
        query: str,
        history: Optional[list[ChatMessage]] = None,
    ) -> Iterator[ChatStreamEvent]:
        self.call_count += 1
        self.last_query = query
        self.last_history = list(history) if history else []
        if self._raise is not None:
            raise self._raise
        for ev in self._events:
            yield ev

    # 为兼容旧接口保留 ``ask``。
    def ask(
        self,
        query: str,
        history: Optional[list[ChatMessage]] = None,
    ) -> ChatMessage:
        for ev in self.stream(query, history=history):
            if getattr(ev, "message", None) is not None:
                return ev.message  # type: ignore[return-value]
        raise LLMError("FakeChatService.ask: stream 未产出 done 事件。")
