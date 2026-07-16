"""firstRAG 检索服务。

职责：

1. 将用户 query 通过 :class:`EmbeddingProvider` 编码为向量；
2. 通过 :class:`FaissVectorStore` 检索 Top-K 相似 chunks；
3. 把结果包装为 :class:`RetrievedChunk` 列表，并按程序生成 ``S1..Sk`` 编号；
4. 不修改 :class:`FaissVectorStore` 中的任何数据；
5. 不调用任何 LLM；本模块**仅**做检索。

安全约束：

- query 不在日志中明文记录，只记录长度、top_k、返回数量、耗时。
- Embedding / VectorStore 异常保持原异常链，不吞掉。
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from config.logging import get_logger
from config.settings import Settings

from .embedding_provider import EmbeddingError, EmbeddingProvider
from .models import RetrievedChunk
from .vector_store import FaissVectorStore, VectorStoreError


class RetrieverError(RuntimeError):
    """Retriever 上层错误的基类。"""


class Retriever:
    """封装「query → embedding → 向量检索 → RetrievedChunk」流程。

    :param embedding_provider: 任何满足 :class:`EmbeddingProvider` 接口的对象
        （生产环境是 :class:`SiliconFlowEmbeddingProvider`，测试可以是 fake）。
    :param vector_store: 已加载的 :class:`FaissVectorStore`。
    :param settings: 全局配置（用于读取 ``retrieval_top_k`` /
        ``retrieval_min_score`` / ``max_history_turns`` 等）。
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: FaissVectorStore,
        settings: Settings,
    ) -> None:
        self._embedding = embedding_provider
        self._store = vector_store
        self._settings = settings
        self._log = get_logger("rag.retriever")

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """根据 query 检索 Top-K 相似 chunks。

        :param query: 用户问题文本；首尾空白被去除；为空抛 :class:`RetrieverError`。
        :param top_k: 返回结果数量上限；缺省为
            ``settings.retrieval_top_k``。必须 > 0。
        :param min_score: 可选最低分过滤；缺省读取
            ``settings.retrieval_min_score``；``None`` 表示不过滤。
        :returns: 按分数从高到低排列的 :class:`RetrievedChunk` 列表；
            ``citation_id`` 已按 ``S1, S2, ...`` 程序生成。
        """
        if not isinstance(query, str):
            raise RetrieverError("query 必须是 str 类型。")
        clean_query = query.strip()
        if not clean_query:
            raise RetrieverError("query 不能为空。")

        if top_k is None:
            top_k = int(self._settings.retrieval_top_k)
        if not isinstance(top_k, int) or top_k <= 0:
            raise RetrieverError(f"top_k 必须为正整数，得到 {top_k!r}。")

        if min_score is None:
            min_score = self._settings.retrieval_min_score

        # 空知识库提前返回
        if not self._store.is_loaded:
            raise RetrieverError("向量库未加载。")
        if self._store.chunk_count == 0:
            self._log.info(
                "检索：query_len=%d top_k=%d → 知识库为空，返回空列表。",
                len(clean_query),
                top_k,
            )
            return []

        started = time.perf_counter()
        try:
            query_vector = self._embedding.embed_query(clean_query)
        except EmbeddingError:
            # 保持原异常链，让上层清晰区分错误类型
            raise
        except Exception as exc:
            # 未知异常包装为 EmbeddingError 上层仍可识别
            raise EmbeddingError(f"Embedding 失败：{type(exc).__name__}: {exc}") from exc

        if not isinstance(query_vector, np.ndarray):
            raise RetrieverError(
                f"Embedding 返回值类型不是 np.ndarray，得到 {type(query_vector).__name__}。"
            )

        try:
            raw_results = self._store.search(query_vector, top_k=top_k)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"向量检索失败：{type(exc).__name__}: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        # min_score 过滤
        if min_score is not None:
            raw_results = [(c, s) for c, s in raw_results if s >= float(min_score)]

        # 生成 S1..Sk
        retrieved: list[RetrievedChunk] = []
        for i, (chunk, score) in enumerate(raw_results, 1):
            retrieved.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    citation_id=f"S{i}",
                )
            )

        self._log.info(
            "检索完成：query_len=%d top_k=%d 返回=%d 耗时=%.2fms",
            len(clean_query),
            top_k,
            len(retrieved),
            elapsed_ms,
        )
        return retrieved