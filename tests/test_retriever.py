"""Retriever 单元测试。

使用 ``FakeEmbeddingProvider``（查询向量 = 预设的第几条 chunk 向量）与
临时 ``FaissVectorStore``；**不调用**远程 API。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from rag.embedding_provider import EmbeddingError, EmbeddingProvider  # noqa: E402
from rag.models import DocumentChunk, DocumentInfo  # noqa: E402
from rag.retriever import Retriever, RetrieverError  # noqa: E402
from rag.vector_store import FaissVectorStore, VectorStoreError  # noqa: E402


DIM = 1024


# ---------------------------------------------------------------------------
# Fake / Mock EmbeddingProvider
# ---------------------------------------------------------------------------
class FakeEmbeddingProvider(EmbeddingProvider):
    """Fake Embedding：把 query 映射到一个固定的 dim 维向量。

    ``mode`` 控制行为：
    - ``"deterministic"``：相同 query → 相同向量；不同 query → 不同向量。
    - ``"always_zero"``：永远返回零向量（构造 0,0,...）。
    - ``"select:n"``：固定返回第 n 个 chunk 的向量（与 store 配合）。
    """

    def __init__(self, dim: int = DIM, mode: str = "deterministic") -> None:
        self._dim = dim
        self._mode = mode
        self.calls: list[str] = []
        self.raise_on_call: Optional[Exception] = None

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.raise_on_call:
            raise self.raise_on_call
        return np.stack([self._vec_for(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if self.raise_on_call:
            raise self.raise_on_call
        v = self._vec_for(text)
        return v.reshape(1, self._dim)

    def _vec_for(self, text: str) -> np.ndarray:
        if self._mode == "always_zero":
            return np.zeros(self._dim, dtype=np.float32)
        if self._mode.startswith("select:"):
            # 用于精确指定「这个 query 等于第 n 条 chunk」
            idx = int(self._mode.split(":", 1)[1])
            return self._seed_vector(idx)
        # deterministic: hash(text) -> seed
        seed = abs(hash(text)) % (2**31 - 1)
        return self._seed_vector(seed)

    def _seed_vector(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.uniform(-1.0, 1.0, size=self._dim).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-12)
        return v


# ---------------------------------------------------------------------------
# 测试 fixtures
# ---------------------------------------------------------------------------
def _new_settings(**overrides) -> Settings:
    with mock.patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _make_vector(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
    v /= max(np.linalg.norm(v), 1e-12)
    return v


def _make_doc(document_id: str = "doc-1", chunk_count: int = 5) -> DocumentInfo:
    return DocumentInfo(
        document_id=document_id,
        file_name=f"{document_id}.docx",
        original_file_name="示例.docx",
        file_type="docx",
        file_hash=f"hash-{document_id}",
        file_size=1024,
        created_at=datetime(2026, 7, 16, 22, 0, 0, tzinfo=timezone.utc),
        chunk_count=chunk_count,
    )


def _make_chunk(chunk_id: str, document_id: str, idx: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"文本内容-{idx}",
        source_name="示例.docx",
        chunk_index=idx,
    )


def _populate_store(
    store: FaissVectorStore,
    doc_id: str,
    n: int = 5,
    base_seed: int = 100,
) -> tuple[DocumentInfo, list[DocumentChunk], np.ndarray]:
    chunks = [_make_chunk(f"chunk-{i}", doc_id, i) for i in range(n)]
    vectors = np.stack([_make_vector(base_seed + i) for i in range(n)])
    doc = _make_doc(doc_id, chunk_count=n)
    store.add_document(doc, chunks, vectors)
    return doc, chunks, vectors


def _build_retriever(
    tmp_path: Path,
    *,
    top_k: int = 3,
    n: int = 5,
    embedding_mode: str = "deterministic",
    min_score: Optional[float] = None,
    settings_overrides: Optional[dict] = None,
    base_seed: int = 10,
) -> tuple[Retriever, FakeEmbeddingProvider, FaissVectorStore]:
    """构造 Retriever；当 embedding_mode 是 ``select:N`` 时，base_seed 自动取 0
    以保证 chunk N 的向量与 query 完全相同。"""
    if embedding_mode.startswith("select:"):
        base_seed = 0
    settings = _new_settings(retrieval_top_k=top_k, retrieval_min_score=min_score, **(settings_overrides or {}))
    store = FaissVectorStore(settings=settings, index_dir=tmp_path / "indexes")
    store.load()
    _populate_store(store, "doc-1", n=n, base_seed=base_seed)
    embed = FakeEmbeddingProvider(dim=DIM, mode=embedding_mode)
    retriever = Retriever(embedding_provider=embed, vector_store=store, settings=settings)
    return retriever, embed, store


# ---------------------------------------------------------------------------
# 1. 单条查询成功
# ---------------------------------------------------------------------------
def test_retrieve_single_query_success(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path, top_k=3, n=5)
    results = r.retrieve("阿莫西林是什么？")
    assert isinstance(results, list)
    assert len(results) == 3
    for rc in results:
        assert rc.score is not None
        assert isinstance(rc.score, float)
        assert rc.citation_id.startswith("S")


# ---------------------------------------------------------------------------
# 2. top_k 默认值
# ---------------------------------------------------------------------------
def test_default_top_k_from_settings(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path, top_k=4, n=10)
    # 不传 top_k → 使用 settings.retrieval_top_k=4
    results = r.retrieve("查询")
    assert len(results) == 4


# ---------------------------------------------------------------------------
# 3. 自定义 top_k
# ---------------------------------------------------------------------------
def test_custom_top_k(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path, top_k=5, n=10)
    results = r.retrieve("查询", top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 4. top_k 大于索引数量
# ---------------------------------------------------------------------------
def test_top_k_exceeds_chunk_count(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path, top_k=3, n=5)
    results = r.retrieve("查询", top_k=100)
    # FaissVectorStore 会自动截断到 chunk_count
    assert len(results) == 5


# ---------------------------------------------------------------------------
# 5. 空知识库返回空
# ---------------------------------------------------------------------------
def test_empty_knowledge_base(tmp_path: Path):
    settings = _new_settings(retrieval_top_k=3)
    store = FaissVectorStore(settings=settings, index_dir=tmp_path / "idx")
    store.load()
    embed = FakeEmbeddingProvider()
    r = Retriever(embed, store, settings)
    results = r.retrieve("查询")
    assert results == []


# ---------------------------------------------------------------------------
# 6. 空查询报错
# ---------------------------------------------------------------------------
def test_empty_query_raises(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path)
    with pytest.raises(RetrieverError):
        r.retrieve("")
    with pytest.raises(RetrieverError):
        r.retrieve("   \n  ")


# ---------------------------------------------------------------------------
# 7. top_k <= 0 报错
# ---------------------------------------------------------------------------
def test_top_k_zero_or_negative(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path)
    with pytest.raises(RetrieverError):
        r.retrieve("查询", top_k=0)
    with pytest.raises(RetrieverError):
        r.retrieve("查询", top_k=-1)


# ---------------------------------------------------------------------------
# 8. 检索顺序正确
# ---------------------------------------------------------------------------
def test_retrieval_order_matches_scores(tmp_path: Path):
    """select:3 让 query 与第 3 条完全一致 → top-1 必为 chunk-3"""
    r, embed, store = _build_retriever(
        tmp_path, top_k=5, n=5, embedding_mode="select:3"
    )
    results = r.retrieve("any")
    assert results[0].chunk.chunk_id == "chunk-3"
    scores = [rc.score for rc in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 9. citation_id 为 S1..Sk
# ---------------------------------------------------------------------------
def test_citation_ids_are_S1_to_Sk(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path, top_k=3, n=5)
    results = r.retrieve("查询")
    assert [rc.citation_id for rc in results] == ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------
# 10. score 为 float
# ---------------------------------------------------------------------------
def test_score_is_python_float(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path)
    results = r.retrieve("查询")
    for rc in results:
        assert isinstance(rc.score, float)


# ---------------------------------------------------------------------------
# 11. min_score 过滤
# ---------------------------------------------------------------------------
def test_min_score_filter(tmp_path: Path):
    r, _, _ = _build_retriever(tmp_path, top_k=5, n=5, embedding_mode="select:3")
    # 先看全量分数
    full = r.retrieve("any")
    assert len(full) == 5
    # 设置一个高阈值
    high_thr = max(rc.score for rc in full) - 1e-4
    filtered = r.retrieve("any", min_score=high_thr)
    assert all(rc.score >= high_thr for rc in filtered)
    assert len(filtered) < 5

    # 设置一个不可能达到的高阈值
    too_high = [rc for rc in full if rc.score >= 2.0]
    filtered2 = r.retrieve("any", min_score=2.0)
    assert filtered2 == []


def test_min_score_from_settings(tmp_path: Path):
    """从 settings.retrieval_min_score 读取阈值。"""
    settings = _new_settings(retrieval_top_k=5, retrieval_min_score=0.999999)
    store = FaissVectorStore(settings=settings, index_dir=tmp_path / "idx")
    store.load()
    # base_seed=0 → chunk 3 的向量种子 = 3，与 select:3 对齐
    _populate_store(store, "doc-1", n=5, base_seed=0)
    embed = FakeEmbeddingProvider(dim=DIM, mode="select:3")
    r = Retriever(embed, store, settings)
    full = r.retrieve("any")
    # 0.999999 仅匹配 query 自己那条（其余 chunks 分数远低）
    assert len(full) == 1
    assert full[0].citation_id == "S1"


# ---------------------------------------------------------------------------
# 12. Embedding 异常正确传播
# ---------------------------------------------------------------------------
def test_embedding_exception_propagates(tmp_path: Path):
    r, embed, _ = _build_retriever(tmp_path)
    embed.raise_on_call = EmbeddingError("测试嵌入异常")
    with pytest.raises(EmbeddingError) as ei:
        r.retrieve("查询")
    assert "测试嵌入异常" in str(ei.value)


# ---------------------------------------------------------------------------
# 13. VectorStore 异常正确传播
# ---------------------------------------------------------------------------
def test_vector_store_exception_propagates(tmp_path: Path, monkeypatch):
    r, _, store = _build_retriever(tmp_path)
    def _boom(*a, **k):  # noqa: ANN001
        raise VectorStoreError("索引损坏（模拟）")
    monkeypatch.setattr(store, "search", _boom)
    with pytest.raises(VectorStoreError) as ei:
        r.retrieve("查询")
    assert "索引损坏" in str(ei.value)


# ---------------------------------------------------------------------------
# 14. 不修改索引内容
# ---------------------------------------------------------------------------
def test_retrieve_does_not_modify_store(tmp_path: Path):
    r, _, store = _build_retriever(tmp_path, top_k=3, n=5)
    n_before = store.chunk_count
    docs_before = store.document_count
    snap_before = store._read_current_snapshot_id()  # type: ignore[attr-defined]
    r.retrieve("查询")
    r.retrieve("另一查询", top_k=2)
    r.retrieve("再一查询", top_k=1, min_score=0.5)
    assert store.chunk_count == n_before
    assert store.document_count == docs_before
    assert store._read_current_snapshot_id() == snap_before  # type: ignore[attr-defined]