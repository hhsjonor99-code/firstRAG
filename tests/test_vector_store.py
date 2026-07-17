"""FaissVectorStore 单元测试。

所有测试使用临时目录与人工构造的小向量，**不调用** SiliconFlow 或任何
远程 API。所有 Key 在测试中保持 None（缺 Key 行为不在本模块范围内）。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import faiss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402

from rag.models import DocumentChunk, DocumentInfo  # noqa: E402
from rag.vector_store import (  # noqa: E402
    DuplicateDocumentError,
    FaissVectorStore,
    IndexCorruptedError,
    IndexIncompatibleError,
    IndexNotFoundError,
    VectorStoreError,
    VectorValidationError,
)


DIM = 1024  # 与项目默认 siliconflow_embedding_dimensions 一致


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _new_settings(**overrides) -> Settings:
    with mock.patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _make_vector(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
    v /= max(np.linalg.norm(v), 1e-12)
    return v


def _make_vectors(n: int, base_seed: int = 0, dim: int = DIM) -> np.ndarray:
    return np.stack([_make_vector(base_seed + i, dim) for i in range(n)])


def _make_document(
    document_id: str = "doc-1",
    file_hash: str = "hash-1",
    file_name: str = "1.docx",
    original_file_name: str = "示例.docx",
    file_type: str = "docx",
    file_size: int = 1024,
    chunk_count: int = 3,
) -> DocumentInfo:
    return DocumentInfo(
        document_id=document_id,
        file_name=file_name,
        original_file_name=original_file_name,
        file_type=file_type,
        file_hash=file_hash,
        file_size=file_size,
        created_at=datetime(2026, 7, 16, 22, 0, 0, tzinfo=timezone.utc),
        chunk_count=chunk_count,
    )


def _make_chunk(
    chunk_id: str,
    document_id: str,
    content: str = "示例内容",
    chunk_index: int = 0,
    **kwargs,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        source_name="示例.docx",
        chunk_index=chunk_index,
        **kwargs,
    )


def _make_store(tmp_path: Path, **settings_overrides) -> FaissVectorStore:
    settings = _new_settings(**settings_overrides)
    return FaissVectorStore(settings=settings, index_dir=tmp_path / "indexes")


# ---------------------------------------------------------------------------
# 1. 初始化空索引
# ---------------------------------------------------------------------------
def test_init_empty(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    assert s.is_loaded
    assert s.document_count == 0
    assert s.chunk_count == 0
    assert s.dimensions == DIM
    # CURRENT 不应存在
    assert not s.current_file.exists()


def test_load_first_time_creates_dirs(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    assert s.snapshots_dir.exists()
    assert s.index_dir.exists()


# ---------------------------------------------------------------------------
# 2. 添加单个文档
# ---------------------------------------------------------------------------
def test_add_one_document(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=3)
    chunks = [_make_chunk(f"chunk-{i}", doc.document_id, chunk_index=i) for i in range(3)]
    vectors = _make_vectors(3, base_seed=10)
    s.add_document(doc, chunks, vectors)

    assert s.document_count == 1
    assert s.chunk_count == 3
    assert s.has_document_hash(doc.file_hash)
    assert s.get_document_by_hash(doc.file_hash).document_id == doc.document_id
    # CURRENT 应指向新快照
    assert s.current_file.exists()


# ---------------------------------------------------------------------------
# 3. 添加多个文档
# ---------------------------------------------------------------------------
def test_add_multiple_documents(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    for i in range(3):
        doc = _make_document(
            document_id=f"doc-{i}",
            file_hash=f"hash-{i}",
            chunk_count=2,
        )
        chunks = [_make_chunk(f"c-{i}-{j}", doc.document_id, chunk_index=j) for j in range(2)]
        vectors = _make_vectors(2, base_seed=i * 100)
        s.add_document(doc, chunks, vectors)

    assert s.document_count == 3
    assert s.chunk_count == 6


# ---------------------------------------------------------------------------
# 4. 搜索结果排序正确
# ---------------------------------------------------------------------------
def test_search_results_sorted_descending(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=5)
    chunks = [_make_chunk(f"c{i}", doc.document_id, content=f"文本{i}", chunk_index=i) for i in range(5)]
    vectors = _make_vectors(5, base_seed=1)
    s.add_document(doc, chunks, vectors)

    # query 与第 3 条最接近（用 vectors[2] 作为 query）
    query = vectors[2].copy()
    results = s.search(query, top_k=5)
    assert len(results) == 5
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)
    # top-1 应该是第 3 条（与 query 完全相同）
    assert results[0][0].chunk_id == "c2"


# ---------------------------------------------------------------------------
# 5. top_k 大于索引数量
# ---------------------------------------------------------------------------
def test_top_k_exceeds_chunk_count(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=3)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(3)]
    vectors = _make_vectors(3, base_seed=2)
    s.add_document(doc, chunks, vectors)

    results = s.search(_make_vector(100), top_k=100)
    assert len(results) == 3  # 自动截断


# ---------------------------------------------------------------------------
# 6. 空索引搜索
# ---------------------------------------------------------------------------
def test_search_empty_index(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    results = s.search(_make_vector(0), top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# 7. 一维 query 自动转二维
# ---------------------------------------------------------------------------
def test_query_1d_accept(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    vectors = _make_vectors(2)
    s.add_document(doc, chunks, vectors)

    # 1D query
    q1 = _make_vector(50)
    results = s.search(q1, top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 8. dtype 转 float32
# ---------------------------------------------------------------------------
def test_input_float64_normalized_to_float32(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    # 传入 float64
    vectors = _make_vectors(2).astype(np.float64)
    s.add_document(doc, chunks, vectors)

    arr = s._vectors  # type: ignore[attr-defined]
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# 9. 维度错误
# ---------------------------------------------------------------------------
def test_wrong_dim_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    wrong = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(VectorValidationError):
        s.add_document(doc, chunks, wrong)


# ---------------------------------------------------------------------------
# 10. vectors 数量与 chunks 不一致
# ---------------------------------------------------------------------------
def test_count_mismatch_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    vectors = _make_vectors(3)  # 3 vectors vs 2 chunks
    with pytest.raises(VectorValidationError):
        s.add_document(doc, chunks, vectors)


# ---------------------------------------------------------------------------
# 11. NaN
# ---------------------------------------------------------------------------
def test_nan_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    vec = _make_vectors(2)
    vec[0, 5] = float("nan")
    with pytest.raises(VectorValidationError):
        s.add_document(doc, chunks, vec)


# ---------------------------------------------------------------------------
# 12. Inf
# ---------------------------------------------------------------------------
def test_inf_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    vec = _make_vectors(2)
    vec[1, 7] = float("inf")
    with pytest.raises(VectorValidationError):
        s.add_document(doc, chunks, vec)


# ---------------------------------------------------------------------------
# 13. 全零向量
# ---------------------------------------------------------------------------
def test_zero_vector_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    vec = _make_vectors(2)
    vec[0] = 0.0
    with pytest.raises(VectorValidationError):
        s.add_document(doc, chunks, vec)


# ---------------------------------------------------------------------------
# 14. 空向量
# ---------------------------------------------------------------------------
def test_empty_vectors_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=0)
    chunks: list[DocumentChunk] = []
    empty = np.zeros((0, DIM), dtype=np.float32)
    with pytest.raises(VectorValidationError):
        s.add_document(doc, chunks, empty)


# ---------------------------------------------------------------------------
# 15. 重复 file_hash
# ---------------------------------------------------------------------------
def test_duplicate_file_hash_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc1 = _make_document(document_id="d1", file_hash="hash-A", chunk_count=2)
    chunks1 = [_make_chunk("a-0", "d1", chunk_index=0), _make_chunk("a-1", "d1", chunk_index=1)]
    vectors1 = _make_vectors(2)
    s.add_document(doc1, chunks1, vectors1)

    doc2 = _make_document(document_id="d2", file_hash="hash-A", chunk_count=2)
    chunks2 = [_make_chunk("b-0", "d2", chunk_index=0), _make_chunk("b-1", "d2", chunk_index=1)]
    vectors2 = _make_vectors(2, base_seed=99)
    with pytest.raises(DuplicateDocumentError):
        s.add_document(doc2, chunks2, vectors2)


# ---------------------------------------------------------------------------
# 16. 重复 document_id
# ---------------------------------------------------------------------------
def test_duplicate_document_id_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc1 = _make_document(document_id="d1", file_hash="h1", chunk_count=2)
    chunks1 = [_make_chunk("a-0", "d1", chunk_index=0), _make_chunk("a-1", "d1", chunk_index=1)]
    s.add_document(doc1, chunks1, _make_vectors(2))

    doc2 = _make_document(document_id="d1", file_hash="h2", chunk_count=2)
    chunks2 = [_make_chunk("b-0", "d1", chunk_index=0), _make_chunk("b-1", "d1", chunk_index=1)]
    with pytest.raises(VectorStoreError):
        s.add_document(doc2, chunks2, _make_vectors(2, base_seed=200))


# ---------------------------------------------------------------------------
# 17. 重复 chunk_id
# ---------------------------------------------------------------------------
def test_duplicate_chunk_id_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(chunk_count=2)
    chunks = [
        _make_chunk("dup", doc.document_id, chunk_index=0),
        _make_chunk("dup", doc.document_id, chunk_index=1),
    ]
    with pytest.raises(VectorStoreError):
        s.add_document(doc, chunks, _make_vectors(2))


# ---------------------------------------------------------------------------
# 18. document_id 与 chunk 不一致
# ---------------------------------------------------------------------------
def test_chunk_document_id_mismatch_raises(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc = _make_document(document_id="docA", chunk_count=2)
    chunks = [
        _make_chunk("c0", "docB", chunk_index=0),  # 错配
        _make_chunk("c1", "docA", chunk_index=1),
    ]
    with pytest.raises(VectorStoreError):
        s.add_document(doc, chunks, _make_vectors(2))


# ---------------------------------------------------------------------------
# 19. 保存后重新加载
# ---------------------------------------------------------------------------
def test_save_and_reload(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=3)
    chunks = [_make_chunk(f"c{i}", doc.document_id, content=f"内容{i}", chunk_index=i) for i in range(3)]
    s.add_document(doc, chunks, _make_vectors(3))

    # 重新构造并 load
    s2 = _make_store(tmp_path)
    s2.load()

    assert s2.document_count == 1
    assert s2.chunk_count == 3
    assert s2._vectors.shape == (3, DIM)  # type: ignore[attr-defined]
    assert s2.get_document_by_hash(doc.file_hash).document_id == doc.document_id


# ---------------------------------------------------------------------------
# 20. 加载后搜索结果一致
# ---------------------------------------------------------------------------
def test_reload_search_consistent(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=5)
    chunks = [_make_chunk(f"c{i}", doc.document_id, content=f"text{i}", chunk_index=i) for i in range(5)]
    vectors = _make_vectors(5, base_seed=7)
    s.add_document(doc, chunks, vectors)

    query = vectors[3].copy()
    pre = s.search(query, top_k=3)
    pre_ids = [c.chunk_id for c, _ in pre]
    pre_scores = [round(s_, 6) for _, s_ in pre]

    # reload
    s2 = _make_store(tmp_path)
    s2.load()
    post = s2.search(query, top_k=3)
    post_ids = [c.chunk_id for c, _ in post]
    post_scores = [round(s_, 6) for _, s_ in post]

    assert pre_ids == post_ids
    assert pre_scores == post_scores


# ---------------------------------------------------------------------------
# 21. 删除单个文档
# ---------------------------------------------------------------------------
def test_delete_document(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc1 = _make_document(document_id="d1", file_hash="h1", chunk_count=2)
    c1 = [_make_chunk("d1-0", "d1", chunk_index=0), _make_chunk("d1-1", "d1", chunk_index=1)]
    s.add_document(doc1, c1, _make_vectors(2))

    doc2 = _make_document(document_id="d2", file_hash="h2", chunk_count=3)
    c2 = [_make_chunk(f"d2-{i}", "d2", chunk_index=i) for i in range(3)]
    s.add_document(doc2, c2, _make_vectors(3, base_seed=50))

    assert s.document_count == 2
    assert s.chunk_count == 5

    ok = s.delete_document("d1")
    assert ok is True
    assert s.document_count == 1
    assert s.chunk_count == 3
    assert not s.has_document_hash("h1")
    assert s.has_document_hash("h2")


# ---------------------------------------------------------------------------
# 22. 删除后索引重建正确
# ---------------------------------------------------------------------------
def test_delete_rebuild_index_correct(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc1 = _make_document(document_id="d1", file_hash="h1", chunk_count=3)
    c1 = [_make_chunk(f"d1-{i}", "d1", chunk_index=i) for i in range(3)]
    s.add_document(doc1, c1, _make_vectors(3, base_seed=10))

    doc2 = _make_document(document_id="d2", file_hash="h2", chunk_count=2)
    c2 = [_make_chunk(f"d2-{i}", "d2", chunk_index=i) for i in range(2)]
    s.add_document(doc2, c2, _make_vectors(2, base_seed=200))

    s.delete_document("d1")

    assert s._index.ntotal == 2  # type: ignore[attr-defined]
    # 重新加载后搜索仍正确
    q = s._vectors[0].copy()  # type: ignore[attr-defined]
    results = s.search(q, top_k=5)
    assert len(results) == 2
    # 全部应为 d2 的 chunk
    for ch, _ in results:
        assert ch.document_id == "d2"


def test_delete_nonexistent_returns_false(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    assert s.delete_document("not-exist") is False


# ---------------------------------------------------------------------------
# 23. 清空索引
# ---------------------------------------------------------------------------
def test_clear(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=3)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(3)]
    s.add_document(doc, chunks, _make_vectors(3))

    s.clear()
    assert s.document_count == 0
    assert s.chunk_count == 0
    assert s._index.ntotal == 0  # type: ignore[attr-defined]
    assert s._vectors is None  # type: ignore[attr-defined]
    # reload 后仍是空
    s2 = _make_store(tmp_path)
    s2.load()
    assert s2.document_count == 0


def test_clear_when_empty(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    s.clear()  # 不应抛错
    assert s.chunk_count == 0


# ---------------------------------------------------------------------------
# 24. manifest 模型不一致
# ---------------------------------------------------------------------------
def test_manifest_model_mismatch(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    # 用不同的 embedding_model 重建 store
    s2 = _make_store(tmp_path, siliconflow_embedding_model="Qwen/Other-Model")
    with pytest.raises(IndexIncompatibleError) as ei:
        s2.load()
    assert "embedding_model" in str(ei.value)


# ---------------------------------------------------------------------------
# 25. manifest 维度不一致
# ---------------------------------------------------------------------------
def test_manifest_dim_mismatch(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    s2 = _make_store(tmp_path, siliconflow_embedding_dimensions=512)
    with pytest.raises((IndexIncompatibleError, IndexCorruptedError)):
        s2.load()


# ---------------------------------------------------------------------------
# 26. chunk_count 不一致
# ---------------------------------------------------------------------------
def test_manifest_chunk_count_mismatch(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    # 篡改 manifest.chunk_count
    snapshot_dir = s.snapshots_dir / s._read_current_snapshot_id()  # type: ignore[attr-defined]
    manifest_path = snapshot_dir / FaissVectorStore.MANIFEST_FILENAME
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["chunk_count"] = 999
    manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    s2 = _make_store(tmp_path)
    with pytest.raises(IndexCorruptedError):
        s2.load()


# ---------------------------------------------------------------------------
# 27. index.ntotal 与 chunks 不一致
# ---------------------------------------------------------------------------
def test_index_ntotal_mismatch(tmp_path: Path):
    """直接构造 index.ntotal 与 chunks 数不一致的快照，验证 load 检测。"""
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=3)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(3)]
    s.add_document(doc, chunks, _make_vectors(3))

    # 手工写一个 ntotal 多 1 的 index（模拟索引损坏）
    snapshot_dir = s.snapshots_dir / s._read_current_snapshot_id()  # type: ignore[attr-defined]
    idx_path = snapshot_dir / FaissVectorStore.INDEX_FILENAME
    extra_idx = faiss.IndexFlatIP(DIM)
    extra_idx.add(_make_vectors(4))  # 4 vs 3 chunks
    faiss.write_index(extra_idx, str(idx_path))

    # manifest 也保持 3 chunks
    s2 = _make_store(tmp_path)
    with pytest.raises(IndexCorruptedError):
        s2.load()


# ---------------------------------------------------------------------------
# 28. CURRENT 指向不存在快照
# ---------------------------------------------------------------------------
def test_current_points_to_missing_snapshot(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    # 替换 CURRENT 指向不存在的 snapshot
    s.current_file.write_text("20990101T000000_deadbeef", encoding="utf-8")
    s2 = _make_store(tmp_path)
    with pytest.raises(IndexNotFoundError):
        s2.load()


# ---------------------------------------------------------------------------
# 29. JSONL 损坏
# ---------------------------------------------------------------------------
def test_corrupted_chunks_jsonl(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    # 篡改 chunks.jsonl
    snapshot_dir = s.snapshots_dir / s._read_current_snapshot_id()  # type: ignore[attr-defined]
    cp = snapshot_dir / FaissVectorStore.CHUNKS_FILENAME
    cp.write_text("not a json line\n", encoding="utf-8")

    s2 = _make_store(tmp_path)
    with pytest.raises(IndexCorruptedError):
        s2.load()


def test_corrupted_documents_jsonl(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    snapshot_dir = s.snapshots_dir / s._read_current_snapshot_id()  # type: ignore[attr-defined]
    dp = snapshot_dir / FaissVectorStore.DOCUMENTS_FILENAME
    dp.write_text("garbage\n", encoding="utf-8")

    s2 = _make_store(tmp_path)
    with pytest.raises(IndexCorruptedError):
        s2.load()


# ---------------------------------------------------------------------------
# 30. save 失败时旧快照仍可加载
# ---------------------------------------------------------------------------
def test_save_failure_old_snapshot_intact(tmp_path: Path, monkeypatch):
    s = _make_store(tmp_path)
    s.load()
    doc1 = _make_document(document_id="d1", file_hash="h1", chunk_count=2)
    c1 = [_make_chunk(f"d1-{i}", "d1", chunk_index=i) for i in range(2)]
    s.add_document(doc1, c1, _make_vectors(2))

    old_snapshot_id = s._read_current_snapshot_id()  # type: ignore[attr-defined]

    # 模拟 save 中段失败：让 faiss.write_index 抛错
    def _boom(*a, **k):  # noqa: ANN001
        raise OSError("disk full (simulated)")

    monkeypatch.setattr("faiss.write_index", _boom)

    doc2 = _make_document(document_id="d2", file_hash="h2", chunk_count=2)
    c2 = [_make_chunk(f"d2-{i}", "d2", chunk_index=i) for i in range(2)]
    with pytest.raises(OSError):
        s.add_document(doc2, c2, _make_vectors(2, base_seed=999))

    # CURRENT 应仍指向旧 snapshot
    s2 = _make_store(tmp_path)
    s2.load()
    assert s2._read_current_snapshot_id() == old_snapshot_id  # type: ignore[attr-defined]
    assert s2.document_count == 1
    assert s2.get_document_by_hash("h1") is not None


# ---------------------------------------------------------------------------
# 31. CURRENT 原子切换
# ---------------------------------------------------------------------------
def test_current_atomic_switch(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    doc1 = _make_document(document_id="d1", file_hash="h1", chunk_count=2)
    c1 = [_make_chunk(f"d1-{i}", "d1", chunk_index=i) for i in range(2)]
    s.add_document(doc1, c1, _make_vectors(2))

    snap1 = s._read_current_snapshot_id()  # type: ignore[attr-defined]
    assert snap1 is not None

    doc2 = _make_document(document_id="d2", file_hash="h2", chunk_count=2)
    c2 = [_make_chunk(f"d2-{i}", "d2", chunk_index=i) for i in range(2)]
    s.add_document(doc2, c2, _make_vectors(2, base_seed=20))

    snap2 = s._read_current_snapshot_id()  # type: ignore[attr-defined]
    assert snap2 is not None
    assert snap1 != snap2

    # 没有遗留 CURRENT.tmp
    assert not s.current_file.with_suffix(s.current_file.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# 32. 临时目录清理
# ---------------------------------------------------------------------------
def test_tmp_snapshot_cleanup_on_load(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=2)
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(2)]
    s.add_document(doc, chunks, _make_vectors(2))

    # 模拟遗留一个 .tmp 快照目录
    leftover = s.snapshots_dir / "20990101T000000_aaaaaaaa.tmp"
    leftover.mkdir(parents=True, exist_ok=True)
    assert leftover.exists()

    # 重新 load：应清理
    s2 = _make_store(tmp_path)
    s2.load()
    assert not leftover.exists()


# ---------------------------------------------------------------------------
# 33. prune_snapshots 保留指定数量
# ---------------------------------------------------------------------------
def test_prune_snapshots(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()

    # 创建 5 个快照
    for i in range(5):
        doc = _make_document(document_id=f"d{i}", file_hash=f"h{i}", chunk_count=1)
        chunks = [_make_chunk(f"c{i}", f"d{i}", chunk_index=0)]
        s.add_document(doc, chunks, _make_vectors(1, base_seed=i))

    all_snaps = s._list_snapshots()  # type: ignore[attr-defined]
    assert len(all_snaps) == 5

    current = s._read_current_snapshot_id()  # type: ignore[attr-defined]
    # 保留最近 2 个 + CURRENT 指向的不被删
    deleted = s.prune_snapshots(keep_last=2)
    kept = s._list_snapshots()  # type: ignore[attr-defined]
    # current 仍在
    assert current in kept
    assert len(kept) == 3  # 2 + current
    assert deleted == 2


# ---------------------------------------------------------------------------
# 34. 中文 metadata 保存和读取不乱码
# ---------------------------------------------------------------------------
def test_chinese_metadata_roundtrip(tmp_path: Path):
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(
        document_id="doc-zh",
        file_hash="hash-zh",
        file_name="国家基本药物目录.docx",
        original_file_name="国家基本药物目录（2026 年版）(OCR).docx",
        chunk_count=3,
    )
    chunks = [
        _make_chunk("c0", "doc-zh", content="化学药品包括抗感染药等。", chunk_index=0, heading="第一章"),
        _make_chunk("c1", "doc-zh", content="中成药部分包括内科用药。", chunk_index=1, heading="第二章"),
        _make_chunk("c2", "doc-zh", content="中药饮片部分略。", chunk_index=2, heading="第三章"),
    ]
    s.add_document(doc, chunks, _make_vectors(3))

    s2 = _make_store(tmp_path)
    s2.load()
    assert s2.document_count == 1
    d = s2.get_document_by_hash("hash-zh")
    assert d.original_file_name == "国家基本药物目录（2026 年版）(OCR).docx"
    assert d.file_name == "国家基本药物目录.docx"
    for i, ch in enumerate(s2._chunks):  # type: ignore[attr-defined]
        assert ch.content == chunks[i].content
        assert ch.heading == chunks[i].heading


# ---------------------------------------------------------------------------
# 35. 额外：unloaded 状态操作应抛错
# ---------------------------------------------------------------------------
def test_unloaded_operations_raise(tmp_path: Path):
    s = _make_store(tmp_path)
    # 未 load
    with pytest.raises(VectorStoreError):
        s.search(_make_vector(0), top_k=1)
    with pytest.raises(VectorStoreError):
        s.add_document(
            _make_document(chunk_count=1),
            [_make_chunk("c0", "doc-1", chunk_index=0)],
            _make_vectors(1),
        )
    with pytest.raises(VectorStoreError):
        s.save()
    with pytest.raises(VectorStoreError):
        s.delete_document("d1")
    with pytest.raises(VectorStoreError):
        s.clear()


# ===========================================================================
# 兼容性：existing 测试不得回归
# ===========================================================================
def test_chunk_count_auto_corrected_on_add(tmp_path: Path):
    """document.chunk_count 与实际不一致时，应自动校正（不抛错）。"""
    s = _make_store(tmp_path)
    s.load()
    doc = _make_document(chunk_count=999)  # 故意不一致
    chunks = [_make_chunk(f"c{i}", doc.document_id, chunk_index=i) for i in range(3)]
    s.add_document(doc, chunks, _make_vectors(3))
    # reload 后 chunk_count 应为 3
    s2 = _make_store(tmp_path)
    s2.load()
    assert s2.get_document_by_id(doc.document_id).chunk_count == 3