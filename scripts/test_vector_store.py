"""FaissVectorStore 调试脚本（手工运行）。

仅使用人工构造的 4 条文本和固定小向量，不调用远程 API。

输出：
- 文档数量
- chunk 数量
- index.ntotal
- 保存路径（临时目录）
- 当前 snapshot_id
- 一次搜索结果的 chunk_id 和 score
- 重新加载后的相同搜索结果
- 不输出 API Key
- 不写入正式 storage/indexes，使用临时目录
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402
from config.settings import Settings  # noqa: E402

from rag.models import DocumentChunk, DocumentInfo  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402


DIM = 1024


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.uniform(-1.0, 1.0, size=DIM).astype(np.float32)
    v /= max(np.linalg.norm(v), 1e-12)
    return v


def _make_doc_and_chunks() -> tuple[DocumentInfo, list[DocumentChunk], np.ndarray]:
    doc_id = "doc-debug"
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{i}",
            document_id=doc_id,
            content=f"调试文本 {i}",
            source_name="debug.txt",
            chunk_index=i,
        )
        for i in range(4)
    ]
    vectors = np.stack([_vec(seed=i + 1) for i in range(4)])
    doc = DocumentInfo(
        document_id=doc_id,
        file_name="debug.txt",
        original_file_name="调试文档.txt",
        file_type="txt",
        file_hash="debug-hash-0001",
        file_size=4096,
        created_at=datetime.now(timezone.utc),
        chunk_count=4,
    )
    return doc, chunks, vectors


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    setup_logging(level="WARNING")

    settings = Settings()

    tmp_dir = Path(tempfile.mkdtemp(prefix="firstrag_debug_"))
    index_dir = tmp_dir / "indexes"
    print(f"[INFO] 使用临时目录：{index_dir}")

    try:
        store = FaissVectorStore(settings=settings, index_dir=index_dir)
        store.load()

        doc, chunks, vectors = _make_doc_and_chunks()
        store.add_document(doc, chunks, vectors)

        _print_section("当前状态")
        print(f"  文档数量   : {store.document_count}")
        print(f"  chunk 数量 : {store.chunk_count}")
        print(f"  index.ntotal: {store._index.ntotal}")  # type: ignore[attr-defined]
        print(f"  保存路径   : {store.index_dir}")
        print(f"  snapshot_id: {store._read_current_snapshot_id()}")  # type: ignore[attr-defined]

        _print_section("首次搜索")
        query = _vec(seed=3)  # 与 chunk-2 最接近
        results = store.search(query, top_k=3)
        for i, (chunk, score) in enumerate(results, 1):
            print(f"  [S{i}] chunk_id={chunk.chunk_id} score={score:.6f}")

        _print_section("重新加载后再次搜索")
        # 重新构造 store 实例模拟重启
        store2 = FaissVectorStore(settings=settings, index_dir=index_dir)
        store2.load()
        results2 = store2.search(query, top_k=3)
        for i, (chunk, score) in enumerate(results2, 1):
            print(f"  [S{i}] chunk_id={chunk.chunk_id} score={score:.6f}")

        _print_section("一致性比对")
        ids1 = [c.chunk_id for c, _ in results]
        ids2 = [c.chunk_id for c, _ in results2]
        scores1 = [round(s, 6) for _, s in results]
        scores2 = [round(s, 6) for _, s in results2]
        print(f"  chunk_ids 一致 : {ids1 == ids2}")
        print(f"  scores   一致 : {scores1 == scores2}")

        _print_section("结论")
        print("  调试脚本执行通过；不涉及远程 API 与真实业务数据。")
        return 0
    finally:
        # 清理临时目录
        import shutil

        try:
            shutil.rmtree(tmp_dir)
            print(f"[INFO] 已清理临时目录：{tmp_dir}")
        except OSError as exc:
            print(f"[WARN] 清理失败：{exc}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)