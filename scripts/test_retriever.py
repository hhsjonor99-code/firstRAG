"""Retriever + PromptBuilder 调试脚本（手工运行）。

只使用 FakeEmbeddingProvider、临时 FaissVectorStore、人工 chunks 和固定向量，
不调用任何远程 API。

输出：
- 查询文本（不输出 embedding 原始值）
- top_k
- 返回结果数量
- S1..Sk 与对应 chunk_id / score
- 格式化后的来源位置
- Prompt 长度（字符数）
- 非法引用过滤示例
- 重载索引后的检索结果是否一致
"""

from __future__ import annotations

import shutil
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
from rag.models import ChatMessage, DocumentChunk, DocumentInfo, RetrievedChunk  # noqa: E402
from rag.prompt_builder import PromptBuilder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402


DIM = 1024


# ---------------------------------------------------------------------------
# FakeEmbeddingProvider（与 tests/test_retriever.py 一致）
# ---------------------------------------------------------------------------
class FakeEmbeddingProvider:
    """Fake Embedding：把 query 映射到一个固定的 dim 维向量。"""

    def __init__(self, dim: int = DIM, mode: str = "select:3") -> None:
        self._dim = dim
        self._mode = mode

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text).reshape(1, self._dim)

    def _vec(self, text):
        if self._mode.startswith("select:"):
            seed = int(self._mode.split(":", 1)[1])
        else:
            seed = abs(hash(text)) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        v = rng.uniform(-1.0, 1.0, size=self._dim).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-12)
        return v


def _make_chunk(chunk_id: str, document_id: str, idx: int, **kwargs) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=f"文本内容-{idx}",
        source_name="国家基本药物目录.docx",
        chunk_index=idx,
        **kwargs,
    )


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    setup_logging(level="WARNING")

    settings = Settings()

    tmp_dir = Path(tempfile.mkdtemp(prefix="firstrag_retriever_debug_"))
    index_dir = tmp_dir / "indexes"
    print(f"[INFO] 使用临时目录：{index_dir}")

    try:
        # 1. 构造 FaissVectorStore 并入库
        store = FaissVectorStore(settings=settings, index_dir=index_dir)
        store.load()

        chunks = [
            _make_chunk(f"chunk-{i}", "doc-1", i,
                        heading=f"第 {i+1} 章" if i < 3 else None,
                        block_type="paragraph",
                        paragraph_start=i + 1,
                        paragraph_end=i + 1)
            for i in range(5)
        ]
        # 向量：种子 i 与 FakeEmbedding 的 select:i 对齐
        vectors_list = []
        for i in range(5):
            rng = np.random.default_rng(i)
            v = rng.uniform(-1.0, 1.0, size=DIM).astype(np.float32)
            v /= max(np.linalg.norm(v), 1e-12)
            vectors_list.append(v)
        vectors = np.stack(vectors_list)
        doc = DocumentInfo(
            document_id="doc-1",
            file_name="demo.docx",
            original_file_name="示例.docx",
            file_type="docx",
            file_hash="demo-hash-1",
            file_size=1024,
            created_at=datetime.now(timezone.utc),
            chunk_count=5,
        )
        store.add_document(doc, chunks, vectors)

        embed = FakeEmbeddingProvider(dim=DIM, mode="select:3")
        retriever = Retriever(embed, store, settings)

        # 2. 检索
        query = "测试查询"
        top_k = 3
        _print_section("检索")
        print(f"  查询文本 : {query}")
        print(f"  top_k    : {top_k}")

        results = retriever.retrieve(query, top_k=top_k)
        print(f"  返回数量 : {len(results)}")
        print("  S 编号 / chunk_id / score :")
        for rc in results:
            print(f"    [{rc.citation_id}] {rc.chunk.chunk_id}  score={rc.score:.6f}")

        # 3. Prompt 构造
        _print_section("Prompt 长度")
        pb = PromptBuilder(settings)
        answer_msgs = pb.build_answer_messages(query, results)
        rewrite_msgs = pb.build_rewrite_messages(query, [])
        print(f"  回答 system prompt 长度 : {len(answer_msgs[0]['content'])} 字符")
        print(f"  回答 user   prompt 长度 : {len(answer_msgs[1]['content'])} 字符")
        print(f"  改写 system prompt 长度 : {len(rewrite_msgs[0]['content'])} 字符")
        print(f"  改写 user   prompt 长度 : {len(rewrite_msgs[1]['content'])} 字符")

        # 4. 格式化来源位置
        _print_section("格式化后的来源")
        print(pb.format_sources(results))

        # 5. 非法引用过滤示例
        _print_section("非法引用过滤示例")
        sample = "参见 [S1] 和 [S8] 与 [S99]，结论见 [S2]。"
        cleaned, illegal = pb.sanitize_invalid_citations(sample, results)
        print(f"  原始答案 : {sample}")
        print(f"  清洗后   : {cleaned}")
        print(f"  非法引用 : {illegal}")

        # 6. 重载索引后检索一致性
        _print_section("重载索引后检索一致性")
        store2 = FaissVectorStore(settings=settings, index_dir=index_dir)
        store2.load()
        retriever2 = Retriever(embed, store2, settings)
        results2 = retriever2.retrieve(query, top_k=top_k)
        ids1 = [rc.chunk.chunk_id for rc in results]
        ids2 = [rc.chunk.chunk_id for rc in results2]
        scores1 = [round(rc.score, 6) for rc in results]
        scores2 = [round(rc.score, 6) for rc in results2]
        print(f"  chunk_ids 一致 : {ids1 == ids2}")
        print(f"  scores   一致 : {scores1 == scores2}")

        _print_section("结论")
        print("  调试脚本执行通过；不涉及远程 API 与真实业务数据。")
        return 0
    finally:
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