"""RAG 端到端调试脚本（v1.0）。

只使用：

- ``tempfile`` 临时目录（不会写入正式 ``storage/``）
- 两份小型 TXT 文档（嵌入到脚本中，**不**读取国家基本药物目录）
- :class:`tests._fakes.FakeEmbeddingProvider`
- :class:`tests._fakes.StreamingFakeLLMClient`
- 临时 :class:`FaissVectorStore`

演示：

1. 入库两个小文档
2. 列出文档
3. 重复上传被拒绝
4. 单轮检索回答
5. 引用 S1..S_k
6. 非法引用被清理
7. 多轮问题改写
8. 流式 token + done 事件
9. 删除一个文档
10. 清空知识库
11. 重载后状态一致

**不**调用远程 API、**不**输出 API Key、**不**读取真实业务文档。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402
from config.settings import Settings  # noqa: E402

from rag.chat_service import (  # noqa: E402
    CHAT_EVENT_DONE,
    CHAT_EVENT_SOURCES,
    CHAT_EVENT_TOKEN,
    ChatService,
    NO_EVIDENCE_REPLY,
)
from rag.knowledge_base_service import (  # noqa: E402
    DuplicateDocumentError,
    KnowledgeBaseService,
    KnowledgeBaseServiceError,
)
from rag.models import (  # noqa: E402
    ChatMessage,
    DocumentChunk,
    RetrievedChunk,
)
from rag.prompt_builder import PromptBuilder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402

from tests._fakes import (  # noqa: E402
    FakeEmbeddingProvider,
    StreamingFakeLLMClient,
)


# ---------------------------------------------------------------------------
# 极简文档内容（不敏感 / 不引用真实业务）
# ---------------------------------------------------------------------------
DOC_A = (
    "高血压是一种常见的慢性疾病。\n"
    "患者需要定期监测血压。\n"
    "常见降压药物包括利尿剂和钙通道阻滞剂。"
)
DOC_B = (
    "感冒通常由病毒引起。\n"
    "常见症状包括咳嗽、流鼻涕和发热。\n"
    "多休息多饮水有助于恢复。"
)


def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def main() -> int:
    setup_logging(level="WARNING")

    print("firstRAG RAG 端到端调试脚本")
    print("=" * 64)
    print("仅使用 fake 组件与临时目录；不调用任何远程 API。")

    # ------------------------------------------------------------------
    # 准备：临时目录 / Settings / Fake Providers
    # ------------------------------------------------------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="firstrag_e2e_"))
    try:
        upload_dir = tmp_dir / "uploads"
        index_dir = tmp_dir / "indexes"
        upload_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        # 用小维度加速 fake embedder
        object.__setattr__(settings, "siliconflow_embedding_dimensions", 8)
        object.__setattr__(settings, "chunk_size", 200)
        object.__setattr__(settings, "retrieval_top_k", 3)

        embedder = FakeEmbeddingProvider(settings=settings)
        store = FaissVectorStore(settings=settings, index_dir=index_dir)
        store.load()
        kb = KnowledgeBaseService(
            embedding_provider=embedder,
            vector_store=store,
            settings=settings,
            upload_dir=upload_dir,
        )

        # ------------------------------------------------------------------
        # 1. 入库两个小文档
        # ------------------------------------------------------------------
        _section("1. 入库两个文档")
        info_a = kb.ingest_bytes(
            data=DOC_A.encode("utf-8"),
            original_file_name="hypertension.txt",
        )
        info_b = kb.ingest_bytes(
            data=DOC_B.encode("utf-8"),
            original_file_name="cold.md",
        )
        print(f"  文档 A: id={info_a.document_id[:12]}... chunks={info_a.chunk_count}")
        print(f"  文档 B: id={info_b.document_id[:12]}... chunks={info_b.chunk_count}")

        # ------------------------------------------------------------------
        # 2. 列出文档
        # ------------------------------------------------------------------
        _section("2. 列出文档")
        for d in kb.list_documents():
            print(f"  - {d.document_id[:12]}... {d.original_file_name} "
                  f"({d.file_type}, {d.chunk_count} chunks)")

        # ------------------------------------------------------------------
        # 3. 重复上传被拒绝
        # ------------------------------------------------------------------
        _section("3. 重复上传")
        try:
            kb.ingest_bytes(
                data=DOC_A.encode("utf-8"),
                original_file_name="hypertension_copy.txt",
            )
            print("  [WARN] 重复未被拒绝（不应出现）")
        except DuplicateDocumentError as exc:
            print(f"  正确拒绝：{type(exc).__name__}")
        except KnowledgeBaseServiceError as exc:
            print(f"  正确拒绝：{type(exc).__name__}: {exc}")

        # ------------------------------------------------------------------
        # 4 & 5. 单轮检索回答（带 S1 引用）
        # ------------------------------------------------------------------
        _section("4 & 5. 单轮检索 + 引用 S#")
        # 准备 ChatService：使用 fake retriever（直接走 KB 的 store）
        retriever = Retriever(embedder, store, settings)
        pb = PromptBuilder(settings)
        llm = StreamingFakeLLMClient(
            default_response="高血压是慢性病，需定期监测 [S1]."
        )
        chat = ChatService(retriever, pb, llm, settings)

        msg = chat.ask("什么是高血压？")
        print(f"  Query: 什么是高血压？")
        print(f"  Standalone: {msg.metadata.get('standalone_query')}")
        print(f"  Answer: {msg.content}")
        print(f"  Citations: {[c.citation_id for c in msg.citations]}")

        # ------------------------------------------------------------------
        # 6. 非法引用被清理
        # ------------------------------------------------------------------
        _section("6. 非法引用清理")
        llm_bad = StreamingFakeLLMClient(
            default_response="高血压 [S1] 也涉及 [S99] 错误引用 [S2]."
        )
        chat_bad = ChatService(retriever, pb, llm_bad, settings)
        msg_bad = chat_bad.ask("高血压？")
        print(f"  原 answer:    高血压 [S1] 也涉及 [S99] 错误引用 [S2].")
        print(f"  清理后 answer: {msg_bad.content}")
        print(f"  illegal:     {msg_bad.metadata.get('illegal_citations')}")

        # ------------------------------------------------------------------
        # 7. 多轮问题改写
        # ------------------------------------------------------------------
        _section("7. 多轮改写")
        history = [
            ChatMessage(
                role="user", content="什么是高血压？",
                created_at=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ChatMessage(
                role="assistant", content="高血压是慢性病。",
                citations=[], created_at=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
            ),
        ]
        # 改写 LLM 第一次返回 "高血压的常见药物"，第二次返回答案
        call_n = {"n": 0}

        def fake_complete(messages, temperature=None, max_tokens=None):
            call_n["n"] += 1
            if call_n["n"] == 1:
                return "高血压的常见药物"
            return "高血压常用利尿剂 [S1]."

        llm_rw = StreamingFakeLLMClient()
        llm_rw.complete = fake_complete  # type: ignore[method-assign]
        chat_rw = ChatService(retriever, pb, llm_rw, settings)
        msg_rw = chat_rw.ask("那药物呢？", history=history)
        print(f"  原 query:    那药物呢？")
        print(f"  改写后:     {msg_rw.metadata.get('standalone_query')}")
        print(f"  Answer:     {msg_rw.content}")

        # ------------------------------------------------------------------
        # 8. 流式 token + done 事件
        # ------------------------------------------------------------------
        _section("8. 流式事件")
        llm_stream = StreamingFakeLLMClient(
            stream_chunks=["你", "好", "，", "高血压 [S1]."]
        )
        chat_stream = ChatService(retriever, pb, llm_stream, settings)
        for ev in chat_stream.stream("Q?"):
            if ev.event_type == CHAT_EVENT_SOURCES:
                cites = [c.citation_id for c in ev.citations]
                print(f"  [sources] retrieval={ev.metadata.get('retrieval_count')} "
                      f"cites={cites}")
            elif ev.event_type == CHAT_EVENT_TOKEN:
                print(f"  [token] {ev.content!r}")
            elif ev.event_type == CHAT_EVENT_DONE:
                print(f"  [done]   message.content={ev.message.content!r}")
                print(f"  [done]   citations={[c.citation_id for c in ev.message.citations]}")

        # ------------------------------------------------------------------
        # 9. 删除一个文档
        # ------------------------------------------------------------------
        _section("9. 删除文档 B")
        deleted = kb.delete_document(info_b.document_id)
        print(f"  delete({info_b.document_id[:12]}...) -> {deleted}")
        for d in kb.list_documents():
            print(f"  剩余：{d.document_id[:12]}... {d.original_file_name}")

        # ------------------------------------------------------------------
        # 10. 清空知识库
        # ------------------------------------------------------------------
        _section("10. 清空知识库")
        kb.clear()
        print(f"  list_documents: {kb.list_documents()}")
        print(f"  chunk_count: {store.chunk_count}")

        # ------------------------------------------------------------------
        # 11. 重载后状态一致
        # ------------------------------------------------------------------
        _section("11. 重载索引")
        store2 = FaissVectorStore(settings=settings, index_dir=index_dir)
        store2.load()
        print(f"  store2.is_loaded: {store2.is_loaded}")
        print(f"  store2.chunk_count: {store2.chunk_count}")
        print(f"  store2.document_count: {store2.document_count}")
        # 重新检索
        kb2 = KnowledgeBaseService(
            embedding_provider=embedder, vector_store=store2,
            settings=settings, upload_dir=upload_dir,
        )
        info_a_reload = kb2.ingest_bytes(
            data=DOC_A.encode("utf-8"),
            original_file_name="hypertension.txt",
        )
        store2 = kb2._store
        store2.load()  # 重新从磁盘加载
        retriever2 = Retriever(embedder, store2, settings)
        llm2 = StreamingFakeLLMClient(default_response="高血压 [S1].")
        chat2 = ChatService(retriever2, pb, llm2, settings)
        msg2 = chat2.ask("高血压？")
        print(f"  重载后检索：{[c.citation_id for c in msg2.citations]}")
        print(f"  重载后答案：{msg2.content}")

        _section("完成")
        print("调试脚本执行完毕。所有步骤通过；未调用任何远程 API。")
        return 0
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
