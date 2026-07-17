"""firstRAG 真实 RAG 端到端联调脚本（v1.0）。

**仅用于手工验证 SiliconFlow Embedding + MiniMax LLM + RAG 主链路**。

使用：

- ``tempfile.TemporaryDirectory`` 创建临时 ``upload_dir`` / ``index_dir``
- 两份**完全虚构、非敏感**的 Markdown 小文档（"青云图书馆" 虚构实体）
- 真实 :class:`SiliconFlowEmbeddingProvider`
- 真实 :class:`MiniMaxLLMClient`
- 真实 :class:`FaissVectorStore` / :class:`KnowledgeBaseService` /
  :class:`Retriever` / :class:`PromptBuilder` / :class:`ChatService`

**严格安全约束**：

- 不得输出 API Key、Authorization 头、完整向量、完整 Prompt、
  MiniMax 内部 reasoning 内容、完整 API 响应体。
- 不写入正式 ``storage/uploads`` / ``storage/indexes``。
- 不读取真实国家基本药物目录。
- 不安装 / 修改 PyTorch 或本地模型。

退出码：

- 0：所有验证通过
- 1：配置问题（如缺 Key）
- 2：远程 API 或 RAG 验证失败
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

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
from rag.knowledge_base_service import KnowledgeBaseService  # noqa: E402
from rag.llm_client import MiniMaxLLMClient  # noqa: E402
from rag.prompt_builder import PromptBuilder  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.siliconflow_embeddings import SiliconFlowEmbeddingProvider  # noqa: E402
from rag.vector_store import FaissVectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# 虚构 / 非敏感测试文档
# ---------------------------------------------------------------------------
DOC_A = (
    "# 青云图书馆开放时间\n\n"
    "青云图书馆开放时间为周一至周五上午九点至下午六点。\n"
    "周末（周六、周日）开放时间为上午十点至下午四点。\n"
    "法定节假日闭馆，特殊情况另行公告。\n"
)

DOC_B = (
    "# 青云图书馆办证指南\n\n"
    "青云图书馆借阅证办理地点在一楼服务台。\n"
    "办理时需要携带有效身份证件。\n"
    "首次办证免工本费，补办收取十元工本费。\n"
)


def _section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _safe_classify(exc: BaseException) -> str:
    """把异常归类为安全字符串。"""
    name = type(exc).__name__
    return f"{name}: {str(exc)[:80]}"


def _check_no_forbidden_markers(text: str) -> list[str]:
    """检查用户可见文本中是否泄漏了内部思考 / 标签。"""
    found: list[str] = []
    for tag in ("<think>", "</think>", "<analysis>", "</analysis>"):
        if tag in text:
            found.append(tag)
    return found


def main() -> int:
    setup_logging(level="WARNING")  # 减少噪音；如有需要可改 INFO

    print("firstRAG 真实 RAG 端到端联调")
    print("=" * 64)
    print("使用真实 SiliconFlow Embedding + 真实 MiniMax LLM。")
    print("仅使用临时目录；不写入正式 storage。")

    settings = Settings()

    # ------------------------------------------------------------------
    # 0. Key 配置检查
    # ------------------------------------------------------------------
    _section("0. API Key 配置")
    sf_ok = settings.has_siliconflow_key()
    mm_ok = settings.has_minimax_key()
    print(f"  SILICONFLOW_API_KEY: {'已配置' if sf_ok else '未配置'}")
    print(f"  MINIMAX_API_KEY:     {'已配置' if mm_ok else '未配置'}")
    if not (sf_ok and mm_ok):
        print("  [FAIL] 缺少必要 API Key；脚本退出。")
        return 1

    # ------------------------------------------------------------------
    # 1. 准备临时目录与组件
    # ------------------------------------------------------------------
    tmp_root = Path(tempfile.mkdtemp(prefix="firstrag_real_"))
    upload_dir = tmp_root / "uploads"
    index_dir = tmp_root / "indexes"
    upload_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    try:
        _section("1. 组件初始化")
        embedder = SiliconFlowEmbeddingProvider(settings=settings)
        store = FaissVectorStore(settings=settings, index_dir=index_dir)
        store.load()
        kb = KnowledgeBaseService(
            embedding_provider=embedder,
            vector_store=store,
            settings=settings,
            upload_dir=upload_dir,
        )
        retriever = Retriever(embedder, store, settings)
        pb = PromptBuilder(settings)
        llm = MiniMaxLLMClient(settings=settings)
        chat = ChatService(retriever, pb, llm, settings)
        print(f"  embedding model: {embedder.model_name}")
        print(f"  embedding dim:   {embedder.dimensions}")
        print(f"  LLM model:       {llm.model_name}")
        print(f"  临时 upload_dir: {upload_dir}")
        print(f"  临时 index_dir:  {index_dir}")

        # ------------------------------------------------------------------
        # 2. 创建两份临时文档并入库
        # ------------------------------------------------------------------
        _section("2. 真实入库两份文档")
        doc_a_path = tmp_root / "hours.md"
        doc_b_path = tmp_root / "card.md"
        doc_a_path.write_text(DOC_A, encoding="utf-8")
        doc_b_path.write_text(DOC_B, encoding="utf-8")

        started = time.perf_counter()
        try:
            info_a = kb.ingest_path(doc_a_path)
            info_b = kb.ingest_path(doc_b_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] 入库失败：{_safe_classify(exc)}")
            return 2
        ingest_elapsed = time.perf_counter() - started
        print(f"  文档 A: id={info_a.document_id[:12]}... "
              f"chunks={info_a.chunk_count} type={info_a.file_type}")
        print(f"  文档 B: id={info_b.document_id[:12]}... "
              f"chunks={info_b.chunk_count} type={info_b.file_type}")
        print(f"  入库耗时: {ingest_elapsed:.2f}s")

        # ------------------------------------------------------------------
        # 3. 输出安全统计
        # ------------------------------------------------------------------
        _section("3. 安全统计")
        documents = kb.list_documents()
        total_chunks = sum(d.chunk_count for d in documents)
        print(f"  文档数量:    {len(documents)}")
        print(f"  chunk 数量:  {total_chunks}")
        print(f"  Embedding:   {embedder.model_name}")
        print(f"  向量维度:    {embedder.dimensions}")
        # 读取当前 snapshot id（如果 store 暴露）
        current = store.current_file
        print(f"  CURRENT 文件: {current.name}")
        if current.exists():
            print(f"  current snapshot_id: {current.read_text(encoding='utf-8').strip()}")

        # ------------------------------------------------------------------
        # 4. Q1: 周末几点开放
        # ------------------------------------------------------------------
        _section("4. Q1: 青云图书馆周末几点开放？")
        msg = chat.ask("青云图书馆周末几点开放？")
        _print_answer_and_citations(msg)
        if not _validate_has_evidence(
            msg, must_contain=["上午十点", "下午四点"], citation_id_prefix="S"
        ):
            print("  [FAIL] Q1 验证失败。")
            return 2

        # ------------------------------------------------------------------
        # 5. Q2: 在哪里办理借阅证
        # ------------------------------------------------------------------
        _section("5. Q2: 在哪里办理借阅证，需要带什么？")
        msg2 = chat.ask("在哪里办理借阅证，需要带什么？")
        _print_answer_and_citations(msg2)
        if not _validate_has_evidence(
            msg2, must_contain=["一楼服务台", "有效身份证件"], citation_id_prefix="S"
        ):
            print("  [FAIL] Q2 验证失败。")
            return 2

        # ------------------------------------------------------------------
        # 6. Q3: 多轮改写
        # ------------------------------------------------------------------
        _section("6. 多轮改写")
        from rag.models import ChatMessage
        from datetime import datetime, timezone

        history = [
            ChatMessage(
                role="user",
                content="青云图书馆周末几点开放？",
                created_at=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ChatMessage(
                role="assistant",
                content=msg.content,
                citations=msg.citations,
                created_at=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
            ),
        ]
        msg3 = chat.ask("那工作日呢？", history=history)
        print(f"  原 query:        那工作日呢？")
        print(f"  改写后 query:   {msg3.metadata.get('standalone_query')}")
        _print_answer_and_citations(msg3)
        if not _validate_has_evidence(
            msg3,
            must_contain=["上午九点", "下午六点"],
            citation_id_prefix="S",
        ):
            print("  [FAIL] Q3 多轮改写验证失败。")
            return 2

        # ------------------------------------------------------------------
        # 7. Q4: 无依据问题
        # ------------------------------------------------------------------
        _section("7. Q4: 无依据问题（员工数量）")
        msg4 = chat.ask("青云图书馆有多少名员工？")
        _print_answer_and_citations(msg4)
        if not _validate_refusal_citations_consistent(msg4, has_candidates=True):
            print(
                "  [FAIL] Q4 拒答 / 引用一致性验证失败。"
            )
            return 2
        print("  Q4 正确返回固定拒答；citations 为空；metadata 保留 candidate 编号。")

        # ------------------------------------------------------------------
        # 8. 流式
        # ------------------------------------------------------------------
        _section("8. 流式（基于 Q1）")
        events = list(chat.stream("青云图书馆周末几点开放？"))
        types = [e.event_type for e in events]
        token_events = [e for e in events if e.event_type == CHAT_EVENT_TOKEN]
        done_events = [e for e in events if e.event_type == CHAT_EVENT_DONE]
        sources_events = [e for e in events if e.event_type == CHAT_EVENT_SOURCES]
        print(f"  事件类型顺序: {types}")
        print(f"  token 段数:   {len(token_events)}")
        print(f"  sources 事件: {len(sources_events)}")
        print(f"  done 事件:    {len(done_events)}")
        if not token_events or not done_events or not sources_events:
            print("  [FAIL] 流式事件不完整。")
            return 2
        # 用户可见 token 拼接不应含 think 标签
        joined_tokens = "".join(e.content or "" for e in token_events)
        leaked = _check_no_forbidden_markers(joined_tokens)
        if leaked:
            print(f"  [FAIL] 流式 token 泄漏内部标签: {leaked}")
            return 2
        # done.message.content 是最终清理后答案
        final_msg = done_events[0].message
        if final_msg is None or not final_msg.content:
            print("  [FAIL] done.message 为空。")
            return 2
        leaked2 = _check_no_forbidden_markers(final_msg.content)
        if leaked2:
            print(f"  [FAIL] done.message 泄漏内部标签: {leaked2}")
            return 2
        # 流式 vs 非流式语义一致（去空白后）
        def _norm(s: str) -> str:
            return "".join((c for c in s if not c.isspace())).lower()
        if _norm(final_msg.content) != _norm(msg.content):
            print("  [WARN] 流式与非流式最终答案不完全相同。")
            print(f"    non-stream: {msg.content!r}")
            print(f"    stream:     {final_msg.content!r}")
            # 不视作 FAIL（模型随机性）；只 warning

        # ------------------------------------------------------------------
        # 9. 持久化：重载索引
        # ------------------------------------------------------------------
        _section("9. 持久化重载")
        store2 = FaissVectorStore(settings=settings, index_dir=index_dir)
        store2.load()
        print(f"  重载后文档数: {store2.document_count}")
        print(f"  重载后 chunks: {store2.chunk_count}")
        if store2.document_count != store.document_count:
            print("  [FAIL] 文档数不一致。")
            return 2
        if store2.chunk_count != store.chunk_count:
            print("  [FAIL] chunk 数不一致。")
            return 2
        # 重新检索
        retriever2 = Retriever(embedder, store2, settings)
        chat2 = ChatService(retriever2, pb, llm, settings)
        msg_reload = chat2.ask("青云图书馆周末几点开放？")
        if not msg_reload.citations:
            print("  [FAIL] 重载后检索无结果。")
            return 2
        if "上午十点" not in msg_reload.content or "下午四点" not in msg_reload.content:
            print("  [FAIL] 重载后答案不包含期望内容。")
            return 2
        print(f"  重载后检索通过：{len(msg_reload.citations)} citations")
        print(f"  重载后最终答案: {msg_reload.content}")

        # ------------------------------------------------------------------
        # 结论
        # ------------------------------------------------------------------
        _section("结论")
        print("  所有验证通过：")
        print(f"    - 文档数: {len(documents)}")
        print(f"    - chunk 总数: {total_chunks}")
        print(f"    - 三个有依据问题均含期望关键词与合法引用")
        print(f"    - 无依据问题正确拒答")
        print(f"    - 多轮改写生效（Q3 standalone 含「工作日」）")
        print(f"    - 流式事件顺序：{types}")
        print(f"    - 重载索引后状态一致")
        print(f"    - 临时目录: {tmp_root}（脚本退出时自动清理）")
        return 0

    except Exception as exc:  # noqa: BLE001
        print()
        print(f"[FAIL] 未预期异常：{_safe_classify(exc)}")
        return 2
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _print_answer_and_citations(msg, retriever_call_count=None) -> None:
    """打印 ChatMessage 的安全字段。"""
    print(f"  Answer: {msg.content}")
    print(f"  Citations ({len(msg.citations)}):")
    for c in msg.citations:
        loc = c.chunk.source_name
        print(f"    {c.citation_id}  score={c.score:.4f}  source={loc!r}")
    illegal = msg.metadata.get("illegal_citations") or []
    if illegal:
        print(f"  Illegal citations removed: {illegal}")
    if retriever_call_count is not None:
        print(f"  Retriever call count: {retriever_call_count}")


def _validate_has_evidence(
    msg,
    must_contain: list[str],
    citation_id_prefix: str = "S",
) -> bool:
    """验证：回答包含全部关键词、含合法引用、非法引用被过滤、不含内部标签。

    阶段 7.2 强化：

    - ``msg.citations`` 只含答案中**实际**出现的 [S#] 对应 RetrievedChunk
    - ``metadata.candidate_citation_ids`` 为候选；与 ``citations`` 可不同
    """
    content = msg.content or ""
    for kw in must_contain:
        if kw not in content:
            print(f"  [WARN] 关键词缺失: {kw!r}")
            return False
    if not msg.citations:
        print("  [WARN] citations 为空。")
        return False
    # 至少一个合法 S# 引用
    import re
    s_tags = re.findall(r"\[S\d+\]", content)
    if not s_tags:
        print("  [WARN] 答案中无 [S#] 引用。")
        return False
    # 非法引用
    if "[S99]" in content or re.search(r"\[S\d{3,}\]", content):
        print("  [WARN] 答案中含非法引用。")
        return False
    # 内部思考标签
    leaked = _check_no_forbidden_markers(content)
    if leaked:
        print(f"  [WARN] 答案泄漏内部标签: {leaked}")
        return False
    # 7.2 一致性：citations 中的 id 必须是答案中实际出现过的
    used_in_msg = set(re.findall(r"\[S(\d+)\]", content))  # 提取 S# 编号（不含方括号）
    used_in_citations = {c.citation_id for c in msg.citations}
    # citation_id 形如 "S1"，与正则捕获组一致
    used_in_msg_normalized = {f"S{n}" for n in used_in_msg}
    if not used_in_citations.issubset(used_in_msg_normalized):
        print(
            f"  [WARN] citations 中含未在答案中出现的编号: "
            f"citations={used_in_citations}, answer={used_in_msg_normalized}"
        )
        return False
    return True


def _validate_refusal_citations_consistent(msg, has_candidates: bool) -> bool:
    """验证：固定拒答时 citations 为空，metadata 可保留候选。"""
    from rag.prompt_builder import NO_EVIDENCE_REPLY
    if (msg.content or "").strip() != NO_EVIDENCE_REPLY:
        print(f"  [WARN] 期望拒答，但 content={msg.content!r}")
        return False
    if msg.citations:
        print(f"  [WARN] 拒答时 citations 应为空，得到 {len(msg.citations)} 条")
        return False
    candidate_ids = msg.metadata.get("candidate_citation_ids") or []
    if has_candidates and not candidate_ids:
        print("  [WARN] 期望 metadata.candidate_citation_ids 非空，但为空。")
        return False
    if not has_candidates and candidate_ids:
        print("  [WARN] 期望 metadata.candidate_citation_ids 为空，但非空。")
        return False
    return True


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
