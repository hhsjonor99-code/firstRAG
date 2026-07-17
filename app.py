"""firstRAG Streamlit 本地知识库 RAG 界面（v1.0）。

启动：

    streamlit run app.py

主要分区：

- 左侧边栏：API 配置状态、文件上传、文档列表、知识库统计、操作按钮
- 主区域：聊天历史、用户输入、引用面板

约束：

- 不实现 FastAPI；不修改底层 RAG 服务
- 不在 ``session_state`` / 页面 / 日志中输出 API Key
- 不在 UI 中实现解析 / 分块 / Embedding；统一走 :class:`KnowledgeBaseService`
- 引用面板数据源 = ``done.message.citations``，**不**使用 sources 候选
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Optional

import streamlit as st

# 让脚本既能以 ``streamlit run app.py`` 也能 ``python -m streamlit run app.py`` 启动
ROOT = __import__("pathlib").Path(__file__).resolve().parent
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from config.logging import get_logger, setup_logging  # noqa: E402
from rag.chat_service import (  # noqa: E402
    CHAT_EVENT_DONE,
    CHAT_EVENT_ERROR,
    CHAT_EVENT_REWRITE,
    CHAT_EVENT_SOURCES,
    CHAT_EVENT_TOKEN,
    NO_EVIDENCE_REPLY,
    ChatService,
)
from rag.llm_client import LLMError  # noqa: E402
from rag.models import (  # noqa: E402
    ChatMessage,
    DocumentInfo,
)
from rag.parsers import (  # noqa: E402
    EmptyDocumentError,
    UnsupportedScannedPDFError,
)

from ui import components, service_factory, state  # noqa: E402


# ---------------------------------------------------------------------------
# 启动配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="本地知识库 RAG",
    page_icon="📚",
    layout="wide",
)

# 启动 logging（不输出 Key）
setup_logging(level="INFO")

_LOG = get_logger("ui.app")


# ---------------------------------------------------------------------------
# 少量样式
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
.firstRAG-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #1f3a93;
    margin-bottom: 0.2rem;
}
.firstRAG-subtitle {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 1rem;
}
.firstRAG-citation-card {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    background-color: #f9fafb;
}
.firstRAG-citation-meta {
    color: #6b7280;
    font-size: 0.85rem;
}
.firstRAG-citation-id {
    color: #1f3a93;
    font-weight: 600;
}
.firstRAG-warning {
    color: #b45309;
}
.firstRAG-error {
    color: #b91c1c;
}
</style>
"""


def _inject_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 工具：file hash
# ---------------------------------------------------------------------------
def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# 入库单文件
# ---------------------------------------------------------------------------
def _ingest_one(kb, name: str, data: bytes) -> dict:
    """对单文件入库；返回 ``{"file_name", "ok", "info"|"error"}``。"""
    try:
        info = kb.ingest_bytes(data=data, original_file_name=name)
        return {"file_name": name, "ok": True, "info": info, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {
            "file_name": name,
            "ok": False,
            "info": None,
            "error": components.safe_error_message(exc),
        }


# ---------------------------------------------------------------------------
# 侧边栏：API 状态 + 上传 + 文档列表 + 统计
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    sid = service_factory.settings_signature()
    has_sf = service_factory.has_siliconflow_key()
    has_mm = service_factory.has_minimax_key()
    idx_dir = str(service_factory.default_index_dir())
    up_dir = str(service_factory.default_upload_dir())

    with st.sidebar:
        st.markdown(
            '<div class="firstRAG-title">📚 本地知识库 RAG</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="firstRAG-subtitle">基于 SiliconFlow Embedding + MiniMax LLM</div>',
            unsafe_allow_html=True,
        )

        # 1. API 状态
        st.subheader("API 配置")
        st.write(
            f"- SiliconFlow Embedding: **{components.api_key_status(has_sf)}**"
        )
        st.write(
            f"- MiniMax LLM: **{components.api_key_status(has_mm)}**"
        )
        if not (has_sf and has_mm):
            st.warning(
                "缺少 API Key：\n"
                "- 缺 SiliconFlow：可浏览已有知识库，但无法新文档入库。\n"
                "- 缺 MiniMax：可入库与检索，但无法生成回答。\n\n"
                "请在 `.env` 中设置 `SILICONFLOW_API_KEY` / `MINIMAX_API_KEY`。"
            )

        st.divider()

        # 2. 上传
        st.subheader("上传文档")
        if not has_sf:
            st.info("SiliconFlow 未配置，无法入库新文档。")
        else:
            uploaded = st.file_uploader(
                "选择文件（可多选）",
                type=["pdf", "docx", "txt", "md", "markdown"],
                accept_multiple_files=True,
                key="uploader",
            )
            if st.button("解析并建立索引", type="primary", use_container_width=True):
                if not uploaded:
                    state.push_notification("warning", "请先选择文件。")
                elif state.is_inflight("upload"):
                    state.push_notification("warning", "已有入库任务进行中，请稍候。")
                else:
                    state.set_inflight("upload")
                    try:
                        kb = service_factory.get_knowledge_base_service(
                            idx_dir, up_dir, sid
                        )
                        results: list[dict] = []
                        progress = st.progress(0.0, text="开始入库…")
                        for i, f in enumerate(uploaded):
                            data = f.getvalue()
                            file_hash = _hash_bytes(data)
                            if not state.remember_ingest_hash(file_hash):
                                results.append({
                                    "file_name": f.name,
                                    "ok": False,
                                    "info": None,
                                    "error": "本次会话内已入库过该文件，已跳过。",
                                })
                            else:
                                res = _ingest_one(kb, f.name, data)
                                results.append(res)
                            progress.progress(
                                (i + 1) / len(uploaded),
                                text=f"已处理 {i + 1} / {len(uploaded)}",
                            )
                        progress.empty()
                        summary = components.summarize_ingest_results(results)
                        if summary["failed"] == 0:
                            state.push_notification(
                                "success",
                                f"全部 {summary['success']} 个文件入库成功。",
                            )
                        elif summary["success"] == 0:
                            state.push_notification(
                                "error",
                                f"全部 {summary['failed']} 个文件入库失败。",
                            )
                        else:
                            state.push_notification(
                                "warning",
                                f"成功 {summary['success']} 个，失败 {summary['failed']} 个。",
                            )
                        # 逐个文件的提示
                        for r in results:
                            if r["ok"]:
                                state.push_notification(
                                    "info", f"✓ {r['file_name']} 入库成功"
                                )
                            else:
                                state.push_notification(
                                    "error", f"✗ {r['file_name']} 失败：{r['error']}"
                                )
                    finally:
                        state.set_inflight(None)

        st.divider()

        # 3. 已入库文档列表
        st.subheader("已入库文档")
        docs = service_factory.list_documents_safe()
        if not docs:
            st.info("知识库为空。")
        else:
            for d in docs:
                _render_document_row(d)

        st.divider()

        # 4. 知识库统计
        st.subheader("知识库统计")
        stats = service_factory.knowledge_base_stats_safe()
        st.metric("文档数量", stats.get("document_count", 0))
        st.metric("chunk 数量", stats.get("chunk_count", 0))
        st.metric("Embedding", stats.get("embedding_model", "—"))
        st.metric("向量维度", stats.get("embedding_dim", 0))
        idx_status = "已加载" if stats.get("index_loaded") else "未加载"
        st.write(f"索引状态: **{idx_status}**")

        st.divider()

        # 5. 操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空知识库", use_container_width=True, type="secondary"):
                if state.is_inflight("clear_kb"):
                    pass
                elif not state.confirm_clear_knowledge_base():
                    state.set_confirm_clear_kb(True)
                else:
                    # 二次确认：实际执行
                    state.set_confirm_clear_kb(False)
                    state.set_inflight("clear_kb")
                    try:
                        try:
                            store = service_factory.get_vector_store(idx_dir, sid)
                            store.clear()
                            state.push_notification("success", "知识库已清空。")
                        except Exception as exc:  # noqa: BLE001
                            state.push_notification(
                                "error",
                                f"清空失败：{components.safe_error_message(exc)}",
                            )
                    finally:
                        state.set_inflight(None)
        with col2:
            if st.button("清空对话", use_container_width=True, type="secondary"):
                if state.is_inflight("clear_chat"):
                    pass
                elif not state.confirm_clear_chat():
                    state.set_confirm_clear_chat(True)
                else:
                    state.set_confirm_clear_chat(False)
                    state.clear_chat_messages()
                    state.push_notification("info", "对话已清空。")

        # 二次确认提示
        if state.confirm_clear_knowledge_base():
            st.warning(
                "再次点击「清空知识库」以确认；该操作将删除全部索引与上传文件。"
            )
            if st.button("取消", key="cancel_clear_kb"):
                state.set_confirm_clear_kb(False)
        if state.confirm_clear_chat():
            st.warning("再次点击「清空对话」以确认。")
            if st.button("取消", key="cancel_clear_chat"):
                state.set_confirm_clear_chat(False)


def _render_document_row(d: DocumentInfo) -> None:
    """渲染单条文档行 + 删除按钮。"""
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown(
            f"**{d.original_file_name}**  \n"
            f"<span class='firstRAG-citation-meta'>"
            f"{d.file_type} · {components.format_file_size(d.file_size)} · "
            f"{d.chunk_count} chunks · {components.format_created_at(d.created_at)}"
            f"</span>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        if state.pending_delete_id() == d.document_id:
            if st.button("确认删除", key=f"del_ok_{d.document_id}", type="primary"):
                _do_delete(d.document_id)
        else:
            if st.button("删除", key=f"del_{d.document_id}"):
                state.request_delete(d.document_id)


def _do_delete(document_id: str) -> None:
    sid = service_factory.settings_signature()
    idx_dir = str(service_factory.default_index_dir())
    up_dir = str(service_factory.default_upload_dir())
    state.set_inflight("delete")
    try:
        try:
            kb = service_factory.get_knowledge_base_service(
                idx_dir, up_dir, sid
            )
            ok = kb.delete_document(document_id)
            if ok:
                state.push_notification("success", "文档已删除。")
            else:
                state.push_notification("warning", "文档不存在，已跳过。")
        except Exception as exc:  # noqa: BLE001
            state.push_notification(
                "error", f"删除失败：{components.safe_error_message(exc)}"
            )
    finally:
        state.cancel_delete()
        state.set_inflight(None)


# ---------------------------------------------------------------------------
# 引用面板
# ---------------------------------------------------------------------------
def render_citation_panel(citations) -> None:
    """渲染引用来源面板；citations 已最终筛选（不含 sources 候选）。"""
    if not citations:
        return
    st.markdown("**📎 引用来源**")
    for c in citations:
        view = components.build_citation_view(c)
        st.markdown(
            f"<div class='firstRAG-citation-card'>"
            f"<div><span class='firstRAG-citation-id'>[{view['citation_id']}]</span> "
            f"<strong>{view['source_name']}</strong></div>"
            f"<div class='firstRAG-citation-meta'>"
            f"位置: {view['location'] or '—'} · "
            f"相似度: {view['score']}"
            f"{(' · 标题: ' + view['heading']) if view['heading'] else ''}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander("查看原文片段", expanded=False):
            st.text(view["content"] or "")


# ---------------------------------------------------------------------------
# 聊天
# ---------------------------------------------------------------------------
def render_chat() -> None:
    state.ensure_session_state()
    sid = service_factory.settings_signature()
    has_sf = service_factory.has_siliconflow_key()
    has_mm = service_factory.has_minimax_key()
    idx_dir = str(service_factory.default_index_dir())

    st.markdown(
        '<div class="firstRAG-title">💬 智能问答</div>',
        unsafe_allow_html=True,
    )

    # 知识库提示
    stats = service_factory.knowledge_base_stats_safe()
    if stats.get("document_count", 0) == 0:
        st.info("知识库为空，请先在左侧上传文档。")

    # 配置缺失提示
    if not has_sf or not has_mm:
        st.warning(
            "当前缺少 API Key：\n"
            + ("- 缺少 SiliconFlow：无法检索（Embedding 不可用）。\n" if not has_sf else "")
            + ("- 缺少 MiniMax：无法生成回答。\n" if not has_mm else "")
        )

    # 渲染历史
    for msg in state.chat_messages():
        with st.chat_message(msg.role):
            st.markdown(msg.content or "")
            # 拒答 / 无引用：均不显示引用面板
            if msg.role == "assistant":
                if components.is_refusal(msg.content or ""):
                    pass
                elif msg.citations:
                    render_citation_panel(msg.citations)
                else:
                    # 既不是拒答，又无 citations：轻量提示
                    warning = (msg.metadata or {}).get("citation_warning")
                    if warning:
                        st.markdown(
                            f"<span class='firstRAG-warning'>⚠️ 答案未引用任何来源；"
                            f"已生成的候选 {len((msg.metadata or {}).get('candidate_citation_ids', []))} 条仅供调试。</span>",
                            unsafe_allow_html=True,
                        )

    # 优先处理 pending 请求：一次 rerun 最多消费一次
    if state.pending_chat_request() is not None:
        chat = service_factory.get_chat_service(idx_dir, sid)
        if process_pending_chat_request(chat):
            st.rerun()
            return

    # 用户输入
    user_input = st.chat_input("请输入你的问题…", key="chat_input_main")
    if user_input:
        if state.is_inflight("chat"):
            state.push_notification("warning", "回答生成中，请稍候。")
            return
        # 1. 持久化 user 消息
        state.append_chat_message(
            ChatMessage(
                role="user",
                content=user_input,
                created_at=datetime.utcnow(),
            )
        )
        # 2. 创建请求 id 并登记 pending
        request_id = uuid.uuid4().hex
        state.set_pending_chat_request(request_id, user_input)
        _LOG.info("chat request queued: id=%s, query_len=%d", request_id, len(user_input))
        # 3. 立即 rerun，由控制器接管
        st.rerun()
        return


def _append_assistant_refusal(reason: str, metadata: Optional[dict] = None) -> None:
    """向 session_state 追加一条拒答消息（统一入口）。"""
    md = {"standalone_query": "", "retrieval_count": 0, "reason": reason}
    if metadata:
        md.update(metadata)
    state.append_chat_message(
        ChatMessage(
            role="assistant",
            content=NO_EVIDENCE_REPLY,
            citations=[],
            created_at=datetime.utcnow(),
            metadata=md,
        )
    )


def process_pending_chat_request(chat: Optional[ChatService]) -> bool:
    """处理 :data:`state.pending_chat_request` 中的请求。

    一次性消费：要么完整处理，要么被拦截；**绝不**在循环中调用
    ``st.rerun``，也**不**依赖 prompt 文本判断是否完成。

    :return: 是否需要调用方 ``st.rerun()``。
    """
    state.ensure_session_state()
    pending = state.consume_pending_chat_request()
    if pending is None:
        return False

    request_id = pending.get("request_id") or ""
    user_content = pending.get("user_content") or ""
    msgs = state.chat_messages()

    # 二次校验：tail 必须与 pending 对应的 user 消息一致
    if not msgs or msgs[-1].role != "user" or msgs[-1].content != user_content:
        _LOG.warning(
            "chat request dropped: tail mismatch; id=%s", request_id
        )
        return False

    # 重复拦截
    if state.is_chat_request_completed(request_id):
        _LOG.warning("duplicate chat request blocked: id=%s", request_id)
        return False

    # inflight 互斥（防御快速重复点击）
    if state.is_inflight("chat"):
        _LOG.warning("chat request blocked by inflight: id=%s", request_id)
        _append_assistant_refusal(
            "已有任务进行中，请稍候。",
            metadata={"standalone_query": user_content, "retrieval_count": 0},
        )
        state.mark_chat_request_completed(request_id)
        return True

    # 依赖缺失 → 拒答
    if chat is None:
        _LOG.info("chat request refused: no service; id=%s", request_id)
        _append_assistant_refusal(
            "聊天服务不可用，请检查 API Key 配置。",
            metadata={"standalone_query": user_content, "retrieval_count": 0},
        )
        state.mark_chat_request_completed(request_id)
        return True

    # 标记 active
    state.set_active_chat_request_id(request_id)
    state.set_inflight("chat")

    user_msg = msgs[-1]
    history = msgs[:-1]
    placeholder = st.empty()
    accumulated: list[str] = []
    sources_count = 0
    rewrite_text: Optional[str] = None
    final_msg: Optional[ChatMessage] = None
    seen_done = False
    should_rerun = True

    try:
        try:
            for ev in chat.stream(user_msg.content, history=history):
                if ev.event_type == CHAT_EVENT_REWRITE:
                    rewrite_text = ev.content
                elif ev.event_type == CHAT_EVENT_SOURCES:
                    sources_count = ev.metadata.get("retrieval_count", 0)
                elif ev.event_type == CHAT_EVENT_TOKEN:
                    accumulated.append(ev.content or "")
                    placeholder.markdown("".join(accumulated) + "▌")
                elif ev.event_type == CHAT_EVENT_DONE:
                    final_msg = ev.message
                    seen_done = True
                    break
                elif ev.event_type == CHAT_EVENT_ERROR:
                    from rag.chat_service import ChatGenerationError

                    raise ChatGenerationError(ev.content or "chat error event")
        except Exception as exc:  # noqa: BLE001
            placeholder.empty()
            if state.is_chat_request_completed(request_id):
                _LOG.warning(
                    "duplicate chat request blocked on exception: id=%s", request_id
                )
                should_rerun = False
            else:
                reason = components.safe_error_message(exc) or type(exc).__name__
                _LOG.info("chat request refused: id=%s, reason=%s", request_id, reason)
                _append_assistant_refusal(
                    reason,
                    metadata={
                        "standalone_query": rewrite_text or user_content,
                        "retrieval_count": sources_count,
                    },
                )
                state.mark_chat_request_completed(request_id)
                state.push_notification("error", f"回答生成失败：{reason}")
    finally:
        # 任何路径都必须恢复 inflight / active
        state.set_active_chat_request_id(None)
        state.set_inflight(None)

    if not seen_done or final_msg is None:
        _LOG.info("chat request incomplete: id=%s", request_id)
        if not state.is_chat_request_completed(request_id):
            _append_assistant_refusal(
                "流式响应未完成。",
                metadata={
                    "standalone_query": rewrite_text or user_content,
                    "retrieval_count": sources_count,
                },
            )
            state.mark_chat_request_completed(request_id)
        return should_rerun

    # 成功路径：直接写入 session_state
    state.append_chat_message(final_msg)
    state.mark_chat_request_completed(request_id)
    _LOG.info(
        "chat request completed: id=%s, retrieval=%d, citations=%d",
        request_id,
        sources_count,
        len(final_msg.citations or []),
    )
    with st.chat_message("assistant"):
        st.markdown(final_msg.content or "")
        if (
            not components.is_refusal(final_msg.content or "")
            and final_msg.citations
        ):
            render_citation_panel(final_msg.citations)
    return True


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    _inject_css()
    state.ensure_session_state()
    # 显示通知（清空后）
    for note in state.drain_notifications():
        kind = note.get("kind", "info")
        msg = note.get("message", "")
        if kind == "success":
            st.toast(f"✅ {msg}")
        elif kind == "warning":
            st.toast(f"⚠️ {msg}")
        elif kind == "error":
            st.toast(f"❌ {msg}")
        else:
            st.toast(msg)
    render_sidebar()
    render_chat()


if __name__ == "__main__" or True:  # streamlit run 不会传 __name__ == "__main__"
    main()
