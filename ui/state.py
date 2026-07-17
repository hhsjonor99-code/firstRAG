"""Streamlit ``session_state`` 初始化与辅助函数。

设计原则：

1. **不在 ``session_state`` 写入 API Key**：仅写入 ``has_sf_key`` / ``has_mm_key`` 布尔值。
2. **不缓存大型对象**（如 vector store、chunks 列表）：服务实例由 :mod:`ui.service_factory`
   通过 ``st.cache_resource`` 管理。
3. **可序列化的用户数据**（如聊天历史、上传结果）放 ``session_state``。
4. **幂等初始化**——``ensure_session_state()`` 可被任意位置调用多次。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from rag.models import ChatMessage


# ---------------------------------------------------------------------------
# 状态键常量
# ---------------------------------------------------------------------------
KEY_CHAT_MESSAGES = "chat_messages"
"""``list[ChatMessage]``：用户 + 助手的历史消息。"""

KEY_PENDING_DELETE_DOC_ID = "pending_delete_document_id"
"""``str | None``：当前待二次确认的 document_id。"""

KEY_CONFIRM_CLEAR_KB = "confirm_clear_knowledge_base"
"""``bool``：是否在二次确认阶段。"""

KEY_CONFIRM_CLEAR_CHAT = "confirm_clear_chat"
"""``bool``：是否在二次确认阶段。"""

KEY_UI_NOTIFICATIONS = "ui_notifications"
"""``list[dict]``：轻量提示队列（type / message / key）。"""

KEY_INFLIGHT = "inflight"
"""``dict``：当前正在执行的动作（``upload`` / ``chat`` / ``delete``）"""

KEY_LAST_INGEST_HASHES = "last_ingest_hashes"
"""``set[str]``：本会话已入库成功的 SHA256，避免同一次 rerun 重复入库。"""

KEY_PENDING_CHAT_REQUEST = "pending_chat_request"
"""``dict | None``：尚未开始消费的聊天请求。"""

KEY_ACTIVE_CHAT_REQUEST_ID = "active_chat_request_id"
"""``str | None``：当前正在 ``chat.stream`` 的请求 id。"""

KEY_COMPLETED_CHAT_REQUEST_IDS = "completed_chat_request_ids"
"""``set[str]``：本次会话内已完成（含成功、拒答、异常、缺 done）的请求 id 集合。"""


def ensure_session_state() -> None:
    """初始化所有 session_state 键。多次调用安全。"""
    if KEY_CHAT_MESSAGES not in st.session_state:
        st.session_state[KEY_CHAT_MESSAGES] = []
    if KEY_PENDING_DELETE_DOC_ID not in st.session_state:
        st.session_state[KEY_PENDING_DELETE_DOC_ID] = None
    if KEY_CONFIRM_CLEAR_KB not in st.session_state:
        st.session_state[KEY_CONFIRM_CLEAR_KB] = False
    if KEY_CONFIRM_CLEAR_CHAT not in st.session_state:
        st.session_state[KEY_CONFIRM_CLEAR_CHAT] = False
    if KEY_UI_NOTIFICATIONS not in st.session_state:
        st.session_state[KEY_UI_NOTIFICATIONS] = []
    if KEY_INFLIGHT not in st.session_state:
        st.session_state[KEY_INFLIGHT] = {"kind": None, "started_at": None}
    if KEY_LAST_INGEST_HASHES not in st.session_state:
        st.session_state[KEY_LAST_INGEST_HASHES] = set()
    if KEY_PENDING_CHAT_REQUEST not in st.session_state:
        st.session_state[KEY_PENDING_CHAT_REQUEST] = None
    if KEY_ACTIVE_CHAT_REQUEST_ID not in st.session_state:
        st.session_state[KEY_ACTIVE_CHAT_REQUEST_ID] = None
    if KEY_COMPLETED_CHAT_REQUEST_IDS not in st.session_state:
        st.session_state[KEY_COMPLETED_CHAT_REQUEST_IDS] = set()


# ---------------------------------------------------------------------------
# 消息序列化
# ---------------------------------------------------------------------------
def chat_messages() -> list[ChatMessage]:
    """获取当前聊天消息列表（拷贝），保证空列表。"""
    ensure_session_state()
    return list(st.session_state[KEY_CHAT_MESSAGES])  # type: ignore[arg-type]


def append_chat_message(msg: ChatMessage) -> None:
    """追加一条消息到聊天历史。"""
    ensure_session_state()
    st.session_state[KEY_CHAT_MESSAGES].append(msg)


def clear_chat_messages() -> None:
    """清空聊天历史。"""
    ensure_session_state()
    st.session_state[KEY_CHAT_MESSAGES] = []
    clear_all_chat_request_lifecycle()


# ---------------------------------------------------------------------------
# 通知
# ---------------------------------------------------------------------------
def push_notification(kind: str, message: str) -> None:
    """向通知队列推一条提示（同一 rerun 期间最多显示一次）。"""
    ensure_session_state()
    key = f"{kind}:{message}"
    existing = st.session_state[KEY_UI_NOTIFICATIONS]
    if any(n.get("key") == key for n in existing):
        return
    existing.append({"kind": kind, "message": message, "key": key})


def drain_notifications() -> list[dict[str, Any]]:
    """取出并清空通知队列。"""
    ensure_session_state()
    out = list(st.session_state[KEY_UI_NOTIFICATIONS])
    st.session_state[KEY_UI_NOTIFICATIONS] = []
    return out


# ---------------------------------------------------------------------------
# Inflight 互斥
# ---------------------------------------------------------------------------
def set_inflight(kind: str | None) -> None:
    """设置当前 inflight 动作。``kind=None`` 表示空闲。"""
    import time
    ensure_session_state()
    if kind is None:
        st.session_state[KEY_INFLIGHT] = {"kind": None, "started_at": None}
    else:
        st.session_state[KEY_INFLIGHT] = {
            "kind": kind,
            "started_at": time.time(),
        }


def inflight_kind() -> str | None:
    ensure_session_state()
    return st.session_state[KEY_INFLIGHT].get("kind")


def is_inflight(kind: str | None = None) -> bool:
    current = inflight_kind()
    if current is None:
        return False
    if kind is None:
        return True
    return current == kind


# ---------------------------------------------------------------------------
# 已入库 SHA256
# ---------------------------------------------------------------------------
def remember_ingest_hash(file_hash: str) -> bool:
    """记录已入库 SHA256；同一 hash 在本会话内已记录则返回 False。"""
    ensure_session_state()
    seen: set[str] = st.session_state[KEY_LAST_INGEST_HASHES]
    if file_hash in seen:
        return False
    seen.add(file_hash)
    return True


# ---------------------------------------------------------------------------
# 二次确认辅助
# ---------------------------------------------------------------------------
def request_delete(document_id: str) -> None:
    st.session_state[KEY_PENDING_DELETE_DOC_ID] = document_id


def cancel_delete() -> None:
    st.session_state[KEY_PENDING_DELETE_DOC_ID] = None


def pending_delete_id() -> str | None:
    return st.session_state.get(KEY_PENDING_DELETE_DOC_ID)


def confirm_clear_knowledge_base() -> bool:
    ensure_session_state()
    return bool(st.session_state[KEY_CONFIRM_CLEAR_KB])


def confirm_clear_chat() -> bool:
    ensure_session_state()
    return bool(st.session_state[KEY_CONFIRM_CLEAR_CHAT])


def set_confirm_clear_kb(flag: bool) -> None:
    st.session_state[KEY_CONFIRM_CLEAR_KB] = flag


def set_confirm_clear_chat(flag: bool) -> None:
    st.session_state[KEY_CONFIRM_CLEAR_CHAT] = flag


# ---------------------------------------------------------------------------
# 聊天请求生命周期
# ---------------------------------------------------------------------------
def set_pending_chat_request(request_id: str, user_content: str) -> None:
    """登记一个尚未被消费的聊天请求。"""
    ensure_session_state()
    import time

    st.session_state[KEY_PENDING_CHAT_REQUEST] = {
        "request_id": request_id,
        "user_content": user_content,
        "submitted_at": time.time(),
    }


def consume_pending_chat_request() -> dict | None:
    """原子地读取并清空当前 pending 请求。"""
    ensure_session_state()
    pending = st.session_state.get(KEY_PENDING_CHAT_REQUEST)
    st.session_state[KEY_PENDING_CHAT_REQUEST] = None
    return pending


def pending_chat_request() -> dict | None:
    """非破坏性读取当前 pending 请求。"""
    ensure_session_state()
    return st.session_state.get(KEY_PENDING_CHAT_REQUEST)


def set_active_chat_request_id(request_id: str | None) -> None:
    ensure_session_state()
    st.session_state[KEY_ACTIVE_CHAT_REQUEST_ID] = request_id


def active_chat_request_id() -> str | None:
    ensure_session_state()
    return st.session_state.get(KEY_ACTIVE_CHAT_REQUEST_ID)


def mark_chat_request_completed(request_id: str) -> None:
    ensure_session_state()
    completed = st.session_state[KEY_COMPLETED_CHAT_REQUEST_IDS]
    if not isinstance(completed, set):
        completed = set()
        st.session_state[KEY_COMPLETED_CHAT_REQUEST_IDS] = completed
    completed.add(request_id)


def is_chat_request_completed(request_id: str) -> bool:
    ensure_session_state()
    completed = st.session_state.get(KEY_COMPLETED_CHAT_REQUEST_IDS)
    if not isinstance(completed, set):
        return False
    return request_id in completed


def clear_completed_chat_request_ids() -> None:
    ensure_session_state()
    st.session_state[KEY_COMPLETED_CHAT_REQUEST_IDS] = set()


def clear_all_chat_request_lifecycle() -> None:
    """一次性重置 pending / active / completed；用于「清空对话」按钮。"""
    ensure_session_state()
    st.session_state[KEY_PENDING_CHAT_REQUEST] = None
    st.session_state[KEY_ACTIVE_CHAT_REQUEST_ID] = None
    st.session_state[KEY_COMPLETED_CHAT_REQUEST_IDS] = set()
