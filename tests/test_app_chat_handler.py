"""``app.process_pending_chat_request`` 重复调用与生命周期回归测试。

**所有测试**仅使用 :class:`tests._fakes.FakeChatService`，不调用任何远程 API。
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.logging import setup_logging  # noqa: E402
from rag.chat_service import (  # noqa: E402
    CHAT_EVENT_DONE,
    CHAT_EVENT_ERROR,
    CHAT_EVENT_REWRITE,
    CHAT_EVENT_SOURCES,
    CHAT_EVENT_TOKEN,
    ChatGenerationError,
)
from rag.models import ChatMessage, ChatStreamEvent  # noqa: E402
from rag.prompt_builder import NO_EVIDENCE_REPLY  # noqa: E402

import app  # noqa: E402
from tests._fakes import FakeChatService  # noqa: E402
from ui import state  # noqa: E402

setup_logging(level="WARNING")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _install_state(monkeypatch) -> None:
    session_state: dict = {}
    monkeypatch.setattr(state.st, "session_state", session_state)
    state.ensure_session_state()
    return None


def _final_message(text: str = "回答 [S1]。", citations=None) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=text,
        citations=list(citations or []),
        created_at=datetime.now(timezone.utc),
    )


def _seed_user_message(text: str) -> None:
    state.append_chat_message(
        ChatMessage(
            role="user",
            content=text,
            created_at=datetime.now(timezone.utc),
        )
    )


def _queue_request(monkeypatch, query: str = "高血压？") -> str:
    request_id = uuid.uuid4().hex
    state.set_pending_chat_request(request_id, query)
    _seed_user_message(query)
    return request_id


def _stub_st_rerun(monkeypatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr(app.st, "rerun", mock)
    return mock


class _PlaceholderRecorder:
    def __init__(self) -> None:
        self.markdowns: list[str] = []
        self.emptied = 0

    def markdown(self, text: str) -> None:
        self.markdowns.append(text)

    def empty(self) -> None:
        self.emptied += 1


def _stub_st_empty(monkeypatch) -> _PlaceholderRecorder:
    recorder = _PlaceholderRecorder()
    placeholder = SimpleNamespace(markdown=recorder.markdown, empty=recorder.empty)
    monkeypatch.setattr(app.st, "empty", lambda: placeholder)
    return recorder


# ---------------------------------------------------------------------------
# 1. happy path
# ---------------------------------------------------------------------------
def test_happy_path_persists_final_message(monkeypatch):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch, "高血压？")
    final = _final_message("答 [S1].")
    fake = FakeChatService(
        events=[
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES,
                citations=[],
                metadata={"retrieval_count": 2},
            ),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content="答 "),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content="[S1]."),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    rerun = _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    should_rerun = app.process_pending_chat_request(fake)

    assert fake.call_count == 1
    assert should_rerun is True
    msgs = state.chat_messages()
    assert len(msgs) == 2
    assert msgs[-1] is final
    assert state.is_chat_request_completed(rid)
    assert state.active_chat_request_id() is None
    assert state.is_inflight("chat") is False
    assert state.pending_chat_request() is None
    assert rerun.call_count == 0  # controller 不调用 rerun


# ---------------------------------------------------------------------------
# 2. duplicate blocked
# ---------------------------------------------------------------------------
def test_duplicate_request_id_blocked(monkeypatch, caplog):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    state.mark_chat_request_completed(rid)
    fake = FakeChatService()
    rerun = _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    with caplog.at_level("WARNING"):
        should_rerun = app.process_pending_chat_request(fake)

    assert fake.call_count == 0
    assert should_rerun is False
    assert any(
        "duplicate chat request blocked" in rec.getMessage()
        for rec in caplog.records
    )
    assert state.pending_chat_request() is None
    assert rerun.call_count == 0


# ---------------------------------------------------------------------------
# 3. no pending
# ---------------------------------------------------------------------------
def test_no_pending_returns_false(monkeypatch):
    _install_state(monkeypatch)
    fake = FakeChatService()
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    should_rerun = app.process_pending_chat_request(fake)

    assert should_rerun is False
    assert fake.call_count == 0


# ---------------------------------------------------------------------------
# 4. rewrite event recorded
# ---------------------------------------------------------------------------
def test_rewrite_event_recorded(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch, "它怎么办？")
    final = _final_message("答 [S1].")
    fake = FakeChatService(
        events=[
            ChatStreamEvent(event_type=CHAT_EVENT_REWRITE, content="高血压怎么办？"),
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES,
                citations=[],
                metadata={"retrieval_count": 1},
            ),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    # 控制器不重写 final_msg 的 metadata（final_msg 由服务提供），
    # 但要求 rewrite 事件被消费过（否则 done 不会被处理）。
    assert fake.call_count == 1
    msgs = state.chat_messages()
    assert msgs[-1] is final


# ---------------------------------------------------------------------------
# 5. sources retriev_count
# ---------------------------------------------------------------------------
def test_sources_event_stored_in_metadata(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    final = _final_message()
    fake = FakeChatService(
        events=[
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES,
                citations=[],
                metadata={"retrieval_count": 3},
            ),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    # sources 事件被消费过，且 final_msg 由服务提供；metadata 不被改写
    assert fake.call_count == 1
    assert state.chat_messages()[-1] is final


# ---------------------------------------------------------------------------
# 6. tokens update placeholder
# ---------------------------------------------------------------------------
def test_tokens_increment_placeholder(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    final = _final_message()
    fake = FakeChatService(
        events=[
            ChatStreamEvent(event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 0}),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content="答"),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content="案"),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content=" [S1]"),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    _stub_st_rerun(monkeypatch)
    placeholder = _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    # placeholder.markdown 至少出现 3 次；空 placeholder 调用不在硬性要求中
    assert len(placeholder.markdowns) == 3
    assert placeholder.markdowns[0].startswith("答")
    assert placeholder.markdowns[1].startswith("答案")
    assert placeholder.markdowns[2].endswith(" [S1]▌")


# ---------------------------------------------------------------------------
# 7. done event persists (duplicate of #1 with extra focus)
# ---------------------------------------------------------------------------
def test_done_event_persists_final_message(monkeypatch):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    final = _final_message("done_only")
    fake = FakeChatService(
        events=[ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final)]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert state.chat_messages()[-1].content == "done_only"
    assert state.is_chat_request_completed(rid)


# ---------------------------------------------------------------------------
# 8. missing done → refusal
# ---------------------------------------------------------------------------
def test_missing_done_appends_refusal(monkeypatch):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    fake = FakeChatService(
        events=[
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 1}
            ),
            # 没有 done
        ]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    msgs = state.chat_messages()
    assert len(msgs) == 2
    assert msgs[-1].content == NO_EVIDENCE_REPLY
    assert msgs[-1].metadata.get("reason") == "流式响应未完成。"
    assert state.is_chat_request_completed(rid)


# ---------------------------------------------------------------------------
# 9. exception → refusal
# ---------------------------------------------------------------------------
def test_exception_appends_refusal(monkeypatch, caplog):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    fake = FakeChatService(raise_on_stream=ChatGenerationError("LLMTimeoutError"))
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    with caplog.at_level("INFO"):
        app.process_pending_chat_request(fake)

    msgs = state.chat_messages()
    assert len(msgs) == 2
    assert msgs[-1].content == NO_EVIDENCE_REPLY
    # 控制器不依赖 safe_error_message 注入 LLM 字样；只确保不重复
    # 出现异常类名（防御 XSS / 状态泄漏）。
    reason = msgs[-1].metadata.get("reason") or ""
    assert "LLMTimeoutError" not in reason
    assert state.is_chat_request_completed(rid)
    assert state.is_inflight("chat") is False
    assert state.active_chat_request_id() is None


# ---------------------------------------------------------------------------
# 10. service-side refusal persisted as-is
# ---------------------------------------------------------------------------
def test_service_refusal_persisted_as_is(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    final = _final_message(NO_EVIDENCE_REPLY)
    fake = FakeChatService(
        events=[
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 0}
            ),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert state.chat_messages()[-1] is final


# ---------------------------------------------------------------------------
# 11. citations → render panel called
# ---------------------------------------------------------------------------
def test_citation_panel_rendered(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    from tests._fakes import make_chunk

    citation = make_chunk(
        chunk_id="c1",
        document_id="d1",
        content="高血压是慢性病。",
        source_name="doc.pdf",
    )
    final = _final_message("答 [S1].".format(), citations=[citation])
    fake = FakeChatService(
        events=[
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES,
                citations=[citation],
                metadata={"retrieval_count": 1},
            ),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    render_calls: list = []
    monkeypatch.setattr(
        app, "render_citation_panel", lambda citations: render_calls.append(list(citations))
    )

    app.process_pending_chat_request(fake)

    assert render_calls and render_calls[0] == [citation]


# ---------------------------------------------------------------------------
# 12–14. inflight reset on done / exception / missing done
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["done", "exception", "missing_done"])
def test_inflight_reset_on_terminal_state(monkeypatch, scenario):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    if scenario == "done":
        events = [ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=_final_message())]
        fake = FakeChatService(events=events)
    elif scenario == "exception":
        fake = FakeChatService(raise_on_stream=LLMErrorAny())
    else:
        events = [
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 0}
            )
        ]
        fake = FakeChatService(events=events)
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert state.is_inflight("chat") is False
    assert state.active_chat_request_id() is None


class LLMErrorAny(Exception):
    pass


# ---------------------------------------------------------------------------
# 15. completed set on all terminal states
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", ["done", "exception", "missing_done"])
def test_completed_set_on_all_terminal_states(monkeypatch, scenario):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    if scenario == "done":
        events = [ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=_final_message())]
        fake = FakeChatService(events=events)
    elif scenario == "exception":
        fake = FakeChatService(raise_on_stream=LLMErrorAny())
    else:
        events = [
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 0}
            )
        ]
        fake = FakeChatService(events=events)
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert state.is_chat_request_completed(rid)


# ---------------------------------------------------------------------------
# 16. active id set before stream
# ---------------------------------------------------------------------------
def test_active_id_set_before_stream(monkeypatch):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    observed: dict = {}

    class SideEffectFake(FakeChatService):
        def stream(self, query, history=None):  # type: ignore[override]
            self.call_count += 1
            self.last_query = query
            self.last_history = list(history) if history else []
            observed["active"] = state.active_chat_request_id()
            observed["inflight"] = state.is_inflight("chat")
            for ev in self._events:
                yield ev

    fake = SideEffectFake(
        events=[ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=_final_message())]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert observed["active"] == rid
    assert observed["inflight"] is True


# ---------------------------------------------------------------------------
# 17–18. active id cleared on done / exception
# ---------------------------------------------------------------------------
def test_active_id_cleared_on_done(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    fake = FakeChatService(
        events=[ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=_final_message())]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert state.active_chat_request_id() is None


def test_active_id_cleared_on_exception(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    fake = FakeChatService(raise_on_stream=LLMErrorAny())
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    assert state.active_chat_request_id() is None


# ---------------------------------------------------------------------------
# 19. no rerun during event loop
# ---------------------------------------------------------------------------
def test_no_rerun_during_event_loop(monkeypatch):
    _install_state(monkeypatch)
    _queue_request(monkeypatch)
    final = _final_message()
    fake = FakeChatService(
        events=[
            ChatStreamEvent(event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 1}),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content="a"),
            ChatStreamEvent(event_type=CHAT_EVENT_TOKEN, content="b"),
            ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=final),
        ]
    )
    rerun = _stub_st_rerun(monkeypatch)
    placeholder = _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    # 控制器自身不调用 rerun；调用方拿到 True 后会 rerun。
    assert rerun.call_count == 0
    assert len(placeholder.markdowns) == 2  # 两次 token


# ---------------------------------------------------------------------------
# 20. duplicate request_id second entry blocked
# ---------------------------------------------------------------------------
def test_duplicate_request_id_second_entry(monkeypatch):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    fake = FakeChatService(
        events=[ChatStreamEvent(event_type=CHAT_EVENT_DONE, message=_final_message())]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    first = app.process_pending_chat_request(fake)
    assert first is True
    assert fake.call_count == 1

    # 第二次：恢复 pending 与 user 消息，但 id 已完成
    state.set_pending_chat_request(rid, "高血压？")
    second = app.process_pending_chat_request(fake)
    assert second is False
    assert fake.call_count == 1
    assert state.is_chat_request_completed(rid)


# ---------------------------------------------------------------------------
# 额外：CHAT_EVENT_ERROR 也能被转换成拒答
# ---------------------------------------------------------------------------
def test_error_event_translated_to_refusal(monkeypatch):
    _install_state(monkeypatch)
    rid = _queue_request(monkeypatch)
    fake = FakeChatService(
        events=[
            ChatStreamEvent(
                event_type=CHAT_EVENT_SOURCES, metadata={"retrieval_count": 0}
            ),
            ChatStreamEvent(event_type=CHAT_EVENT_ERROR, content="upstream broken"),
        ]
    )
    _stub_st_rerun(monkeypatch)
    _stub_st_empty(monkeypatch)

    app.process_pending_chat_request(fake)

    msgs = state.chat_messages()
    assert msgs[-1].content == NO_EVIDENCE_REPLY
    assert state.is_chat_request_completed(rid)
    assert state.is_inflight("chat") is False
