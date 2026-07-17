"""UI 纯函数辅助测试。

不涉及 Streamlit 渲染；只测试 :mod:`ui.components` 与 :mod:`ui.state` 中的纯函数。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.models import DocumentChunk, RetrievedChunk  # noqa: E402
from rag.prompt_builder import NO_EVIDENCE_REPLY  # noqa: E402

from ui import components, service_factory, state  # noqa: E402


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _make_chunk(**overrides) -> DocumentChunk:
    defaults = dict(
        chunk_id="c1",
        document_id="d1",
        content="content",
        source_name="doc.txt",
        chunk_index=0,
    )
    defaults.update(overrides)
    return DocumentChunk(**defaults)


def _make_retrieved(citation_id: str = "S1", chunk: DocumentChunk = None) -> RetrievedChunk:
    if chunk is None:
        chunk = _make_chunk()
    return RetrievedChunk(chunk=chunk, score=0.85, citation_id=citation_id)


def _make_signature_settings(**overrides):
    """构造不读取 .env 的签名测试配置。"""
    defaults = {
        "siliconflow_api_key": None,
        "siliconflow_base_url": "https://sf.example.test/v1",
        "siliconflow_embedding_model": "test-embedding-model",
        "siliconflow_embedding_dimensions": 1024,
        "siliconflow_embedding_batch_size": 16,
        "siliconflow_timeout": 60,
        "minimax_api_key": None,
        "minimax_base_url": "https://minimax.example.test/v1",
        "minimax_model": "test-chat-model",
        "minimax_timeout": 60.0,
        "minimax_max_retries": 2,
        "minimax_temperature": 0.2,
        "minimax_max_tokens": None,
        "chunk_size": 800,
        "chunk_overlap": 120,
        "retrieval_top_k": 5,
        "retrieval_min_score": None,
        "max_upload_mb": 20,
        "max_history_turns": 10,
    }
    defaults.update(overrides)
    settings = SimpleNamespace(**defaults)
    settings.has_siliconflow_key = lambda: bool(
        (settings.siliconflow_api_key or "").strip()
    )
    settings.has_minimax_key = lambda: bool(
        (settings.minimax_api_key or "").strip()
    )
    return settings


def _patch_signature_context(
    monkeypatch,
    settings=None,
    index_dir=Path("test-storage/indexes"),
    upload_dir=Path("test-storage/uploads"),
):
    settings = settings or _make_signature_settings()
    monkeypatch.setattr(service_factory, "load_settings", lambda: settings)
    monkeypatch.setattr(service_factory, "default_index_dir", lambda: index_dir)
    monkeypatch.setattr(service_factory, "default_upload_dir", lambda: upload_dir)
    return settings


# ---------------------------------------------------------------------------
# 1. 引用位置格式化
# ---------------------------------------------------------------------------
def test_format_location_pdf():
    ch = _make_chunk(page_number=5)
    assert components.format_chunk_location(ch) == "第 5 页"


def test_format_location_docx_paragraph():
    ch = _make_chunk(paragraph_start=10, paragraph_end=16, block_type="paragraph")
    assert components.format_chunk_location(ch) == "第 10～16 段"


def test_format_location_docx_paragraph_single():
    ch = _make_chunk(paragraph_start=10, paragraph_end=10, block_type="paragraph")
    assert components.format_chunk_location(ch) == "第 10 段"


def test_format_location_docx_table():
    ch = _make_chunk(
        block_type="table",
        table_index=1,
        row_start=4,
        row_end=7,
    )
    assert components.format_chunk_location(ch) == "第 2 个表格，第 5～8 行"


def test_format_location_docx_textbox():
    ch = _make_chunk(block_type="textbox", block_indices=[2])
    assert components.format_chunk_location(ch) == "文本块 3"


def test_format_location_txt_lines():
    ch = _make_chunk(line_start=10, line_end=20)
    assert components.format_chunk_location(ch) == "第 10～20 行"


def test_format_location_md_lines():
    ch = _make_chunk(line_start=5, line_end=5)
    assert components.format_chunk_location(ch) == "第 5 行"


def test_format_location_returns_empty_for_no_loc():
    ch = _make_chunk()
    assert components.format_chunk_location(ch) == ""


# ---------------------------------------------------------------------------
# 2. 文件大小 / 时间格式化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (2048, "2.00 KB"),
    (5 * 1024 * 1024, "5.00 MB"),
    (2 * 1024 * 1024 * 1024, "2.00 GB"),
])
def test_format_file_size(size, expected):
    assert components.format_file_size(size) == expected


def test_format_file_size_negative():
    assert components.format_file_size(-1) == "—"


def test_format_created_at():
    dt = datetime(2024, 1, 5, 9, 30, tzinfo=timezone.utc)
    assert components.format_created_at(dt) == "2024-01-05 09:30"


def test_format_created_at_none():
    assert components.format_created_at(None) == "—"


# ---------------------------------------------------------------------------
# 3. 安全错误映射
# ---------------------------------------------------------------------------
def test_safe_error_message_known_types():
    from rag.knowledge_base_service import (
        DuplicateDocumentError as KSDuplicate,
        EmptyUploadError,
        InvalidUploadError,
        KnowledgeBaseServiceError,
        UnsupportedUploadTypeError,
        UploadTooLargeError,
    )
    from rag.llm_client import (
        LLMAuthenticationError,
        LLMError,
        LLMTimeoutError,
    )
    from rag.parsers import (
        EmptyDocumentError,
        UnsupportedScannedPDFError,
    )

    cases = [
        (EmptyUploadError("x"), "空"),
        (UploadTooLargeError("x"), "大小"),
        (UnsupportedUploadTypeError("x"), "格式"),
        (InvalidUploadError("x"), "路径"),
        (KSDuplicate("x"), "已存在"),
        (KnowledgeBaseServiceError("x"), "知识库"),
        (EmptyDocumentError("x"), "解析"),
        (UnsupportedScannedPDFError("x"), "扫描"),
        (LLMAuthenticationError("x"), "鉴权"),
        (LLMTimeoutError("x"), "超时"),
        (LLMError("x"), "生成"),
    ]
    for exc, must_contain in cases:
        msg = components.safe_error_message(exc)
        assert must_contain in msg, f"got: {msg!r}"


def test_safe_error_message_unknown():
    class Custom(Exception):
        pass
    msg = components.safe_error_message(Custom("xxx"))
    # 兜底：只输出类型名，不含原始消息
    assert "Custom" in msg
    assert "xxx" not in msg  # 异常原文不应泄漏


# ---------------------------------------------------------------------------
# 4. 上传结果汇总
# ---------------------------------------------------------------------------
def test_summarize_ingest_results_all_success():
    results = [
        {"file_name": "a.txt", "ok": True, "info": object(), "error": None},
        {"file_name": "b.txt", "ok": True, "info": object(), "error": None},
    ]
    s = components.summarize_ingest_results(results)
    assert s == {"total": 2, "success": 2, "failed": 0}


def test_summarize_ingest_results_mixed():
    results = [
        {"file_name": "a.txt", "ok": True, "info": object(), "error": None},
        {"file_name": "b.txt", "ok": False, "info": None, "error": "失败"},
        {"file_name": "c.txt", "ok": True, "info": object(), "error": None},
        {"file_name": "d.txt", "ok": False, "info": None, "error": "失败"},
    ]
    s = components.summarize_ingest_results(results)
    assert s == {"total": 4, "success": 2, "failed": 2}


def test_summarize_ingest_results_empty():
    assert components.summarize_ingest_results([]) == {"total": 0, "success": 0, "failed": 0}


# ---------------------------------------------------------------------------
# 5. 状态指示
# ---------------------------------------------------------------------------
def test_api_key_status_true():
    assert components.api_key_status(True) == "已配置"


def test_api_key_status_false():
    assert components.api_key_status(False) == "未配置"


def test_clear_confirmation_state_accessors(monkeypatch):
    session_state = {}
    monkeypatch.setattr(state.st, "session_state", session_state)

    assert state.confirm_clear_knowledge_base() is False
    assert state.confirm_clear_chat() is False

    state.set_confirm_clear_kb(True)
    state.set_confirm_clear_chat(True)
    assert state.confirm_clear_knowledge_base() is True
    assert state.confirm_clear_chat() is True


def test_ensure_session_state_is_idempotent(monkeypatch):
    session_state = {}
    monkeypatch.setattr(state.st, "session_state", session_state)

    state.ensure_session_state()
    state.set_pending_chat_request("abc", "用户问题")
    state.mark_chat_request_completed("abc")
    state.set_active_chat_request_id("abc")

    state.ensure_session_state()
    pending = state.pending_chat_request()
    assert pending is not None
    assert pending["request_id"] == "abc"
    assert pending["user_content"] == "用户问题"
    assert state.active_chat_request_id() == "abc"
    assert state.is_chat_request_completed("abc")


def test_consume_pending_chat_request_returns_none(monkeypatch):
    session_state = {}
    monkeypatch.setattr(state.st, "session_state", session_state)
    state.ensure_session_state()
    state.set_pending_chat_request("abc", "x")

    first = state.consume_pending_chat_request()
    second = state.consume_pending_chat_request()

    assert first is not None
    assert first["request_id"] == "abc"
    assert second is None


def test_mark_chat_request_completed_is_idempotent(monkeypatch):
    session_state = {}
    monkeypatch.setattr(state.st, "session_state", session_state)
    state.ensure_session_state()

    state.mark_chat_request_completed("abc")
    state.mark_chat_request_completed("abc")
    state.mark_chat_request_completed("def")

    assert state.is_chat_request_completed("abc")
    assert state.is_chat_request_completed("def")
    assert not state.is_chat_request_completed("zzz")


def test_clear_all_chat_request_lifecycle_resets_keys(monkeypatch):
    session_state = {}
    monkeypatch.setattr(state.st, "session_state", session_state)
    state.ensure_session_state()
    state.set_pending_chat_request("abc", "x")
    state.mark_chat_request_completed("abc")
    state.set_active_chat_request_id("abc")

    state.clear_all_chat_request_lifecycle()

    assert state.pending_chat_request() is None
    assert state.active_chat_request_id() is None
    assert state.is_chat_request_completed("abc") is False


def test_clear_chat_messages_also_resets_lifecycle(monkeypatch):
    session_state = {}
    monkeypatch.setattr(state.st, "session_state", session_state)
    state.ensure_session_state()
    state.set_pending_chat_request("abc", "x")
    state.mark_chat_request_completed("abc")
    state.set_active_chat_request_id("abc")

    state.clear_chat_messages()

    assert state.pending_chat_request() is None
    assert state.active_chat_request_id() is None
    assert state.is_chat_request_completed("abc") is False


# ---------------------------------------------------------------------------
# 6. 内部泄漏检测
# ---------------------------------------------------------------------------
def test_detect_internal_leakage_clean():
    assert components.detect_internal_leakage("正常回答 [S1].") == []


def test_detect_internal_leakage_think():
    assert "<think>" in components.detect_internal_leakage("前缀<think>reasoning</think>答案")


def test_detect_internal_leakage_analysis():
    assert "<analysis>" in components.detect_internal_leakage("x<analysis>r</analysis>y")


def test_detect_internal_leakage_non_string():
    assert components.detect_internal_leakage(None) == []


# ---------------------------------------------------------------------------
# 7. 引用编号提取
# ---------------------------------------------------------------------------
def test_extract_citation_ids_basic():
    assert components.extract_citation_ids("[S1] [S2] [S3]") == ["S1", "S2", "S3"]


def test_extract_citation_ids_dedupe_keep_order():
    assert components.extract_citation_ids("[S2] [S1] [S2] [S3] [S1]") == ["S2", "S1", "S3"]


def test_extract_citation_ids_s10_not_s1():
    # 关键：[S10] 不应被误匹配为 [S1]
    assert components.extract_citation_ids("[S10] [S1]") == ["S10", "S1"]


def test_extract_citation_ids_no_match():
    assert components.extract_citation_ids("无引用") == []


# ---------------------------------------------------------------------------
# 8. 引用面板构建
# ---------------------------------------------------------------------------
def test_build_citation_view():
    ch = _make_chunk(
        chunk_id="c1",
        document_id="d1",
        content="高血压是慢性病。",
        source_name="doc.pdf",
        page_number=5,
    )
    rc = _make_retrieved(citation_id="S1", chunk=ch)
    view = components.build_citation_view(rc)
    assert view["citation_id"] == "S1"
    assert view["source_name"] == "doc.pdf"
    assert view["location"] == "第 5 页"
    assert view["content"] == "高血压是慢性病。"
    assert view["score"] == "0.8500"


def test_build_citation_panel_empty():
    assert components.build_citation_panel([]) == []


def test_build_citation_panel_filters_s10_correctly():
    # 多个引用，应按 S# 顺序
    rc1 = _make_retrieved(citation_id="S1", chunk=_make_chunk(content="C1"))
    rc2 = _make_retrieved(citation_id="S10", chunk=_make_chunk(chunk_id="c10", content="C10"))
    panel = components.build_citation_panel([rc1, rc2])
    assert [v["citation_id"] for v in panel] == ["S1", "S10"]


# ---------------------------------------------------------------------------
# 9. 拒答 / 角色辅助
# ---------------------------------------------------------------------------
def test_is_refusal():
    assert components.is_refusal(NO_EVIDENCE_REPLY) is True
    assert components.is_refusal(f"  {NO_EVIDENCE_REPLY}  ") is True
    assert components.is_refusal("正常回答") is False
    assert components.is_refusal(None) is False
    assert components.is_refusal("") is False


def test_chat_role_label():
    assert components.chat_role_label("user") == "用户"
    assert components.chat_role_label("assistant") == "助手"
    assert components.chat_role_label("system") == "系统"
    assert components.chat_role_label("unknown") == "unknown"


# ---------------------------------------------------------------------------
# 10. 评分格式
# ---------------------------------------------------------------------------
def test_format_score():
    assert components.format_score(0.85) == "0.8500"
    assert components.format_score(0) == "0.0000"
    assert components.format_score(1) == "1.0000"


def test_format_score_invalid():
    assert components.format_score(None) == "—"
    assert components.format_score("not_a_number") == "—"


# ---------------------------------------------------------------------------
# 11. Settings cache 签名
# ---------------------------------------------------------------------------
def test_settings_signature_serializes_int(monkeypatch):
    settings = _make_signature_settings(siliconflow_embedding_batch_size=32)
    _patch_signature_context(monkeypatch, settings)
    assert isinstance(service_factory.settings_signature(), int)


def test_settings_signature_serializes_float(monkeypatch):
    settings = _make_signature_settings(minimax_temperature=0.35)
    _patch_signature_context(monkeypatch, settings)
    assert isinstance(service_factory.settings_signature(), int)


def test_settings_signature_serializes_bool(monkeypatch):
    settings = _make_signature_settings(
        siliconflow_api_key="test-secret-never-log",
        minimax_api_key=None,
    )
    _patch_signature_context(monkeypatch, settings)
    assert isinstance(service_factory.settings_signature(), int)


def test_settings_signature_serializes_none(monkeypatch):
    settings = _make_signature_settings(
        minimax_max_tokens=None,
        retrieval_min_score=None,
    )
    _patch_signature_context(monkeypatch, settings)
    assert isinstance(service_factory.settings_signature(), int)


def test_settings_signature_serializes_path(monkeypatch, tmp_path):
    _patch_signature_context(
        monkeypatch,
        index_dir=tmp_path / "indexes",
        upload_dir=tmp_path / "uploads",
    )
    assert isinstance(service_factory.settings_signature(), int)


def test_settings_signature_is_stable(monkeypatch):
    _patch_signature_context(monkeypatch)
    first = service_factory.settings_signature()
    second = service_factory.settings_signature()
    assert first == second


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("siliconflow_embedding_dimensions", 2048),
        ("chunk_size", 1000),
        ("retrieval_top_k", 8),
        ("siliconflow_embedding_model", "different-embedding-model"),
        ("minimax_model", "different-chat-model"),
    ],
)
def test_settings_signature_changes_with_cache_config(
    monkeypatch,
    field,
    new_value,
):
    settings = _patch_signature_context(monkeypatch)
    before = service_factory.settings_signature()
    setattr(settings, field, new_value)
    after = service_factory.settings_signature()
    assert after != before


def test_settings_signature_changes_with_index_dir(monkeypatch, tmp_path):
    settings = _make_signature_settings()
    current_index_dir = tmp_path / "indexes-a"
    monkeypatch.setattr(service_factory, "load_settings", lambda: settings)
    monkeypatch.setattr(
        service_factory,
        "default_index_dir",
        lambda: current_index_dir,
    )
    monkeypatch.setattr(
        service_factory,
        "default_upload_dir",
        lambda: tmp_path / "uploads",
    )
    before = service_factory.settings_signature()
    current_index_dir = tmp_path / "indexes-b"
    after = service_factory.settings_signature()
    assert after != before


def test_settings_signature_excludes_api_key_contents(monkeypatch):
    test_key = "test-secret-never-log"
    settings = _make_signature_settings(
        siliconflow_api_key=test_key,
        minimax_api_key=test_key,
    )
    _patch_signature_context(monkeypatch, settings)
    captured_payloads = []
    real_sha256 = service_factory.hashlib.sha256

    def capture_sha256(payload):
        captured_payloads.append(payload.decode("utf-8"))
        return real_sha256(payload)

    monkeypatch.setattr(service_factory.hashlib, "sha256", capture_sha256)
    first = service_factory.settings_signature()
    settings.siliconflow_api_key = "another-fake-key"
    settings.minimax_api_key = "another-fake-key"
    second = service_factory.settings_signature()

    assert first == second
    assert captured_payloads
    assert all(test_key not in payload for payload in captured_payloads)
    assert all("another-fake-key" not in payload for payload in captured_payloads)
    assert all("api_key" not in payload for payload in captured_payloads)


def test_stable_payload_signature_ignores_dictionary_key_order():
    first = {
        "string": "value",
        "integer": 3,
        "float": 0.25,
        "boolean": True,
        "nothing": None,
        "path": Path("test-storage/indexes"),
    }
    second = dict(reversed(list(first.items())))
    assert service_factory._stable_payload_signature(first) == (
        service_factory._stable_payload_signature(second)
    )
