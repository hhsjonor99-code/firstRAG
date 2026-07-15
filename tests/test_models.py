"""数据模型测试。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.models import (  # noqa: E402
    ChatMessage,
    DocumentChunk,
    DocumentInfo,
    RetrievedChunk,
)


def _now() -> datetime:
    return datetime(2026, 7, 15, 10, 30, 0)


def test_document_info_roundtrip():
    info = DocumentInfo(
        document_id="abc123",
        file_name="abc123.docx",
        original_file_name="测试文档.docx",
        file_type="docx",
        file_hash="0" * 64,
        file_size=1024,
        created_at=_now(),
        chunk_count=3,
    )
    s = info.model_dump_json()
    info2 = DocumentInfo.model_validate_json(s)
    assert info2 == info
    assert info2.file_type == "docx"
    assert info2.original_file_name == "测试文档.docx"


def test_document_chunk_optional_fields():
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="abc123",
        content="hello",
        source_name="测试文档.docx",
        page_number=None,
        paragraph_number=2,
        heading="第一章",
        line_start=None,
        line_end=None,
        chunk_index=0,
        metadata={"language": "zh"},
    )
    assert chunk.paragraph_number == 2
    assert chunk.heading == "第一章"
    assert chunk.metadata == {"language": "zh"}
    # 序列化往返
    s = chunk.model_dump_json()
    c2 = DocumentChunk.model_validate_json(s)
    assert c2 == chunk


def test_retrieved_chunk_citation_format():
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="abc123",
        content="hello",
        source_name="t.txt",
        page_number=None,
        paragraph_number=None,
        heading=None,
        line_start=1,
        line_end=1,
        chunk_index=0,
        metadata={},
    )
    rc = RetrievedChunk(chunk=chunk, score=0.91, citation_id="S1")
    assert rc.citation_id == "S1"
    assert rc.score == pytest_approx(0.91)


def test_chat_message_default_empty_citations():
    m = ChatMessage(role="user", content="hi", created_at=_now())
    assert m.citations == []
    assert m.role == "user"


def test_chat_message_with_citations():
    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="abc123",
        content="hello",
        source_name="t.txt",
        page_number=None,
        paragraph_number=None,
        heading=None,
        line_start=1,
        line_end=1,
        chunk_index=0,
        metadata={},
    )
    rc = RetrievedChunk(chunk=chunk, score=0.5, citation_id="S1")
    m = ChatMessage(role="assistant", content="answer [S1]", citations=[rc], created_at=_now())
    assert len(m.citations) == 1
    assert m.citations[0].citation_id == "S1"


# 简易近似比较（避免引入 pytest 依赖，脚本可直接 python -m unittest 运行）
def pytest_approx(value: float, tol: float = 1e-9) -> bool:
    class _Aprox:
        def __eq__(self, other):
            return abs(other - value) <= tol

        def __repr__(self):
            return f"approx({value})"

    return _Aprox()


# 直接以 `python -m unittest` 运行时的入口
if __name__ == "__main__":  # pragma: no cover
    import unittest

    unittest.main()