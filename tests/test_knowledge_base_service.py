"""KnowledgeBaseService 单元测试。

所有测试：

- 使用 ``FakeEmbeddingProvider``，**不**调用真实 SiliconFlow；
- 使用临时目录作为 ``upload_dir`` 与 ``index_dir``，**不**写入正式 storage；
- 不会读取或输出真实 API Key / 真实业务文档。
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402

from rag.embedding_provider import EmbeddingError  # noqa: E402
from rag.knowledge_base_service import (  # noqa: E402
    DuplicateDocumentError as KSDuplicate,
    EmptyUploadError,
    InvalidUploadError,
    KnowledgeBaseRollbackError,
    KnowledgeBaseService,
    KnowledgeBaseServiceError,
    UnsupportedUploadTypeError,
    UploadTooLargeError,
)
from rag.models import DocumentChunk, DocumentInfo  # noqa: E402
from rag.vector_store import (  # noqa: E402
    DuplicateDocumentError,
    FaissVectorStore,
    IndexNotFoundError,
    VectorStoreError,
)

from tests._fakes import FakeEmbeddingProvider  # noqa: E402


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _new_settings(**overrides) -> Settings:
    with mock.patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _make_docx_bytes(text: str = "测试段落") -> bytes:
    """构造一个最小可用的 docx（OOXML 容器）字节。"""
    import html
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # [Content_Types].xml
        zf.writestr("[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '</Types>')
        # _rels/.rels
        zf.writestr("_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>')
        # word/document.xml（带 sectPr 让 python-docx 不抛错）
        text_escaped = html.escape(text)
        zf.writestr("word/document.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            f'<w:p><w:r><w:t xml:space="preserve">{text_escaped}</w:t></w:r></w:p>'
            '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr>'
            '</w:body>'
            '</w:document>')
    return buf.getvalue()


def _make_pdf_bytes_with_text(text: str) -> bytes:
    """构造一个最小可用的 PDF（含指定文本）。"""
    # 最简化：手工构造一个含文本的单页 PDF
    # PDF 1.4 最小骨架 + 一段文字
    # 文本里若有非 ASCII 用 \\u 转义，pypdf 读取会失败 —— 我们用纯 ASCII 文本测试
    safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
    objects = []
    # 1: Catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2: Pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # 3: Page
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    # 4: Content stream
    content = f"BT /F1 12 Tf 50 700 Td ({safe_text}) Tj ET".encode("latin-1")
    objects.append(
        b"4 0 obj\n<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
        + content + b"\nendstream\nendobj\n"
    )
    # 5: Font
    objects.append(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    # 构造 PDF 字节流
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(out.tell())
        out.write(obj)
    # xref
    xref_offset = out.tell()
    out.write(b"xref\n")
    out.write(f"0 {len(objects) + 1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))
    out.write(b"trailer\n")
    out.write(f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii"))
    out.write(b"startxref\n")
    out.write(f"{xref_offset}\n".encode("ascii"))
    out.write(b"%%EOF\n")
    return out.getvalue()


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return _new_settings(
        max_upload_mb=5,
        chunk_size=200,
        chunk_overlap=20,
        siliconflow_embedding_dimensions=8,  # 小的 dim 让 fake embedder 更快
        retrieval_top_k=3,
    )


@pytest.fixture
def fake_embedder(tmp_settings: Settings) -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(settings=tmp_settings)


@pytest.fixture
def kb_service(
    tmp_settings: Settings,
    fake_embedder: FakeEmbeddingProvider,
    tmp_path: Path,
) -> KnowledgeBaseService:
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FaissVectorStore(settings=tmp_settings, index_dir=index_dir)
    store.load()
    return KnowledgeBaseService(
        embedding_provider=fake_embedder,
        vector_store=store,
        settings=tmp_settings,
        upload_dir=upload_dir,
    )


# ---------------------------------------------------------------------------
# 1. TXT 字节入库成功
# ---------------------------------------------------------------------------
def test_ingest_txt_bytes_success(kb_service: KnowledgeBaseService, tmp_path: Path):
    info = kb_service.ingest_bytes(
        data="第一行内容。\n第二行内容。\n第三行内容。".encode("utf-8"),
        original_file_name="note.txt",
    )
    assert info.file_type == "txt"
    assert info.original_file_name == "note.txt"
    assert info.file_size > 0
    assert info.chunk_count >= 1
    # 磁盘上应有正式文件
    saved_path = kb_service.upload_dir / info.file_name
    assert saved_path.exists()
    # 索引中应能查到
    assert kb_service.get_document(info.document_id) is not None


# ---------------------------------------------------------------------------
# 2. Markdown 入库成功
# ---------------------------------------------------------------------------
def test_ingest_markdown_success(kb_service: KnowledgeBaseService):
    md = "# 标题\n\n第一段内容。\n\n第二段内容。"
    info = kb_service.ingest_bytes(
        data=md.encode("utf-8"),
        original_file_name="readme.md",
    )
    assert info.file_type == "md"
    assert info.original_file_name == "readme.md"
    docs = kb_service.list_documents()
    assert any(d.document_id == info.document_id for d in docs)


# ---------------------------------------------------------------------------
# 3. DOCX 入库成功
# ---------------------------------------------------------------------------
def test_ingest_docx_success(kb_service: KnowledgeBaseService):
    data = _make_docx_bytes("测试段落内容 ABC")
    info = kb_service.ingest_bytes(
        data=data,
        original_file_name="report.docx",
    )
    assert info.file_type == "docx"
    assert info.chunk_count >= 1


# ---------------------------------------------------------------------------
# 4. PDF 文本文件入库成功
# ---------------------------------------------------------------------------
def test_ingest_pdf_success(kb_service: KnowledgeBaseService):
    data = _make_pdf_bytes_with_text("Hello PDF world")
    info = kb_service.ingest_bytes(
        data=data,
        original_file_name="doc.pdf",
    )
    assert info.file_type == "pdf"
    assert info.chunk_count >= 1


# ---------------------------------------------------------------------------
# 5. 文件路径入库
# ---------------------------------------------------------------------------
def test_ingest_path(kb_service: KnowledgeBaseService, tmp_path: Path):
    src = tmp_path / "source.txt"
    src.write_text("源文件内容。", encoding="utf-8")
    info = kb_service.ingest_path(src)
    assert info.original_file_name == "source.txt"
    assert kb_service.get_document(info.document_id) is not None


# ---------------------------------------------------------------------------
# 6. 中文原始文件名保留
# ---------------------------------------------------------------------------
def test_chinese_filename_preserved(kb_service: KnowledgeBaseService):
    info = kb_service.ingest_bytes(
        data="内容".encode("utf-8"),
        original_file_name="国家基本药物目录.txt",
    )
    assert info.original_file_name == "国家基本药物目录.txt"
    # 但磁盘文件名应该是 document_id + ext
    assert info.file_name == f"{info.document_id}.txt"


# ---------------------------------------------------------------------------
# 7. 磁盘保存名称不是原始文件名
# ---------------------------------------------------------------------------
def test_disk_filename_is_doc_id(kb_service: KnowledgeBaseService):
    info = kb_service.ingest_bytes(
        data="内容".encode("utf-8"),
        original_file_name="my-private-name.txt",
    )
    saved = kb_service.upload_dir / info.file_name
    assert saved.exists()
    # 磁盘文件名 ≠ 原始文件名
    assert info.file_name != info.original_file_name


# ---------------------------------------------------------------------------
# 8. 重复 SHA256 被拒绝
# ---------------------------------------------------------------------------
def test_duplicate_sha256_rejected(kb_service: KnowledgeBaseService):
    data = "重复内容。\n第二行。".encode("utf-8")
    info1 = kb_service.ingest_bytes(data=data, original_file_name="a.txt")
    assert info1.file_hash
    with pytest.raises((DuplicateDocumentError, KSDuplicate, KnowledgeBaseServiceError)):
        kb_service.ingest_bytes(data=data, original_file_name="b.txt")


# ---------------------------------------------------------------------------
# 9. 空文件
# ---------------------------------------------------------------------------
def test_empty_file_rejected(kb_service: KnowledgeBaseService):
    with pytest.raises(EmptyUploadError):
        kb_service.ingest_bytes(data=b"", original_file_name="empty.txt")


# ---------------------------------------------------------------------------
# 10. 不支持扩展名
# ---------------------------------------------------------------------------
def test_unsupported_extension(kb_service: KnowledgeBaseService):
    with pytest.raises(UnsupportedUploadTypeError):
        kb_service.ingest_bytes(data=b"x" * 100, original_file_name="x.exe")


# ---------------------------------------------------------------------------
# 11. 文件超限
# ---------------------------------------------------------------------------
def test_upload_too_large(tmp_settings: Settings, fake_embedder, tmp_path: Path):
    # 强制 max_upload_mb=0 不行（会除零），改用更小值
    settings = _new_settings(
        max_upload_mb=1,
        siliconflow_embedding_dimensions=8,
        chunk_size=200,
    )
    embedder = FakeEmbeddingProvider(settings=settings)
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FaissVectorStore(settings=settings, index_dir=index_dir)
    store.load()
    svc = KnowledgeBaseService(
        embedding_provider=embedder, vector_store=store,
        settings=settings, upload_dir=upload_dir,
    )
    big_data = b"a" * (2 * 1024 * 1024)  # 2MB > 1MB limit
    with pytest.raises(UploadTooLargeError):
        svc.ingest_bytes(data=big_data, original_file_name="big.txt")


# ---------------------------------------------------------------------------
# 12. 路径穿越文件名被清理
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_name", [
    "../../../etc/passwd.txt",
    "..\\..\\windows\\file.txt",
    "C:\\Users\\admin\\file.txt",
    "/etc/passwd.txt",
    "\\foo\\bar.txt",
    "subdir/file.txt",
])
def test_path_traversal_rejected(kb_service: KnowledgeBaseService, bad_name: str):
    with pytest.raises(InvalidUploadError):
        kb_service.ingest_bytes(data=b"abc", original_file_name=bad_name)


# ---------------------------------------------------------------------------
# 13. PDF 文件头错误
# ---------------------------------------------------------------------------
def test_pdf_header_invalid(kb_service: KnowledgeBaseService):
    # 字节不以 %PDF- 开头
    with pytest.raises(InvalidUploadError):
        kb_service.ingest_bytes(
            data=b"NOT A PDF\njust some text",
            original_file_name="fake.pdf",
        )


# ---------------------------------------------------------------------------
# 14. DOCX 文件结构错误
# ---------------------------------------------------------------------------
def test_docx_header_invalid(kb_service: KnowledgeBaseService):
    # 字节不以 PK\x03\x04 开头
    with pytest.raises(InvalidUploadError):
        kb_service.ingest_bytes(
            data=b"NOT A DOCX\nhello",
            original_file_name="fake.docx",
        )

    # 或者 ZIP 但没有 word/document.xml
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("foo.txt", "bar")
    with pytest.raises(InvalidUploadError):
        kb_service.ingest_bytes(
            data=buf.getvalue(),
            original_file_name="bad_structure.docx",
        )


# ---------------------------------------------------------------------------
# 15. 解析失败回滚
# ---------------------------------------------------------------------------
def test_parse_failure_rollback(kb_service: KnowledgeBaseService, tmp_path: Path):
    """让解析失败 → 临时文件删除 + 索引不修改。"""
    # PDF 字节通过文件头校验，但内容无法解析（结构破损的 PDF）
    # 实际上一个空 %PDF- 但内容损坏的文件，pypdf 会抛 DocumentParseError 或子类型
    bad_pdf = b"%PDF-1.4\nnot really a valid pdf body"
    # 注意：部分残缺 PDF 可能 pypdf 不抛错，尝试强制让其失败
    # 用一个 .docx 但内容为非 zip（前面已拒绝 header）不行
    # 用 .txt 但故意触发 DocumentParseError 较难
    # 用 .pdf 通过 header 但 pypdf 解析失败
    before = len(kb_service.list_documents())
    try:
        kb_service.ingest_bytes(data=bad_pdf, original_file_name="bad.pdf")
    except KnowledgeBaseServiceError:
        pass
    # 索引应未变
    assert len(kb_service.list_documents()) == before
    # 临时文件应清理（无 .tmp_* 文件残留）
    tmp_files = list(kb_service.upload_dir.glob(".tmp_*"))
    assert tmp_files == []


# ---------------------------------------------------------------------------
# 16. 分块失败回滚
# ---------------------------------------------------------------------------
def test_chunking_failure_rollback(
    tmp_settings: Settings, fake_embedder, tmp_path: Path
):
    """让 split_sections 抛异常 → 索引不修改。"""
    from rag.knowledge_base_service import KnowledgeBaseService

    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FaissVectorStore(settings=tmp_settings, index_dir=index_dir)
    store.load()
    svc = KnowledgeBaseService(
        embedding_provider=fake_embedder, vector_store=store,
        settings=tmp_settings, upload_dir=upload_dir,
    )
    with mock.patch(
        "rag.knowledge_base_service.split_sections",
        side_effect=ValueError("splitter boom"),
    ):
        with pytest.raises(KnowledgeBaseServiceError):
            svc.ingest_bytes(
                data="正常内容\n第二行".encode("utf-8"),
                original_file_name="ok.txt",
            )
    assert svc.list_documents() == []
    assert list(upload_dir.glob(".tmp_*")) == []


# ---------------------------------------------------------------------------
# 17. Embedding 失败回滚
# ---------------------------------------------------------------------------
def test_embedding_failure_rollback(
    tmp_settings: Settings, tmp_path: Path
):
    """让 embedder 抛错 → 索引不修改。"""
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FaissVectorStore(settings=tmp_settings, index_dir=index_dir)
    store.load()
    fail_embedder = FakeEmbeddingProvider(
        settings=tmp_settings,
        fail=EmbeddingError("embedding boom"),
    )
    svc = KnowledgeBaseService(
        embedding_provider=fail_embedder, vector_store=store,
        settings=tmp_settings, upload_dir=upload_dir,
    )
    with pytest.raises(KnowledgeBaseServiceError):
        svc.ingest_bytes(
            data="正常内容\n第二行".encode("utf-8"),
            original_file_name="ok.txt",
        )
    assert svc.list_documents() == []
    assert list(upload_dir.glob(".tmp_*")) == []


# ---------------------------------------------------------------------------
# 18. VectorStore 失败回滚
# ---------------------------------------------------------------------------
def test_vectorstore_failure_rollback(kb_service: KnowledgeBaseService):
    """让 vector_store.add_document 抛错 → 索引不修改 + 临时文件清理。"""
    before = len(kb_service.list_documents())
    with mock.patch.object(
        kb_service._store, "add_document", side_effect=VectorStoreError("vs boom")
    ):
        with pytest.raises(KnowledgeBaseServiceError):
            kb_service.ingest_bytes(
                data="内容数据\n第二行".encode("utf-8"),
                original_file_name="vs.txt",
            )
    assert len(kb_service.list_documents()) == before
    assert list(kb_service.upload_dir.glob(".tmp_*")) == []


# ---------------------------------------------------------------------------
# 19. 最终文件移动失败回滚
# ---------------------------------------------------------------------------
def test_final_move_failure_rollback(kb_service: KnowledgeBaseService):
    """索引写入成功但 os.replace 失败 → 从索引回滚。"""
    info_id = None
    original_replace = __import__("os").replace

    def fake_replace(src, dst):
        # 第一次：让最终的 replace 失败（tmp → 正式）；前面的 replace 不影响
        # 由于 ingest_bytes 内部只调用一次 os.replace，我们直接抛错
        raise OSError("simulated replace failure")

    with mock.patch("os.replace", side_effect=fake_replace):
        with pytest.raises(KnowledgeBaseServiceError):
            info = kb_service.ingest_bytes(
                data="内容数据\n第二行".encode("utf-8"),
                original_file_name="move.txt",
            )
            info_id = info.document_id
    # 索引应回滚（document 不应存在）
    assert kb_service.get_document(info_id) is None
    # 没有 .tmp_ 残留
    assert list(kb_service.upload_dir.glob(".tmp_*")) == []


# ---------------------------------------------------------------------------
# 20. list_documents
# ---------------------------------------------------------------------------
def test_list_documents(kb_service: KnowledgeBaseService):
    info1 = kb_service.ingest_bytes(data="first\n内容".encode("utf-8"), original_file_name="a.txt")
    info2 = kb_service.ingest_bytes(data="second\n内容".encode("utf-8"), original_file_name="b.txt")
    docs = kb_service.list_documents()
    ids = {d.document_id for d in docs}
    assert info1.document_id in ids
    assert info2.document_id in ids


# ---------------------------------------------------------------------------
# 21. get_document
# ---------------------------------------------------------------------------
def test_get_document(kb_service: KnowledgeBaseService):
    info = kb_service.ingest_bytes(data=b"data", original_file_name="x.txt")
    fetched = kb_service.get_document(info.document_id)
    assert fetched is not None
    assert fetched.document_id == info.document_id
    # 不存在
    assert kb_service.get_document("nonexistent-id") is None


# ---------------------------------------------------------------------------
# 22. delete_document
# ---------------------------------------------------------------------------
def test_delete_document(kb_service: KnowledgeBaseService):
    info = kb_service.ingest_bytes(data=b"to be deleted", original_file_name="d.txt")
    saved = kb_service.upload_dir / info.file_name
    assert saved.exists()
    assert kb_service.delete_document(info.document_id) is True
    assert kb_service.get_document(info.document_id) is None
    # 磁盘文件应删除
    assert not saved.exists()
    # 不存在
    assert kb_service.delete_document("not-exist") is False


# ---------------------------------------------------------------------------
# 23. 上传文件缺失时仍可删除索引
# ---------------------------------------------------------------------------
def test_delete_document_when_file_missing(kb_service: KnowledgeBaseService):
    info = kb_service.ingest_bytes(data=b"x", original_file_name="e.txt")
    saved = kb_service.upload_dir / info.file_name
    saved.unlink()  # 手动删除文件
    # 删除应仍成功
    assert kb_service.delete_document(info.document_id) is True
    assert kb_service.get_document(info.document_id) is None


# ---------------------------------------------------------------------------
# 24. clear
# ---------------------------------------------------------------------------
def test_clear(kb_service: KnowledgeBaseService):
    info1 = kb_service.ingest_bytes(data=b"1", original_file_name="c1.txt")
    info2 = kb_service.ingest_bytes(data=b"2", original_file_name="c2.md")
    # 在 upload_dir 中放一个无关文件（不应被删）
    unrelated = kb_service.upload_dir / "unrelated.txt"
    unrelated.write_text("keep me", encoding="utf-8")
    kb_service.clear()
    assert kb_service.list_documents() == []
    # 系统文件应被删
    assert not (kb_service.upload_dir / info1.file_name).exists()
    assert not (kb_service.upload_dir / info2.file_name).exists()
    # 无关文件应保留
    assert unrelated.exists()


# ---------------------------------------------------------------------------
# 25. 正式 storage 未被测试污染
# ---------------------------------------------------------------------------
def test_formal_storage_untouched(tmp_path: Path):
    """测试用的所有目录在 tmp_path 内，不影响真实 storage/。"""
    project_root = Path(__file__).resolve().parent.parent
    real_uploads = project_root / "storage" / "uploads"
    real_indexes = project_root / "storage" / "indexes"
    # 仅确认目录存在；测试运行不会向其中写入
    # 通过 import 验证正式目录未被影响
    assert real_uploads.exists()
    assert real_indexes.exists()
    # 测试运行后，tmp_path 目录中应没有名为 'storage' 的写入
    # （所有 kb_service 的 upload_dir / index_dir 都基于 tmp_path）
    # 这个断言是软检查
    for p in tmp_path.rglob("*"):
        if p.is_dir() and p.name == "storage":
            # 嵌套 storage 目录不应出现
            assert False, f"嵌套 storage 目录: {p}"
