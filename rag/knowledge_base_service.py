"""firstRAG 知识库服务（业务编排层）。

职责：

1. 接收上传文件（路径或字节）。
2. 校验文件名 / 扩展名 / 大小 / 文件头。
3. 计算 SHA256，去重。
4. 保存原始文件到 ``storage/uploads/{document_id}{ext}``。
5. 调用解析器 → 切分器 → Embedding → FaissVectorStore。
6. 维护「索引与文件一致」的事务语义：
   - 临时文件先入；索引保存成功后再原子移动到正式目录。
   - 任一步骤失败 → 清理临时文件、撤销索引改动。

公开 API：

- :meth:`ingest_path` —— 从磁盘路径入库（内部统一走 ``ingest_bytes``）
- :meth:`ingest_bytes` —— 从字节数据入库
- :meth:`list_documents`
- :meth:`get_document`
- :meth:`delete_document`
- :meth:`clear`

异常层次：

- :class:`KnowledgeBaseServiceError` —— 基类
- :class:`InvalidUploadError` —— 文件名 / 后缀 / 路径非法
- :class:`UploadTooLargeError` —— 超过 ``settings.max_upload_mb``
- :class:`UnsupportedUploadTypeError` —— 扩展名不在白名单
- :class:`EmptyUploadError` —— 字节为空 / 文件大小为 0
- :class:`KnowledgeBaseRollbackError` —— 入库成功但回滚失败（异常嵌套）

安全约束：

- 原始文件名仅作为展示，不决定磁盘保存路径。
- 磁盘文件名为 ``{document_id}{extension}``，``document_id`` 来自 uuid4 hex。
- 路径穿越（``../``、``..\\``、绝对路径、Windows 盘符）一律拒绝。
- 日志只记录：document_id、字节数、扩展名、解析结果数量、最终 chunk 数、
  Embedding 维度、错误类型；**不**记录文档正文、API Key、请求头。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.logging import get_logger
from config.settings import Settings
from config.settings import get_settings as _get_settings

from .embedding_provider import EmbeddingError, EmbeddingProvider
from .models import DocumentChunk, DocumentInfo
from .parsers import (
    BLOCK_TYPE_PARAGRAPH,
    BLOCK_TYPE_TABLE,
    BLOCK_TYPE_TEXTBOX,
    DocumentParseError,
    EmptyDocumentError,
    TextEncodingError,
    UnsupportedFileTypeError,
    UnsupportedScannedPDFError,
    parse_document,
    supported_extensions,
)
from .splitter import SplitOptions, split_sections
from .vector_store import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    FaissVectorStore,
    VectorStoreError,
)


# ---------------------------------------------------------------------------
# 异常层次
# ---------------------------------------------------------------------------
class KnowledgeBaseServiceError(RuntimeError):
    """知识库服务错误的基类。"""


class InvalidUploadError(KnowledgeBaseServiceError):
    """上传文件本身不合法（路径 / 名称 / 后缀）。"""


class UploadTooLargeError(KnowledgeBaseServiceError):
    """文件大小超过 ``settings.max_upload_mb``。"""


class UnsupportedUploadTypeError(KnowledgeBaseServiceError):
    """扩展名不在白名单。"""


class EmptyUploadError(KnowledgeBaseServiceError):
    """上传内容为空。"""


class KnowledgeBaseRollbackError(KnowledgeBaseServiceError):
    """入库主流程成功，但回滚阶段发生次生错误。

    异常 ``__cause__`` 保留原始主错误，``secondary_error`` 属性记录回滚错误。
    """


# ---------------------------------------------------------------------------
# 路径安全
# ---------------------------------------------------------------------------
_DISALLOWED_PATH_FRAGMENTS = ("..",)
_DRIVE_LETTER_PREFIX = re_drive = __import__("re").compile(r"^[a-zA-Z]:[\\/]")


def _sanitize_original_file_name(name: str) -> str:
    """去除路径前缀 / 盘符 / 穿越段；返回安全的 basename。

    规则：
    1. 拒绝空名 / 非字符串。
    2. 拒绝含 ``..`` 段。
    3. 拒绝 Windows 盘符前缀（``C:`` / ``D:`` 等）。
    4. 拒绝绝对路径（``/foo`` 或 ``\\foo`` 开头）。
    5. 拒绝含路径分隔符（``/`` 或 ``\\``）—— 一律视作试图用路径决定磁盘位置。
    6. 返回 ``Path(name).name`` + 清理控制字符。

    异常：:class:`InvalidUploadError`
    """
    import re

    if not isinstance(name, str):
        raise InvalidUploadError("original_file_name 必须是 str 类型。")
    raw = name.strip()
    if not raw:
        raise InvalidUploadError("original_file_name 不能为空。")
    # Windows 盘符
    if _DRIVE_LETTER_PREFIX.match(raw):
        raise InvalidUploadError(
            f"original_file_name 含盘符，不允许：{raw[:8]!r}"
        )
    # 包含 .. 路径片段
    if ".." in raw.replace("\\", "/").split("/"):
        raise InvalidUploadError(
            f"original_file_name 含 '..' 路径片段：{raw!r}"
        )
    # 绝对路径
    if raw.startswith("/") or raw.startswith("\\"):
        raise InvalidUploadError(
            f"original_file_name 为绝对路径，不允许：{raw!r}"
        )
    # 含路径分隔符（子目录形式）：拒绝
    if "/" in raw or "\\" in raw:
        raise InvalidUploadError(
            f"original_file_name 含路径分隔符：{raw!r}"
        )
    # 取 basename（防御性）
    base = Path(raw).name
    if not base:
        raise InvalidUploadError(
            f"original_file_name 解析后为空：{raw!r}"
        )
    # 清理控制字符
    base = re.sub(r"[\x00-\x1f\x7f]", "", base)
    if not base:
        raise InvalidUploadError(
            f"original_file_name 含全部控制字符：{raw!r}"
        )
    return base


# ---------------------------------------------------------------------------
# 文件头校验
# ---------------------------------------------------------------------------
_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"


def _validate_file_header(data: bytes, extension: str) -> None:
    """基于扩展名校验文件头。"""
    if extension == ".pdf":
        if not data.startswith(_PDF_MAGIC):
            raise InvalidUploadError(
                "PDF 文件头校验失败：缺少 %PDF- 签名。"
            )
    elif extension == ".docx":
        # DOCX 是 ZIP 容器；校验前 4 字节
        if not data.startswith(_ZIP_MAGIC):
            raise InvalidUploadError(
                "DOCX 文件头校验失败：不是有效的 ZIP/OOXML 容器。"
            )
        # 进一步校验是否含 word/document.xml（OOXML 关键文件）
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as zf:
                names = zf.namelist()
                if "word/document.xml" not in names:
                    raise InvalidUploadError(
                        "DOCX 缺少关键 OOXML 文件 word/document.xml。"
                    )
        except zipfile.BadZipFile as exc:
            raise InvalidUploadError(
                f"DOCX 解析为 ZIP 失败：{type(exc).__name__}。"
            ) from exc
    # txt / md / markdown：仅做 UTF-8/GB18030 解码尝试；不强校验
    return None


# ---------------------------------------------------------------------------
# KnowledgeBaseService
# ---------------------------------------------------------------------------
class KnowledgeBaseService:
    """知识库业务编排服务。

    :param embedding_provider: 满足 :class:`EmbeddingProvider` 协议的对象。
    :param vector_store: 已加载的 :class:`FaissVectorStore`。
    :param settings: 全局配置。
    :param upload_dir: 原始文件保存目录；缺省 ``<project>/storage/uploads``。
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: FaissVectorStore,
        settings: Optional[Settings] = None,
        upload_dir: Optional[Path] = None,
    ) -> None:
        self._embedding = embedding_provider
        self._store = vector_store
        self._settings = settings or _get_settings()
        self._upload_dir = Path(upload_dir) if upload_dir else self._default_upload_dir()
        self._log = get_logger("rag.knowledge_base_service")

    @property
    def upload_dir(self) -> Path:
        return self._upload_dir

    def _default_upload_dir(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "storage" / "uploads"

    # ------------------------------------------------------------------
    # 公共 API：入库
    # ------------------------------------------------------------------
    def ingest_path(self, path: Path) -> DocumentInfo:
        """从磁盘路径入库。

        :raises FileNotFoundError: 文件不存在。
        :raises KnowledgeBaseServiceError: 见 :meth:`ingest_bytes`。
        """
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"文件不存在或不是普通文件：{p}")
        data = p.read_bytes()
        # 原始文件名：使用 basename（防止用户传入含路径的字符串）
        return self.ingest_bytes(data=data, original_file_name=p.name)

    def ingest_bytes(
        self,
        data: bytes,
        original_file_name: str,
    ) -> DocumentInfo:
        """从字节数据入库。

        完整流程（每步失败都会回滚）：

        1. 校验文件名 / 扩展名 / 大小 / 字节非空 / 文件头。
        2. 计算 SHA256。
        3. 重复检查（``vector_store.has_document_hash``）。
        4. 生成 ``document_id``（uuid4 hex）。
        5. 解析 → 切分 → Embedding。
        6. ``vector_store.add_document()`` + ``save()``。
        7. 临时上传文件原子移动到正式目录。
        8. 返回 :class:`DocumentInfo`。

        :raises InvalidUploadError: 文件名 / 头非法。
        :raises UploadTooLargeError: 超过大小限制。
        :raises UnsupportedUploadTypeError: 扩展名不支持。
        :raises EmptyUploadError: 字节为空。
        :raises DuplicateDocumentError: 同一 SHA256 已存在。
        :raises KnowledgeBaseServiceError: 解析 / 切分 / Embedding / 索引 失败。
        """
        safe_name = _sanitize_original_file_name(original_file_name)
        ext = Path(safe_name).suffix.lower()
        if ext not in set(supported_extensions()):
            raise UnsupportedUploadTypeError(
                f"不支持的文件类型 '{ext}'；支持：{sorted(supported_extensions())}"
            )
        # 字节基础校验
        if not isinstance(data, (bytes, bytearray)):
            raise InvalidUploadError("data 必须是 bytes 类型。")
        if len(data) == 0:
            raise EmptyUploadError("上传内容为空。")
        max_bytes = int(self._settings.max_upload_mb) * 1024 * 1024
        if len(data) > max_bytes:
            raise UploadTooLargeError(
                f"文件大小 {len(data)} 字节 超过 max_upload_mb={self._settings.max_upload_mb}。"
            )
        # 文件头校验
        _validate_file_header(bytes(data), ext)

        # SHA256
        file_hash = hashlib.sha256(bytes(data)).hexdigest()

        # 重复检查
        if not self._store.is_loaded:
            raise KnowledgeBaseServiceError(
                "VectorStore 未加载；请先调用 vector_store.load()。"
            )
        if self._store.has_document_hash(file_hash):
            existing = self._store.get_document_by_hash(file_hash)
            existing_id = existing.document_id if existing else "?"
            raise DuplicateDocumentError(
                f"file_hash={file_hash[:12]}... 已存在（document_id={existing_id}）。"
            )

        # 生成 document_id
        document_id = uuid.uuid4().hex

        # 准备临时上传目录
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".tmp_{document_id}_", suffix=ext, dir=str(self._upload_dir)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(bytes(data))
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"临时上传文件写入失败：{type(exc).__name__}。"
            ) from exc

        # 解析 / 切分 / Embedding / 入库
        try:
            sections = parse_document(tmp_path)
        except UnsupportedScannedPDFError:
            self._safe_unlink(tmp_path)
            raise
        except (DocumentParseError, TextEncodingError) as exc:
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"文档解析失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"文档解析失败（未预期异常）：{type(exc).__name__}。"
            ) from exc

        try:
            chunks = split_sections(
                sections=sections,
                document_id=document_id,
                options=SplitOptions(),
            )
        except Exception as exc:  # noqa: BLE001
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"文档分块失败：{type(exc).__name__}。"
            ) from exc

        if not chunks:
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                "文档分块结果为空；无法入库。"
            )

        try:
            vectors = self._embedding.embed_documents([c.content for c in chunks])
        except EmbeddingError as exc:
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"Embedding 失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"Embedding 失败（未预期异常）：{type(exc).__name__}。"
            ) from exc

        # 构造 DocumentInfo
        now = datetime.now(timezone.utc)
        doc_info = DocumentInfo(
            document_id=document_id,
            file_name=f"{document_id}{ext}",
            original_file_name=safe_name,
            file_type=ext.lstrip("."),
            file_hash=file_hash,
            file_size=len(data),
            created_at=now,
            chunk_count=len(chunks),
        )

        # 写入索引
        try:
            self._store.add_document(doc_info, chunks, vectors)
        except DuplicateDocumentError:
            # 极端竞态：上传时别人抢先入库
            self._safe_unlink(tmp_path)
            raise
        except VectorStoreError as exc:
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"向量索引写入失败：{type(exc).__name__}。"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            self._safe_unlink(tmp_path)
            raise KnowledgeBaseServiceError(
                f"向量索引写入失败（未预期异常）：{type(exc).__name__}。"
            ) from exc

        # 索引成功 → 临时文件原子移动到正式目录
        final_path = self._upload_dir / doc_info.file_name
        try:
            os.replace(str(tmp_path), str(final_path))
        except OSError as exc:
            # 索引已写入但文件移动失败 → 从索引中删除
            self._log.error(
                "索引写入后文件移动失败：document_id=%s 错误类型=%s",
                document_id,
                type(exc).__name__,
            )
            rollback_err: Optional[BaseException] = None
            try:
                self._store.delete_document(document_id)
            except Exception as rb_exc:  # noqa: BLE001
                rollback_err = rb_exc
                self._log.error(
                    "回滚索引删除失败：document_id=%s 错误类型=%s",
                    document_id,
                    type(rb_exc).__name__,
                )
            self._safe_unlink(tmp_path)
            if rollback_err is not None:
                raise KnowledgeBaseRollbackError(
                    f"文件移动失败且索引回滚失败：主错误={type(exc).__name__}；"
                    f"回滚错误={type(rollback_err).__name__}"
                ) from exc
            raise KnowledgeBaseServiceError(
                f"文件移动失败：{type(exc).__name__}；已从索引回滚。"
            ) from exc

        self._log.info(
            "入库完成：document_id=%s ext=%s size=%d chunks=%d",
            document_id,
            ext,
            len(data),
            len(chunks),
        )
        return doc_info

    # ------------------------------------------------------------------
    # 公共 API：查询 / 删除 / 清空
    # ------------------------------------------------------------------
    def list_documents(self) -> list[DocumentInfo]:
        return self._store.list_documents()

    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        return self._store.get_document_by_id(document_id)

    def delete_document(self, document_id: str) -> bool:
        """删除指定 document。

        顺序：

        1. 校验 document_id 存在；不存在返回 ``False``。
        2. 从 VectorStore 删除并保存快照。
        3. 删除 uploads 中对应文件（不存在时仅 warning）。

        :returns: 删除成功（含文件缺失）返回 True；document_id 不存在返回 False。
        """
        if not self._store.is_loaded:
            raise KnowledgeBaseServiceError("VectorStore 未加载。")
        doc = self._store.get_document_by_id(document_id)
        if doc is None:
            return False
        # 先从索引删除（保证索引与文件状态一致）
        try:
            deleted = self._store.delete_document(document_id)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise KnowledgeBaseServiceError(
                f"删除索引失败：{type(exc).__name__}。"
            ) from exc
        if not deleted:
            return False
        # 再删除 uploads 中的实际文件
        upload_path = self._upload_dir / doc.file_name
        if upload_path.exists():
            try:
                upload_path.unlink()
            except OSError as exc:
                # 索引已删但文件删除失败 —— warning，不视作致命错误
                self._log.warning(
                    "已删除索引但文件删除失败：document_id=%s path=%s 错误类型=%s",
                    document_id,
                    str(upload_path),
                    type(exc).__name__,
                )
        else:
            self._log.warning(
                "删除文档时文件已不存在：document_id=%s path=%s",
                document_id,
                str(upload_path),
            )
        return True

    def clear(self) -> None:
        """清空整个知识库：先清空 VectorStore 并保存空快照；再清理 uploads。

        只删除 ``document_id{ext}`` 命名的系统文件；不会触碰 upload_dir 中
        任何非系统命名的文件（如 .gitkeep、用户手放文件等）。
        """
        if not self._store.is_loaded:
            raise KnowledgeBaseServiceError("VectorStore 未加载。")
        # 先收集要删除的 file_name（之后索引会清空）
        keep_filenames = {d.file_name for d in self._store.list_documents()}
        # 清空 VectorStore（带 save 快照回滚）
        self._store.clear()
        # 再删除 uploads 中对应文件
        if not self._upload_dir.exists():
            return
        failures: list[tuple[str, str]] = []
        for fname in keep_filenames:
            path = self._upload_dir / fname
            if not path.exists():
                continue
            try:
                path.unlink()
            except OSError as exc:
                failures.append((fname, type(exc).__name__))
        if failures:
            self._log.warning(
                "清空后部分文件删除失败：%s",
                "; ".join(f"{n}({e})" for n, e in failures),
            )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_unlink(path: Path) -> None:
        """尽力删除文件；任何异常都吞掉。"""
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


# 显式 re-export 公共符号
__all__ = [
    "KnowledgeBaseService",
    "KnowledgeBaseServiceError",
    "InvalidUploadError",
    "UploadTooLargeError",
    "UnsupportedUploadTypeError",
    "EmptyUploadError",
    "KnowledgeBaseRollbackError",
]
