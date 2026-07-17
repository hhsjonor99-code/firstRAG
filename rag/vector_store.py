"""firstRAG FAISS 向量库与本地持久化。

核心设计：

- **IndexFlatIP** + L2 归一化 → 余弦相似度检索。
- **版本快照 + CURRENT 指针**：避免出现「新 faiss.index + 旧 chunks.jsonl」
  的撕裂状态。每次 :meth:`FaissVectorStore.save` 都生成一个新的快照目录，
  写完 4 个文件并完成一致性校验后，原子切换 ``CURRENT``；切换前的旧快照
  始终可读，保存失败不会破坏可用性。
- **延迟加载**：构造时不读磁盘；调用 :meth:`load` 时才读。
- **索引一致性校验**：加载时对 manifest / index.ntotal / chunk 数 /
  document_id / chunk_id / file_hash 做完整比对，任一不一致抛专用异常。
- **不删除原文档**：本模块不触碰 ``storage/uploads`` 中的原始文件，
  仅维护 FAISS 索引与元数据。
- **不调用远程 API**：所有向量化由调用方提供 ``np.ndarray``；本模块不依赖
  SiliconFlow 或任何外部服务。
- **向量内存缓存**：``self._vectors`` 与 ``self._chunks`` 等长、同序，
  便于 delete / rollback 时直接基于已知向量重建索引，**不**依赖
  ``faiss.downcast_index(index).xb`` 这类内部 API。

异常层次（按严重度递增）：

- :class:`VectorStoreError` —— 所有错误的基类
- :class:`VectorValidationError` —— 向量维度 / dtype / NaN / 空 / 全零
- :class:`DuplicateDocumentError` —— 同一 file_hash 重复入库
- :class:`DocumentNotFoundError` —— 删除不存在的 document_id
- :class:`IndexNotFoundError` —— 快照目录或文件缺失
- :class:`IndexCorruptedError` —— index.ntotal 与 chunks 不一致等
- :class:`IndexIncompatibleError` —— manifest 与当前 Settings 不一致
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import faiss
import numpy as np

from config.logging import get_logger
from config.settings import Settings

from .models import DocumentChunk, DocumentInfo


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class VectorStoreError(RuntimeError):
    """向量库通用错误基类。"""


class VectorValidationError(VectorStoreError):
    """向量本身不合法（维度、dtype、NaN、空、全零）。"""


class DuplicateDocumentError(VectorStoreError):
    """同一 file_hash 已存在（重复入库）。"""


class DocumentNotFoundError(VectorStoreError):
    """删除 / 查询时找不到对应 document_id。"""


class IndexNotFoundError(VectorStoreError):
    """快照目录或必需文件缺失。"""


class IndexCorruptedError(VectorStoreError):
    """索引或元数据损坏（index.ntotal 与 chunks 不一致等）。"""


class IndexIncompatibleError(VectorStoreError):
    """manifest 与当前 Settings 不一致。"""


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
_SNAPSHOT_ID_RE = re.compile(r"^\d{8}T\d{6}_[0-9a-fA-F]{8}$")


def _new_snapshot_id() -> str:
    """生成 ``YYYYMMDDTHHMMSS_xxxxxxxx`` 形式的快照 ID。"""
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    import uuid

    suffix = uuid.uuid4().hex[:8]
    return f"{now}_{suffix}"


# ---------------------------------------------------------------------------
# FaissVectorStore
# ---------------------------------------------------------------------------
class FaissVectorStore:
    """FAISS 向量库 + 元数据持久化。

    :param settings: 全局配置。
    :param index_dir: 索引根目录；缺省 ``<project>/storage/indexes``。
    """

    INDEX_FILENAME = "faiss.index"
    CHUNKS_FILENAME = "chunks.jsonl"
    DOCUMENTS_FILENAME = "documents.jsonl"
    MANIFEST_FILENAME = "manifest.json"
    CURRENT_FILENAME = "CURRENT"
    SNAPSHOT_DIRNAME = "snapshots"

    MANIFEST_SCHEMA_VERSION = 1
    INDEX_TYPE = "IndexFlatIP"
    EMBEDDING_PROVIDER = "siliconflow"

    def __init__(self, settings: Settings, index_dir: Optional[Path] = None) -> None:
        self._settings = settings
        self._index_dir = Path(index_dir) if index_dir else self._default_index_dir()
        self._snapshots_dir = self._index_dir / self.SNAPSHOT_DIRNAME
        self._current_file = self._index_dir / self.CURRENT_FILENAME

        self._index: Optional[faiss.Index] = None
        self._documents: dict[str, DocumentInfo] = {}  # document_id -> info
        self._hash_index: dict[str, str] = {}          # file_hash -> document_id
        self._chunks: list[DocumentChunk] = []
        self._chunk_ids: dict[str, int] = {}           # chunk_id -> list index
        self._vectors: Optional[np.ndarray] = None     # (chunk_count, dim) float32

        self._log = get_logger("rag.vector_store")

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @property
    def dimensions(self) -> int:
        return int(self._settings.siliconflow_embedding_dimensions)

    @property
    def index_dir(self) -> Path:
        return self._index_dir

    @property
    def snapshots_dir(self) -> Path:
        return self._snapshots_dir

    @property
    def current_file(self) -> Path:
        return self._current_file

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _default_index_dir(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "storage" / "indexes"

    # ------------------------------------------------------------------
    # 公共 API：add / search / has / get / list / delete / clear
    # ------------------------------------------------------------------
    def add_document(
        self,
        document: DocumentInfo,
        chunks: list[DocumentChunk],
        vectors: np.ndarray,
    ) -> None:
        """添加一个文档的所有 chunks 与向量。

        :raises DuplicateDocumentError: file_hash 已存在
        :raises VectorValidationError: vectors/chunks 不合法
        :raises VectorStoreError: 其它校验失败
        """
        if not self.is_loaded:
            raise VectorStoreError("向量库未加载，请先调用 load()。")

        if not chunks:
            raise VectorValidationError("chunks 列表为空，禁止入库。")

        if document.file_hash in self._hash_index:
            raise DuplicateDocumentError(
                f"file_hash={document.file_hash} 已存在对应的文档（document_id="
                f"{self._hash_index[document.file_hash]}），不允许重复入库。"
            )
        if document.document_id in self._documents:
            existing = self._documents[document.document_id]
            if existing.file_hash != document.file_hash:
                raise VectorStoreError(
                    f"document_id={document.document_id} 已被其它文档占用。"
                )

        for c in chunks:
            if c.document_id != document.document_id:
                raise VectorStoreError(
                    f"chunk_id={c.chunk_id} 的 document_id={c.document_id} 与传入 "
                    f"document.document_id={document.document_id} 不一致。"
                )
            if c.chunk_id in self._chunk_ids:
                raise VectorStoreError(
                    f"chunk_id={c.chunk_id} 已存在，禁止重复。"
                )

        prepared = self._prepare_vectors(vectors, expected_count=len(chunks))

        # snapshot 用于回滚
        snap_docs = dict(self._documents)
        snap_hash = dict(self._hash_index)
        snap_chunks = list(self._chunks)
        snap_chunk_ids = dict(self._chunk_ids)
        snap_vectors = None if self._vectors is None else self._vectors.copy()

        # 校正 document.chunk_count
        if document.chunk_count != len(chunks):
            self._log.warning(
                "document.chunk_count=%d 与实际 chunks=%d 不一致，自动校正。",
                document.chunk_count,
                len(chunks),
            )
            document = document.model_copy(update={"chunk_count": len(chunks)})

        # 写内存
        start = self.chunk_count
        new_chunks = list(self._chunks)
        for i, c in enumerate(chunks):
            new_chunks.append(c)
        new_vectors = (
            np.vstack([self._vectors, prepared]) if self._vectors is not None
            else prepared.copy()
        )
        new_chunk_ids = dict(self._chunk_ids)
        for i, c in enumerate(chunks):
            new_chunk_ids[c.chunk_id] = start + i

        self._chunks = new_chunks
        self._chunk_ids = new_chunk_ids
        self._vectors = new_vectors
        self._documents[document.document_id] = document
        self._hash_index[document.file_hash] = document.document_id

        # 重建索引（保证与 _vectors 完全一致）
        self._index = faiss.IndexFlatIP(self.dimensions)
        if self._vectors.shape[0] > 0:
            self._index.add(self._vectors)

        try:
            self.save()
        except Exception as exc:
            # 回滚
            self._documents = snap_docs
            self._hash_index = snap_hash
            self._chunks = snap_chunks
            self._chunk_ids = snap_chunk_ids
            self._vectors = snap_vectors
            self._index = faiss.IndexFlatIP(self.dimensions)
            if snap_vectors is not None and snap_vectors.shape[0] > 0:
                self._index.add(snap_vectors)
            self._log.error(
                "add_document 保存失败，已回滚内存状态：%s",
                type(exc).__name__,
            )
            raise

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[tuple[DocumentChunk, float]]:
        if not self.is_loaded:
            raise VectorStoreError("向量库未加载，请先调用 load()。")
        if not isinstance(top_k, int) or top_k <= 0:
            raise VectorValidationError(f"top_k 必须为正整数，得到 {top_k!r}。")
        if self.chunk_count == 0 or self._index is None or self._index.ntotal == 0:
            return []

        arr = self._prepare_query(query_vector)
        k = min(top_k, self.chunk_count)
        scores, indices = self._index.search(arr, k)

        results: list[tuple[DocumentChunk, float]] = []
        for raw_idx, raw_score in zip(indices[0].tolist(), scores[0].tolist()):
            if raw_idx == -1:
                continue
            if raw_idx < 0 or raw_idx >= len(self._chunks):
                continue
            chunk = self._chunks[int(raw_idx)]
            results.append((chunk, float(raw_score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def has_document_hash(self, file_hash: str) -> bool:
        return file_hash in self._hash_index

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentInfo]:
        doc_id = self._hash_index.get(file_hash)
        if doc_id is None:
            return None
        return self._documents.get(doc_id)

    def get_document_by_id(self, document_id: str) -> Optional[DocumentInfo]:
        return self._documents.get(document_id)

    def list_documents(self) -> list[DocumentInfo]:
        return sorted(self._documents.values(), key=lambda d: d.created_at)

    def list_chunks_by_document(self, document_id: str) -> list[DocumentChunk]:
        return [c for c in self._chunks if c.document_id == document_id]

    def delete_document(self, document_id: str) -> bool:
        """删除指定文档的所有 chunks 与元数据。

        :returns: 删除成功返回 True；document_id 不存在返回 False。
        """
        if not self.is_loaded:
            raise VectorStoreError("向量库未加载，请先调用 load()。")
        if document_id not in self._documents:
            return False

        snap_docs = dict(self._documents)
        snap_hash = dict(self._hash_index)
        snap_chunks = list(self._chunks)
        snap_chunk_ids = dict(self._chunk_ids)
        snap_vectors = None if self._vectors is None else self._vectors.copy()

        # 内存修改
        doc = self._documents.pop(document_id)
        self._hash_index.pop(doc.file_hash, None)

        keep_indices = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
        new_chunks = [self._chunks[i] for i in keep_indices]
        new_vectors = self._vectors[keep_indices].copy() if self._vectors is not None and keep_indices else None
        new_chunk_ids = {c.chunk_id: i for i, c in enumerate(new_chunks)}

        self._chunks = new_chunks
        self._chunk_ids = new_chunk_ids
        self._vectors = new_vectors
        # 重建索引
        self._index = faiss.IndexFlatIP(self.dimensions)
        if self._vectors is not None and self._vectors.shape[0] > 0:
            self._index.add(self._vectors)

        try:
            self.save()
        except Exception as exc:
            self._documents = snap_docs
            self._hash_index = snap_hash
            self._chunks = snap_chunks
            self._chunk_ids = snap_chunk_ids
            self._vectors = snap_vectors
            self._index = faiss.IndexFlatIP(self.dimensions)
            if snap_vectors is not None and snap_vectors.shape[0] > 0:
                self._index.add(snap_vectors)
            self._log.error("delete_document 保存失败，已回滚：%s", type(exc).__name__)
            raise
        return True

    def clear(self) -> None:
        """清空整个知识库（但保留历史快照）。"""
        if not self.is_loaded:
            raise VectorStoreError("向量库未加载，请先调用 load()。")
        snap_docs = dict(self._documents)
        snap_hash = dict(self._hash_index)
        snap_chunks = list(self._chunks)
        snap_chunk_ids = dict(self._chunk_ids)
        snap_vectors = None if self._vectors is None else self._vectors.copy()

        self._documents.clear()
        self._hash_index.clear()
        self._chunks.clear()
        self._chunk_ids.clear()
        self._vectors = None
        self._index = faiss.IndexFlatIP(self.dimensions)

        try:
            self.save()
        except Exception as exc:
            self._documents = snap_docs
            self._hash_index = snap_hash
            self._chunks = snap_chunks
            self._chunk_ids = snap_chunk_ids
            self._vectors = snap_vectors
            self._index = faiss.IndexFlatIP(self.dimensions)
            if snap_vectors is not None and snap_vectors.shape[0] > 0:
                self._index.add(snap_vectors)
            self._log.error("clear 保存失败，已回滚：%s", type(exc).__name__)
            raise

    def rebuild(self) -> None:
        """从内存中的 chunks / vectors 重新构建索引。"""
        if not self.is_loaded:
            raise VectorStoreError("向量库未加载，请先调用 load()。")
        self._index = faiss.IndexFlatIP(self.dimensions)
        if self._vectors is not None and self._vectors.shape[0] > 0:
            self._index.add(self._vectors)
        self.save()

    def prune_snapshots(self, keep_last: int = 2) -> int:
        """保留最近 ``keep_last`` 个快照，删除更早的。返回删除数量。"""
        if keep_last < 1:
            raise VectorValidationError(f"keep_last 必须 >= 1，得到 {keep_last}。")
        snapshots = self._list_snapshots()
        current_id = self._read_current_snapshot_id()
        to_delete = [s for s in snapshots if s != current_id]
        to_delete = to_delete[: max(0, len(to_delete) - keep_last)]
        for sid in to_delete:
            try:
                shutil.rmtree(self._snapshots_dir / sid)
                self._log.info("已删除旧快照: %s", sid)
            except OSError as exc:
                self._log.warning("删除快照失败 %s: %s", sid, exc)
        return len(to_delete)

    # ------------------------------------------------------------------
    # 公共 API：save / load
    # ------------------------------------------------------------------
    def save(self) -> None:
        """写入新快照并原子切换 CURRENT。"""
        if not self.is_loaded:
            raise VectorStoreError("向量库未加载，无法 save。")

        self._validate_in_memory_state()
        snapshot_id = _new_snapshot_id()
        tmp_dir = self._snapshots_dir / f"{snapshot_id}.tmp"
        final_dir = self._snapshots_dir / snapshot_id

        if final_dir.exists():
            raise VectorStoreError(f"快照 ID 冲突：{final_dir} 已存在。")

        try:
            self._snapshots_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir.mkdir(parents=False, exist_ok=False)

            # faiss.index
            assert self._index is not None
            faiss.write_index(self._index, str(tmp_dir / self.INDEX_FILENAME))

            # documents.jsonl
            docs_path = tmp_dir / self.DOCUMENTS_FILENAME
            with self._open_atomic(docs_path, "wb") as f:
                for d in self.list_documents():
                    f.write((d.model_dump_json() + "\n").encode("utf-8"))
                    f.flush()
                    os.fsync(f.fileno())

            # chunks.jsonl
            chunks_path = tmp_dir / self.CHUNKS_FILENAME
            with self._open_atomic(chunks_path, "wb") as f:
                for c in self._chunks:
                    f.write((c.model_dump_json() + "\n").encode("utf-8"))
                    f.flush()
                    os.fsync(f.fileno())

            # manifest.json
            manifest_path = tmp_dir / self.MANIFEST_FILENAME
            manifest = self._build_manifest()
            manifest["snapshot_id"] = snapshot_id
            with self._open_atomic(manifest_path, "wb") as f:
                payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

            # 临时快照内部一致性校验
            self._validate_snapshot_dir(tmp_dir)

            # 原子重命名
            os.replace(str(tmp_dir), str(final_dir))

            # 切换 CURRENT
            self._write_current_atomic(snapshot_id)

            self._log.info(
                "已保存快照：snapshot_id=%s docs=%d chunks=%d",
                snapshot_id,
                manifest["document_count"],
                manifest["chunk_count"],
            )
        except Exception as exc:
            if tmp_dir.exists():
                try:
                    shutil.rmtree(tmp_dir)
                except OSError:
                    pass
            self._log.error("保存快照失败：%s: %s", type(exc).__name__, exc)
            raise

    def load(self) -> None:
        """从磁盘加载索引与元数据。"""
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._cleanup_tmp_snapshots()

        current_id = self._read_current_snapshot_id()
        if current_id is None:
            self._index = faiss.IndexFlatIP(self.dimensions)
            self._documents.clear()
            self._hash_index.clear()
            self._chunks.clear()
            self._chunk_ids.clear()
            self._vectors = None
            self._log.info("首次启动：初始化空索引。")
            return

        snapshot_dir = self._snapshots_dir / current_id
        if not snapshot_dir.exists():
            raise IndexNotFoundError(
                f"CURRENT 指向的快照目录不存在：{snapshot_dir}"
            )
        self._validate_snapshot_dir(snapshot_dir)
        self._load_snapshot(snapshot_dir)
        self._log.info(
            "已加载快照：snapshot_id=%s docs=%d chunks=%d",
            current_id,
            len(self._documents),
            len(self._chunks),
        )

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _validate_in_memory_state(self) -> None:
        if self._index is None:
            raise VectorStoreError("内部状态异常：index 为空。")
        if self._index.ntotal != len(self._chunks):
            raise IndexCorruptedError(
                f"内存 index.ntotal={self._index.ntotal} 与 chunks={len(self._chunks)} 不一致。"
            )
        if self._index.d != self.dimensions:
            raise IndexIncompatibleError(
                f"内存 index.d={self._index.d} 与配置 dimensions={self.dimensions} 不一致。"
            )
        doc_ids = set(self._documents.keys())
        for c in self._chunks:
            if c.document_id not in doc_ids:
                raise IndexCorruptedError(
                    f"chunk={c.chunk_id} 引用了不存在的 document_id={c.document_id}。"
                )
        if len(self._chunk_ids) != len(self._chunks):
            raise IndexCorruptedError("chunk_id 出现重复。")
        if len(self._hash_index) != len(self._documents):
            raise IndexCorruptedError("file_hash 出现重复或与 documents 不一致。")

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "embedding_provider": self.EMBEDDING_PROVIDER,
            "embedding_model": self._settings.siliconflow_embedding_model,
            "embedding_dimensions": self.dimensions,
            "chunk_size": int(self._settings.chunk_size),
            "chunk_overlap": int(self._settings.chunk_overlap),
            "index_type": self.INDEX_TYPE,
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_id": "TBD",
        }

    def _validate_snapshot_dir(self, snapshot_dir: Path) -> None:
        for name in (
            self.INDEX_FILENAME,
            self.CHUNKS_FILENAME,
            self.DOCUMENTS_FILENAME,
            self.MANIFEST_FILENAME,
        ):
            if not (snapshot_dir / name).exists():
                raise IndexNotFoundError(
                    f"快照目录 {snapshot_dir} 缺少文件 {name}。"
                )
        try:
            manifest = json.loads(
                (snapshot_dir / self.MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise IndexCorruptedError(
                f"manifest.json 解析失败：{type(exc).__name__}。"
            ) from exc
        if not isinstance(manifest, dict):
            raise IndexCorruptedError("manifest 顶层不是 object。")
        for key in (
            "schema_version",
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
            "chunk_size",
            "chunk_overlap",
            "index_type",
            "document_count",
            "chunk_count",
        ):
            if key not in manifest:
                raise IndexCorruptedError(f"manifest 缺少字段 {key}。")

    def _load_snapshot(self, snapshot_dir: Path) -> None:
        manifest = json.loads(
            (snapshot_dir / self.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        self._check_manifest_compatibility(manifest)

        # documents
        documents: dict[str, DocumentInfo] = {}
        hash_index: dict[str, str] = {}
        with (snapshot_dir / self.DOCUMENTS_FILENAME).open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = DocumentInfo.model_validate_json(line)
                except Exception as exc:
                    raise IndexCorruptedError(
                        f"documents.jsonl 第 {lineno} 行解析失败：{type(exc).__name__}。"
                    ) from exc
                if doc.document_id in documents:
                    raise IndexCorruptedError(
                        f"documents.jsonl 中 document_id={doc.document_id} 重复。"
                    )
                if doc.file_hash in hash_index:
                    raise IndexCorruptedError(
                        f"documents.jsonl 中 file_hash={doc.file_hash} 重复。"
                    )
                documents[doc.document_id] = doc
                hash_index[doc.file_hash] = doc.document_id

        # chunks
        chunks: list[DocumentChunk] = []
        chunk_ids: dict[str, int] = {}
        with (snapshot_dir / self.CHUNKS_FILENAME).open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = DocumentChunk.model_validate_json(line)
                except Exception as exc:
                    raise IndexCorruptedError(
                        f"chunks.jsonl 第 {lineno} 行解析失败：{type(exc).__name__}。"
                    ) from exc
                if chunk.chunk_id in chunk_ids:
                    raise IndexCorruptedError(
                        f"chunks.jsonl 中 chunk_id={chunk.chunk_id} 重复。"
                    )
                if chunk.document_id not in documents:
                    raise IndexCorruptedError(
                        f"chunks.jsonl 第 {lineno} 行的 document_id="
                        f"{chunk.document_id} 在 documents 中不存在。"
                    )
                chunk_ids[chunk.chunk_id] = len(chunks)
                chunks.append(chunk)

        # index
        index_path = snapshot_dir / self.INDEX_FILENAME
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise IndexCorruptedError(
                f"faiss.index 读取失败：{type(exc).__name__}。"
            ) from exc

        if index.d != self.dimensions:
            raise IndexIncompatibleError(
                f"index.d={index.d} 与配置 embedding_dimensions={self.dimensions} 不一致。"
            )
        if index.ntotal != len(chunks):
            raise IndexCorruptedError(
                f"index.ntotal={index.ntotal} 与 chunks={len(chunks)} 不一致。"
            )
        if manifest["chunk_count"] != len(chunks):
            raise IndexCorruptedError(
                f"manifest.chunk_count={manifest['chunk_count']} 与 chunks={len(chunks)} 不一致。"
            )
        if manifest["document_count"] != len(documents):
            raise IndexCorruptedError(
                f"manifest.document_count={manifest['document_count']} 与 "
                f"documents={len(documents)} 不一致。"
            )

        self._index = index
        self._documents = documents
        self._hash_index = hash_index
        self._chunks = chunks
        self._chunk_ids = chunk_ids
        # 从 index 重建 _vectors 缓存（用于之后 save 失败回滚 / delete）
        try:
            n = index.ntotal
            if n > 0:
                buf = np.empty((n, index.d), dtype=np.float32)
                for i in range(n):
                    r = index.reconstruct(i)
                    # IndexFlatIP.reconstruct 已返回 numpy.ndarray
                    buf[i] = np.asarray(r, dtype=np.float32)
                self._vectors = buf
            else:
                self._vectors = None
        except Exception:
            self._vectors = None

    def _check_manifest_compatibility(self, manifest: dict[str, Any]) -> None:
        diffs: list[str] = []

        def _cmp(key: str, current: Any, indexed: Any) -> None:
            if current != indexed:
                diffs.append(f"  - {key}: 当前={current!r}, 索引={indexed!r}")

        _cmp("schema_version", self.MANIFEST_SCHEMA_VERSION, manifest.get("schema_version"))
        _cmp("embedding_provider", self.EMBEDDING_PROVIDER, manifest.get("embedding_provider"))
        _cmp(
            "embedding_model",
            self._settings.siliconflow_embedding_model,
            manifest.get("embedding_model"),
        )
        _cmp(
            "embedding_dimensions",
            self.dimensions,
            manifest.get("embedding_dimensions"),
        )
        _cmp("chunk_size", int(self._settings.chunk_size), manifest.get("chunk_size"))
        _cmp("chunk_overlap", int(self._settings.chunk_overlap), manifest.get("chunk_overlap"))
        _cmp("index_type", self.INDEX_TYPE, manifest.get("index_type"))

        if diffs:
            msg = (
                "索引与当前配置不兼容（重建索引前请勿继续使用）：\n"
                + "\n".join(diffs)
                + "\n请在 UI 中点击「重建索引」或删除 storage/indexes 后重启。"
            )
            raise IndexIncompatibleError(msg)

    def _read_current_snapshot_id(self) -> Optional[str]:
        if not self._current_file.exists():
            return None
        content = self._current_file.read_text(encoding="utf-8").strip()
        if not content:
            return None
        if not _SNAPSHOT_ID_RE.match(content):
            raise IndexCorruptedError(f"CURRENT 内容不是合法 snapshot_id：{content!r}")
        return content

    def _write_current_atomic(self, snapshot_id: str) -> None:
        tmp = self._current_file.with_suffix(self._current_file.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(snapshot_id)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self._current_file))
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise

    def _list_snapshots(self) -> list[str]:
        if not self._snapshots_dir.exists():
            return []
        ids = sorted(
            p.name
            for p in self._snapshots_dir.iterdir()
            if p.is_dir() and _SNAPSHOT_ID_RE.match(p.name)
        )
        return ids

    def _cleanup_tmp_snapshots(self) -> int:
        if not self._snapshots_dir.exists():
            return 0
        removed = 0
        for p in list(self._snapshots_dir.iterdir()):
            if not p.is_dir():
                continue
            if p.name.endswith(".tmp"):
                try:
                    shutil.rmtree(p)
                    removed += 1
                except OSError:
                    pass
        return removed

    def _prepare_vectors(
        self,
        vectors: np.ndarray | Iterable[Iterable[float]],
        *,
        expected_count: Optional[int] = None,
    ) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise VectorValidationError(f"向量必须是 2D，得到 ndim={arr.ndim}。")
        if arr.shape[1] != self.dimensions:
            raise VectorValidationError(
                f"向量维度 {arr.shape[1]} 与配置 dimensions={self.dimensions} 不一致。"
            )
        if arr.shape[0] == 0:
            raise VectorValidationError("向量数量为 0，禁止入库。")
        if expected_count is not None and arr.shape[0] != expected_count:
            raise VectorValidationError(
                f"向量数量 {arr.shape[0]} 与 chunks 数量 {expected_count} 不一致。"
            )
        if not np.isfinite(arr).all():
            raise VectorValidationError("向量包含 NaN 或 Inf。")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms == 0):
            raise VectorValidationError("向量中存在全零行（norm=0）。")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        faiss.normalize_L2(arr)
        return arr

    def _prepare_query(self, query_vector: np.ndarray | Iterable[float]) -> np.ndarray:
        arr = np.asarray(query_vector, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != self.dimensions:
                raise VectorValidationError(
                    f"query 维度 {arr.shape[0]} 与配置 dimensions={self.dimensions} 不一致。"
                )
            arr = arr.reshape(1, self.dimensions)
        elif arr.ndim == 2:
            if arr.shape != (1, self.dimensions):
                raise VectorValidationError(
                    f"query 形状 {arr.shape} 与期望 (1, {self.dimensions}) 不一致。"
                )
        else:
            raise VectorValidationError(f"query 必须是 1D 或 2D，得到 ndim={arr.ndim}。")
        if not np.isfinite(arr).all():
            raise VectorValidationError("query 包含 NaN 或 Inf。")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise VectorValidationError("query 为全零向量。")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        faiss.normalize_L2(arr)
        return arr

    @staticmethod
    def _open_atomic(path: Path, mode: str):
        """以普通 ``open`` 打开；外层用 ``os.replace`` 保证原子性。"""
        return open(path, mode)

    def __enter__(self) -> "FaissVectorStore":
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        return None