"""UI 层的 RAG 服务工厂。

使用 ``st.cache_resource`` 缓存：

- :class:`Settings`（lru_cache 也行，但为统一使用 streamlit 缓存）
- :class:`FaissVectorStore`（按 index_dir 路径缓存）
- :class:`KnowledgeBaseService`
- :class:`Retriever`
- :class:`PromptBuilder`
- :class:`MiniMaxLLMClient`
- :class:`ChatService`
- :class:`SiliconFlowEmbeddingProvider`

安全约束：

- **不**缓存：API Key（不写入 ``session_state``、不写入 st 输出）；
- 每次脚本执行时，``has_sf_key`` / ``has_mm_key`` 通过 ``Settings.has_*_key()`` 重新读取；
- 文档增删后：KnowledgeBaseService 在 ``add_document`` / ``delete_document`` 内部自动
  持久化到磁盘快照；UI 重新拉取列表即可，无需重建服务实例。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import streamlit as st

from config.settings import Settings, get_settings

from rag.chat_service import ChatService
from rag.embedding_provider import EmbeddingProvider
from rag.knowledge_base_service import KnowledgeBaseService
from rag.llm_client import MiniMaxLLMClient
from rag.prompt_builder import PromptBuilder
from rag.retriever import Retriever
from rag.siliconflow_embeddings import SiliconFlowEmbeddingProvider
from rag.vector_store import FaissVectorStore


# ---------------------------------------------------------------------------
# 项目根目录（默认 index_dir / upload_dir 来自 storage/）
# ---------------------------------------------------------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_index_dir() -> Path:
    return project_root() / "storage" / "indexes"


def default_upload_dir() -> Path:
    return project_root() / "storage" / "uploads"


# ---------------------------------------------------------------------------
# Settings（不缓存——每次读 .env 即可）
# ---------------------------------------------------------------------------
def load_settings() -> Settings:
    return get_settings()


def has_siliconflow_key() -> bool:
    return load_settings().has_siliconflow_key()


def has_minimax_key() -> bool:
    return load_settings().has_minimax_key()


# ---------------------------------------------------------------------------
# 缓存的资源
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_provider(_settings_id: int) -> Optional[EmbeddingProvider]:
    """获取 SiliconFlow Embedding 提供商。

    若 ``SILICONFLOW_API_KEY`` 未配置，返回 ``None``（UI 应禁止入库）。
    ``_settings_id`` 仅用于让 cache 在 Settings 变化时刷新。
    """
    settings = load_settings()
    if not settings.has_siliconflow_key():
        return None
    return SiliconFlowEmbeddingProvider(settings=settings)


@st.cache_resource(show_spinner=False)
def get_vector_store(_index_dir_str: str, _settings_id: int) -> FaissVectorStore:
    """获取 FaissVectorStore 并 load()。

    按 ``index_dir`` 字符串缓存；同一目录多次访问共享同一实例。
    """
    settings = load_settings()
    index_dir = Path(_index_dir_str)
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FaissVectorStore(settings=settings, index_dir=index_dir)
    store.load()
    return store


@st.cache_resource(show_spinner=False)
def get_knowledge_base_service(
    _index_dir_str: str,
    _upload_dir_str: str,
    _settings_id: int,
) -> KnowledgeBaseService:
    """获取 KnowledgeBaseService。"""
    settings = load_settings()
    embedder = get_embedding_provider(_settings_id)
    if embedder is None:
        raise RuntimeError("SiliconFlow Embedding 不可用：缺少 API Key。")
    store = get_vector_store(_index_dir_str, _settings_id)
    upload_dir = Path(_upload_dir_str)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return KnowledgeBaseService(
        embedding_provider=embedder,
        vector_store=store,
        settings=settings,
        upload_dir=upload_dir,
    )


@st.cache_resource(show_spinner=False)
def get_retriever(_index_dir_str: str, _settings_id: int) -> Retriever:
    """获取 Retriever。"""
    settings = load_settings()
    embedder = get_embedding_provider(_settings_id)
    if embedder is None:
        raise RuntimeError("Retriever 不可用：缺少 Embedding。")
    store = get_vector_store(_index_dir_str, _settings_id)
    return Retriever(embedder, store, settings)


@st.cache_resource(show_spinner=False)
def get_prompt_builder(_settings_id: int) -> PromptBuilder:
    return PromptBuilder(load_settings())


@st.cache_resource(show_spinner=False)
def get_llm_client(_settings_id: int) -> Optional[MiniMaxLLMClient]:
    """获取 LLM 客户端；缺 Key 时返回 None。"""
    settings = load_settings()
    if not settings.has_minimax_key():
        return None
    return MiniMaxLLMClient(settings=settings)


@st.cache_resource(show_spinner=False)
def get_chat_service(
    _index_dir_str: str,
    _settings_id: int,
) -> Optional[ChatService]:
    """获取 ChatService；缺关键组件时返回 None。"""
    settings = load_settings()
    if not settings.has_minimax_key():
        return None
    if not settings.has_siliconflow_key():
        return None
    retriever = get_retriever(_index_dir_str, _settings_id)
    pb = get_prompt_builder(_settings_id)
    llm = get_llm_client(_settings_id)
    if llm is None:
        return None
    return ChatService(retriever, pb, llm, settings)


# ---------------------------------------------------------------------------
# 工具：给 cache key 提供一个稳定 hash
# ---------------------------------------------------------------------------
def _stable_payload_signature(payload_data: dict[str, object]) -> int:
    """对配置数据进行稳定序列化并生成 cache key。"""
    payload = json.dumps(
        payload_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def settings_signature() -> int:
    """生成 Settings 内容的稳定 hash（不包含 API Key 内容）。"""
    settings = load_settings()
    payload_data = {
        "siliconflow_base_url": settings.siliconflow_base_url,
        "siliconflow_embedding_model": settings.siliconflow_embedding_model,
        "siliconflow_embedding_dimensions": settings.siliconflow_embedding_dimensions,
        "siliconflow_embedding_batch_size": settings.siliconflow_embedding_batch_size,
        "siliconflow_timeout": settings.siliconflow_timeout,
        "minimax_base_url": settings.minimax_base_url,
        "minimax_model": settings.minimax_model,
        "minimax_timeout": settings.minimax_timeout,
        "minimax_max_retries": settings.minimax_max_retries,
        "minimax_temperature": settings.minimax_temperature,
        "minimax_max_tokens": settings.minimax_max_tokens,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_min_score": settings.retrieval_min_score,
        "max_upload_mb": settings.max_upload_mb,
        "max_history_turns": settings.max_history_turns,
        "index_dir": str(default_index_dir()),
        "upload_dir": str(default_upload_dir()),
        "has_siliconflow_key": settings.has_siliconflow_key(),
        "has_minimax_key": settings.has_minimax_key(),
    }
    return _stable_payload_signature(payload_data)


# ---------------------------------------------------------------------------
# 便捷聚合
# ---------------------------------------------------------------------------
def list_documents_safe() -> list:
    """列出已入库文档（容错）。"""
    try:
        idx_dir = str(default_index_dir())
        sid = settings_signature()
        store = get_vector_store(idx_dir, sid)
        return list(store.list_documents())
    except Exception:  # noqa: BLE001
        return []


def knowledge_base_stats_safe() -> dict:
    """获取安全统计信息（容错）。"""
    sid = settings_signature()
    idx_dir = str(default_index_dir())
    embedder = get_embedding_provider(sid)
    stats = {
        "embedding_model": embedder.model_name if embedder else "未配置",
        "embedding_dim": embedder.dimensions if embedder else 0,
    }
    try:
        store = get_vector_store(idx_dir, sid)
        stats["document_count"] = store.document_count
        stats["chunk_count"] = store.chunk_count
        stats["index_loaded"] = store.is_loaded
    except Exception:  # noqa: BLE001
        stats["document_count"] = 0
        stats["chunk_count"] = 0
        stats["index_loaded"] = False
    return stats
