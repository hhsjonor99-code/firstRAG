"""firstRAG 全局配置。

设计原则：
- API Key 允许为 ``None``；``Settings()`` 构造不会因为缺 Key 而抛错。
- 仅在实际使用某个 API（Embedding 或 LLM）时，通过 :meth:`require_siliconflow_key` /
  :meth:`require_minimax_key` 校验，错误消息明确指出缺失的环境变量名。
- 没有 Key 时，本地文档解析、分块、索引结构等仍可使用。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingAPIKeyError(RuntimeError):
    """缺少 API Key 时抛出。错误消息必须明确指出缺失的环境变量名。"""


class Settings(BaseSettings):
    """全局配置。"""

    # ----- SiliconFlow Embedding -----
    siliconflow_api_key: Optional[str] = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    siliconflow_embedding_dimensions: int = 1024
    siliconflow_embedding_batch_size: int = 16
    siliconflow_timeout: int = 60

    # ----- MiniMax LLM -----
    minimax_api_key: Optional[str] = None
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    minimax_model: str = "MiniMax-M3"
    minimax_timeout: float = 60.0
    minimax_max_retries: int = 2
    minimax_temperature: float = 0.2
    minimax_max_tokens: Optional[int] = None

    # ----- 切分 / 检索 -----
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 5
    # 检索最低分数阈值；None 表示不过滤（不同 Embedding 模型分数分布不同）
    retrieval_min_score: Optional[float] = None

    # ----- 上传 / 会话 -----
    max_upload_mb: int = 20
    max_history_turns: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # 延迟校验：实际使用 API 时才检查 Key
    # ------------------------------------------------------------------
    def require_siliconflow_key(self) -> str:
        """获取 SiliconFlow API Key；缺失时抛 :class:`MissingAPIKeyError`。

        错误消息必须明确指出环境变量名 ``SILICONFLOW_API_KEY``。
        """
        key = (self.siliconflow_api_key or "").strip()
        if not key:
            raise MissingAPIKeyError(
                "缺少环境变量 SILICONFLOW_API_KEY；"
                "文档解析、分块、本地索引仍可使用，但 Embedding 与检索需要此 Key。"
            )
        return key

    def require_minimax_key(self) -> str:
        """获取 MiniMax API Key；缺失时抛 :class:`MissingAPIKeyError`。

        错误消息必须明确指出环境变量名 ``MINIMAX_API_KEY``。
        """
        key = (self.minimax_api_key or "").strip()
        if not key:
            raise MissingAPIKeyError(
                "缺少环境变量 MINIMAX_API_KEY；"
                "文档解析、分块、本地索引仍可使用，但 LLM 调用需要此 Key。"
            )
        return key

    # ------------------------------------------------------------------
    # 状态查询（不抛错），供 check_env / UI 显示
    # ------------------------------------------------------------------
    def has_siliconflow_key(self) -> bool:
        return bool((self.siliconflow_api_key or "").strip())

    def has_minimax_key(self) -> bool:
        return bool((self.minimax_api_key or "").strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取单例 :class:`Settings` 实例。

    使用 :func:`functools.lru_cache` 在同一进程内只构造一次。
    ``Settings()`` 自身不会因为缺 Key 而抛错。
    """
    return Settings()