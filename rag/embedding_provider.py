"""Embedding 抽象接口。

业务模块（retriever / chat_service / UI）只能依赖本接口，不能直接调用具体
实现（如 :class:`SiliconFlowEmbeddingProvider`）。这样未来可以无痛替换为
其它 Embedding 提供商（OpenAI / HuggingFace / 本地 sentence-transformers
等），而无需改动上层调用代码。

设计要点：

- 使用 :class:`typing.Protocol` 而不是 ABC：只要对象实现所需方法即可，
  无需显式继承，方便第三方适配。
- 输入文本统一为 ``list[str]``；返回 :class:`numpy.ndarray` 二维数组
  ``(N, dimensions)``，``dtype=np.float32``。
- ``embed_query`` 复用 ``embed_documents``，但返回 ``shape=(1, dimensions)``
  的二维数组（保持项目统一约定：始终返回二维）。
- 空输入必须抛 :class:`EmbeddingError`；不允许返回空数组。
- ``model_name`` / ``dimensions`` 必须与实际请求结果一致，用于 manifest
  校验与索引维度比对。
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np


class EmbeddingError(RuntimeError):
    """Embedding 调用相关错误的基类。"""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding 提供商抽象接口。

    实现类必须提供以下属性与方法。
    """

    # ---- 元信息 ----
    model_name: str
    dimensions: int

    # ---- 文档向量化 ----
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """将多条文本批量编码为向量。

        :param texts: 待编码文本列表。**禁止为空**；空列表必须抛
            :class:`EmbeddingError`。
        :returns: ``shape == (len(texts), self.dimensions)`` 的
            ``dtype == np.float32`` 二维数组。
        :raises EmbeddingError: 输入为空、维度错误、返回数据缺失、
            包含 NaN/Inf 或其它解析问题。
        """

    # ---- 查询向量化 ----
    def embed_query(self, text: str) -> np.ndarray:
        """将单条查询文本编码为向量。

        实现应复用 :meth:`embed_documents`，返回 ``shape == (1, dimensions)``
        的二维数组（与项目约定一致）。
        """


def _empty_input_error() -> EmbeddingError:
    return EmbeddingError("Embedding 输入文本列表为空，至少需要 1 条文本。")


def validate_non_empty_texts(texts: Iterable[str]) -> list[str]:
    """工具函数：校验输入文本列表非空；空则抛 :class:`EmbeddingError`."""
    materialized = list(texts) if not isinstance(texts, list) else texts
    if not materialized:
        raise _empty_input_error()
    return materialized