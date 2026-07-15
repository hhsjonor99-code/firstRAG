"""firstRAG 业务数据模型（Pydantic v2）。

定义：
- :class:`DocumentInfo`：文档元数据
- :class:`DocumentChunk`：文档片段
- :class:`RetrievedChunk`：检索结果（带引用编号与相似度分数）
- :class:`ChatMessage`：会话消息

这些模型仅描述数据结构与序列化；不涉及业务逻辑。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentInfo(BaseModel):
    """文档元数据。"""

    model_config = ConfigDict(extra="ignore")

    document_id: str = Field(..., description="内部 UUID 标识")
    file_name: str = Field(..., description="磁盘上存储的文件名（document_id + ext）")
    original_file_name: str = Field(..., description="用户上传时的原始文件名")
    file_type: str = Field(..., description="文件类型：pdf / docx / txt / md")
    file_hash: str = Field(..., description="SHA256 哈希")
    file_size: int = Field(..., ge=0, description="字节数")
    created_at: datetime = Field(..., description="入库时间")
    chunk_count: int = Field(0, ge=0, description="切分后的片段数")


class DocumentChunk(BaseModel):
    """文档片段。"""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(..., description="片段唯一 ID")
    document_id: str = Field(..., description="所属文档 ID")
    content: str = Field(..., description="片段文本内容")
    source_name: str = Field(..., description="原始文件名（仅用于展示）")
    page_number: Optional[int] = Field(None, description="PDF 页码；其他类型可为 None")
    paragraph_number: Optional[int] = Field(
        None,
        description="DOCX 起始段落编号（兼容旧模型；新代码请同时读 paragraph_start）",
    )
    paragraph_start: Optional[int] = Field(
        None, description="DOCX 起始段落编号；与 paragraph_number 等价"
    )
    paragraph_end: Optional[int] = Field(
        None, description="DOCX 结束段落编号（聚合多段时填）"
    )
    paragraph_numbers: list[int] = Field(
        default_factory=list, description="聚合的所有段落编号（DOCX）"
    )
    heading: Optional[str] = Field(None, description="标题（DOCX / MD）")
    line_start: Optional[int] = Field(None, description="起始行号（MD / TXT）")
    line_end: Optional[int] = Field(None, description="结束行号（MD / TXT）")
    chunk_index: int = Field(..., ge=0, description="在所属文档中的片段序号")
    metadata: dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class RetrievedChunk(BaseModel):
    """检索结果。引用编号由程序生成，禁止依赖 LLM 自报。"""

    model_config = ConfigDict(extra="ignore")

    chunk: DocumentChunk
    score: float = Field(..., description="相似度分数")
    citation_id: str = Field(..., description="程序生成的引用编号，如 S1")


class ChatMessage(BaseModel):
    """会话消息。"""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., description="user / assistant / system")
    content: str = Field(..., description="消息文本")
    citations: list[RetrievedChunk] = Field(
        default_factory=list, description="assistant 消息附带的引用"
    )
    created_at: datetime = Field(..., description="消息创建时间")