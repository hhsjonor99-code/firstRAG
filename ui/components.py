"""UI 纯函数辅助。

本模块**不**调用任何 ``streamlit.*`` 渲染函数；只提供可单测的纯函数。
渲染函数与 :mod:`streamlit` API 的耦合放到 :mod:`app`。
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

from rag.knowledge_base_service import (
    DuplicateDocumentError as KSDuplicate,
    EmptyUploadError,
    InvalidUploadError,
    KnowledgeBaseRollbackError,
    KnowledgeBaseServiceError,
    UnsupportedUploadTypeError,
    UploadTooLargeError,
)
from rag.llm_client import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from rag.models import (
    DocumentChunk,
    RetrievedChunk,
)
from rag.parsers import (
    EmptyDocumentError,
    UnsupportedScannedPDFError,
)
from rag.prompt_builder import NO_EVIDENCE_REPLY


# ---------------------------------------------------------------------------
# 1. 引用位置格式化（沿用 PromptBuilder 规则）
# ---------------------------------------------------------------------------
def format_chunk_location(chunk: DocumentChunk) -> str:
    """把 ``DocumentChunk`` 渲染为简短位置描述。

    规则（与 :func:`rag.prompt_builder.PromptBuilder._format_location` 一致）：

    - DOCX 普通段落：第 N～M 段
    - DOCX 表格：第 N 个表格，第 X～Y 行
    - DOCX 文本框：文本块 N
    - PDF：第 N 页
    - TXT / Markdown：第 X～Y 行
    """
    bt = (chunk.block_type or "").lower().strip()

    # 表格
    if bt == "table":
        parts: list[str] = []
        if chunk.table_index is not None:
            parts.append(f"第 {chunk.table_index + 1} 个表格")
        else:
            parts.append("表格")
        if chunk.row_start is not None:
            rs = chunk.row_start + 1
            if chunk.row_end is not None and chunk.row_end != chunk.row_start:
                parts.append(f"第 {rs}～{chunk.row_end + 1} 行")
            else:
                parts.append(f"第 {rs} 行")
        return "，".join(parts)

    # 文本框
    if bt == "textbox":
        if chunk.block_indices:
            return f"文本块 {chunk.block_indices[0] + 1}"
        return "文本块"

    # PDF
    if (
        chunk.page_number is not None
        and chunk.line_start is None
        and chunk.paragraph_start is None
        and chunk.paragraph_number is None
    ):
        return f"第 {chunk.page_number} 页"

    # TXT / Markdown
    if chunk.line_start is not None:
        if chunk.line_end is not None and chunk.line_end != chunk.line_start:
            return f"第 {chunk.line_start}～{chunk.line_end} 行"
        return f"第 {chunk.line_start} 行"

    # DOCX 段落
    ps = chunk.paragraph_start if chunk.paragraph_start is not None else chunk.paragraph_number
    if ps is not None:
        pe = chunk.paragraph_end
        if pe is not None and pe != ps:
            return f"第 {ps}～{pe} 段"
        return f"第 {ps} 段"

    return ""


# ---------------------------------------------------------------------------
# 2. 文件大小 / 时间格式化
# ---------------------------------------------------------------------------
def format_file_size(num_bytes: int) -> str:
    """把字节数渲染为 ``1.23 MB`` 风格。"""
    if num_bytes is None or num_bytes < 0:
        return "—"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def format_created_at(dt: Optional[datetime]) -> str:
    """格式化入库时间为 ``YYYY-MM-DD HH:MM``。"""
    if dt is None:
        return "—"
    try:
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return str(dt)


# ---------------------------------------------------------------------------
# 3. 安全错误映射
# ---------------------------------------------------------------------------
# 出错信息中文模板；不会拼接 Key、Prompt、文档正文
_ERROR_MESSAGES: list[tuple[type, str]] = [
    (LLMConfigurationError, "LLM 配置错误：缺少 API Key 或参数错误。"),
    (LLMAuthenticationError, "LLM 鉴权失败：请检查 MINIMAX_API_KEY。"),
    (LLMRateLimitError, "LLM 触发限流，请稍后重试。"),
    (LLMTimeoutError, "LLM 请求超时，请稍后重试。"),
    (LLMConnectionError, "LLM 网络连接错误，请稍后重试。"),
    (LLMServerError, "LLM 服务端错误，请稍后重试。"),
    (LLMError, "回答生成失败，请稍后重试。"),
    (EmptyUploadError, "上传内容为空。"),
    (UploadTooLargeError, "文件超过大小限制。"),
    (UnsupportedUploadTypeError, "不支持的文件格式，仅支持 PDF / DOCX / TXT / Markdown。"),
    (InvalidUploadError, "文件不合法（路径或名称）。"),
    (KSDuplicate, "该文件已存在于知识库中。"),
    (KnowledgeBaseRollbackError, "入库失败，且回滚未完成；请重新上传。"),
    (KnowledgeBaseServiceError, "知识库服务错误，请稍后重试。"),
    (EmptyDocumentError, "文档解析后无内容。"),
    (UnsupportedScannedPDFError, "该 PDF 可能是扫描件，当前版本暂不支持 OCR。"),
]


def safe_error_message(exc: BaseException) -> str:
    """把异常归类为安全的中文消息（不包含 Key / Prompt / 文档内容）。"""
    for exc_type, msg in _ERROR_MESSAGES:
        if isinstance(exc, exc_type):
            return msg
    # 兜底：只输出错误类型名
    return f"操作失败：{type(exc).__name__}"


# ---------------------------------------------------------------------------
# 4. 上传结果汇总
# ---------------------------------------------------------------------------
def summarize_ingest_results(
    results: list[dict[str, Any]],
) -> dict[str, int]:
    """汇总一批入库结果。

    :param results: ``[{"file_name": str, "ok": bool, "error": str|None, "info": DocumentInfo|None}]``
    :returns: 字典，含 ``total`` / ``success`` / ``failed``。
    """
    total = len(results)
    success = sum(1 for r in results if r.get("ok"))
    failed = total - success
    return {"total": total, "success": success, "failed": failed}


# ---------------------------------------------------------------------------
# 5. 状态指示
# ---------------------------------------------------------------------------
def api_key_status(has_key: bool) -> str:
    """``已配置 / 未配置``。"""
    return "已配置" if has_key else "未配置"


# ---------------------------------------------------------------------------
# 6. 内容脱敏 / 校验
# ---------------------------------------------------------------------------
_FORBIDDEN_IN_USER_TEXT = ("<think>", "</think>", "<analysis>", "</analysis>")


def detect_internal_leakage(text: str) -> list[str]:
    """检测用户可见文本中是否含内部思考 / 标签；返回命中的标签列表。"""
    if not isinstance(text, str):
        return []
    return [t for t in _FORBIDDEN_IN_USER_TEXT if t in text]


# ---------------------------------------------------------------------------
# 7. 引用面板辅助
# ---------------------------------------------------------------------------
def format_score(score: float) -> str:
    """统一相似度格式。"""
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return "—"


def build_citation_view(
    citation: RetrievedChunk,
) -> dict[str, Any]:
    """构造引用面板的视图数据（纯字典，不依赖 streamlit）。"""
    chunk = citation.chunk
    return {
        "citation_id": citation.citation_id,
        "source_name": chunk.source_name,
        "location": format_chunk_location(chunk),
        "heading": chunk.heading or "",
        "score": format_score(citation.score),
        "content": chunk.content or "",
    }


def build_citation_panel(citations: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """构造完整引用面板数据列表。"""
    return [build_citation_view(c) for c in (citations or [])]


# ---------------------------------------------------------------------------
# 8. 拒答 / 引用提示
# ---------------------------------------------------------------------------
def is_refusal(content: str) -> bool:
    """判断是否为固定拒答语。"""
    if not isinstance(content, str):
        return False
    return content.strip() == NO_EVIDENCE_REPLY


def chat_role_label(role: str) -> str:
    """``user / assistant / system`` → ``用户 / 助手 / 系统``。"""
    return {
        "user": "用户",
        "assistant": "助手",
        "system": "系统",
    }.get(role, role)


# ---------------------------------------------------------------------------
# 9. 引用编号格式化（与展示面板一致）
# ---------------------------------------------------------------------------
_CITATION_TAG_RE = re.compile(r"\[S\d+\]")


def extract_citation_ids(text: str) -> list[str]:
    """从答案中提取合法引用编号（按出现顺序去重），形如 ``["S1", "S2"]``。"""
    if not isinstance(text, str):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _CITATION_TAG_RE.finditer(text):
        tag = m.group(0)  # [S1]
        cid = tag[1:-1]  # S1
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out
