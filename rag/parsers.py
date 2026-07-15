"""firstRAG 文档解析器。

支持的格式：
- PDF（仅带文本层；扫描 PDF 抛 :class:`UnsupportedScannedPDFError`）
- DOCX（python-docx；两层标题识别 + 段落继承最近标题）
- Markdown（按行扫描；标题与正文分离；记录 line_start/line_end）
- TXT（自动尝试 UTF-8 / UTF-8-SIG / GB18030）

公共入口：
- :func:`parse_document` —— 统一按扩展名分发，返回 :class:`RawDocumentSection` 列表
- :class:`DocumentParseError` 及子类 —— 异常层级
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ======================================================================
# 异常层级
# ======================================================================
class DocumentParseError(Exception):
    """所有解析相关异常的基类。"""


class UnsupportedFileTypeError(DocumentParseError):
    """不支持的文件扩展名。"""


class EmptyDocumentError(DocumentParseError):
    """解析成功但内容为空（PDF 可能是扫描版 / 文件本身无文本）。"""


class UnsupportedScannedPDFError(DocumentParseError):
    """PDF 没有可提取的文本层（疑似扫描版，第一版不支持 OCR）。"""


class TextEncodingError(DocumentParseError):
    """所有候选编码均解码失败。"""


# ======================================================================
# 解析结果结构
# ======================================================================
@dataclass
class RawDocumentSection:
    """解析后的语义单元（一页 / 一个段落 / 一个行号范围等）。

    Attributes:
        content: 文本内容（已经过基本清理）。
        source_name: 原始文件名（仅用于展示，不会用作磁盘文件名）。
        page_number: PDF 页码（从 1 开始），其他类型为 None。
        paragraph_number: DOCX 段落编号（从 1 开始），其他类型为 None。
        heading: 当前 section 所属最近标题（DOCX / Markdown）。
        line_start: 行号范围起点（Markdown / TXT），从 1 开始。
        line_end: 行号范围终点（Markdown / TXT）。
        metadata: 额外元数据字典。
    """

    content: str
    source_name: str
    page_number: Optional[int] = None
    paragraph_number: Optional[int] = None
    heading: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: dict = field(default_factory=dict)


# ======================================================================
# 解析器抽象与实现
# ======================================================================
class DocumentParser(ABC):
    """文档解析器协议。"""

    @abstractmethod
    def parse(self, path: Path) -> list[RawDocumentSection]:
        """解析文件，返回 section 列表。空结果视情况抛 :class:`EmptyDocumentError`
        或 :class:`UnsupportedScannedPDFError`。"""


# ------------------------- PDF -------------------------
class PdfParser(DocumentParser):
    """基于 pypdf 的 PDF 解析器。

    - 按页提取文本；页码从 1 开始。
    - 单页文本为空时跳过该页（不创建 section）。
    - **所有页均为空** → 抛 :class:`UnsupportedScannedPDFError`。
    - 不实现 OCR。
    """

    def parse(self, path: Path) -> list[RawDocumentSection]:
        # 延迟导入：减少非 PDF 测试用例的启动开销
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        sections: list[RawDocumentSection] = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:  # noqa: BLE001
                raise DocumentParseError(
                    f"PDF 第 {idx} 页文本提取失败: {e}"
                ) from e
            cleaned = _clean_text(text)
            if not cleaned:
                continue
            sections.append(
                RawDocumentSection(
                    content=cleaned,
                    source_name=path.name,
                    page_number=idx,
                )
            )
        if not sections:
            raise UnsupportedScannedPDFError(
                f"PDF 未提取到任何文本（可能为扫描 PDF）：{path.name}"
            )
        return sections


# ------------------------- DOCX -------------------------
# 常见中文/数字标题模式（用于 OCR 文档的启发式标题识别）
_HEADING_PATTERN_RE = re.compile(
    r"^("
    r"第[一二三四五六七八九十百千零]+(?:部分|章|节|篇)|"  # 第一部分 / 第一章 / 第一节
    r"第\d+(?:部分|章|节|篇)|"                          # 第1章 / 第2部分
    r"[一二三四五六七八九十]+、|"                       # 一、 二、
    r"（[一二三四五六七八九十]+）|"                     # （一）
    r"\([一二三四五六七八九十]+\)|"                     # (一)
    r"\d+[\.、\)]\s*|"                                 # 1. / 1、 / 1)
    r"[一二三四五六七八九十]+\s*[\.、\)]"
    r").{0,60}$"
)


class DocxParser(DocumentParser):
    """基于 python-docx 的 DOCX 解析器。

    **两层标题识别策略**：

    1. **第一优先级（标准样式）**：Word Heading 1-9 / Title / Subtitle。
    2. **第二优先级（OCR 启发式）**：当文档未使用标准样式时，按以下信号
       保守识别候选标题（**至少满足 2 个条件**）：
       - 文本较短（≤ 30 字符）
       - 段落居中
       - 字体明显大于正文平均（≥ avg + 2pt）
       - 整段加粗（≥ 80% runs 加粗）
       - 匹配 :data:`_HEADING_PATTERN_RE`（"第一章"/"一、"/"1." 等常见模式）

    普通段落继承最近出现的有效标题。
    保留 paragraph_number（DOCX 段落顺序，从 1 开始）。
    对 OCR 噪声进行基本清理（合并空白、合并连续换行）。
    """

    _HEADING_STYLE_RE = re.compile(r"^Heading\s+(\d+)$", re.IGNORECASE)

    # 启发式阈值
    _MAX_HEADING_LEN = 30         # 候选标题最大长度
    _BOLD_RATIO_THRESHOLD = 0.8   # 整段加粗比例
    _SIZE_DELTA_PT = 2.0          # 字号差（候选 ≥ avg + 此值）

    def parse(self, path: Path) -> list[RawDocumentSection]:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        doc = Document(str(path))
        # 1. 先扫描所有段落，计算正文平均字号（排除明显异常的极端值）
        avg_body_size = self._compute_average_body_size(doc)
        # 2. 第二轮扫描：识别标题并构造 sections
        sections: list[RawDocumentSection] = []
        current_heading: Optional[str] = None
        paragraph_number = 0

        for para in doc.paragraphs:
            paragraph_number += 1
            raw_text = para.text or ""
            cleaned = _clean_text(raw_text)
            style_name = (para.style.name or "") if para.style is not None else ""

            is_heading_text, heading_text = self._detect_heading(
                para, cleaned, style_name, avg_body_size
            )
            if is_heading_text:
                current_heading = heading_text
                continue  # 标题自身不作为独立 section

            if not cleaned:
                continue

            sections.append(
                RawDocumentSection(
                    content=cleaned,
                    source_name=path.name,
                    paragraph_number=paragraph_number,
                    heading=current_heading,
                    metadata={"style": style_name} if style_name else {},
                )
            )

        if not sections:
            raise EmptyDocumentError(
                f"DOCX 解析后无有效段落：{path.name}"
            )
        return sections

    # ------------------------------------------------------------------
    # 标题识别
    # ------------------------------------------------------------------
    def _detect_heading(
        self,
        para,
        cleaned: str,
        style_name: str,
        avg_body_size: float,
    ) -> tuple[bool, Optional[str]]:
        """返回 (是否标题, 标题文本)。"""

        # 第一优先级：标准样式
        if self._HEADING_STYLE_RE.match(style_name) or style_name in {"Title", "Subtitle"}:
            return (bool(cleaned), cleaned or None)

        if not cleaned:
            return (False, None)

        # 第二优先级：启发式（保守）
        # 信号收集
        signals: list[str] = []
        text = cleaned.strip()

        # 信号 1：长度
        if len(text) <= self._MAX_HEADING_LEN:
            signals.append("short")

        # 信号 2：居中
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        if para.alignment == WD_ALIGN_PARAGRAPH.CENTER:
            signals.append("centered")

        # 信号 3：字号偏大
        max_size = self._max_run_size(para)
        if max_size and max_size >= avg_body_size + self._SIZE_DELTA_PT:
            signals.append("large_font")

        # 信号 4：整段加粗
        bold_ratio = self._bold_ratio(para)
        if bold_ratio >= self._BOLD_RATIO_THRESHOLD:
            signals.append("bold")

        # 信号 5：匹配常见标题模式
        if _HEADING_PATTERN_RE.match(text):
            signals.append("pattern")

        # 信号 6：不包含句子终止标点（标题一般不出现 "。" "?" "!"）
        if not re.search(r"[。？！!?]", text):
            signals.append("no_terminator")

        # 保守策略：要求至少 2 个强信号（short / large_font / centered / bold / pattern）
        strong_signals = {s for s in signals if s in {"short", "large_font", "centered", "bold", "pattern"}}
        if len(strong_signals) < 2:
            return (False, None)
        # 防止误判药品名称：长度 ≤ 30 且没有 pattern 但只有 large_font → 不算标题
        # （已在 strong_signals 集合限制里得到约束）
        return (True, text)

    @staticmethod
    def _compute_average_body_size(doc) -> float:
        """计算正文平均字号；用于启发式标题识别中的对比基准。

        - 取所有 run.font.size 有效的字号；
        - 去除明显异常（如 < 6pt 或 > 50pt）的极端值；
        - 默认值 10.5pt（小四）。
        """
        sizes: list[float] = []
        for para in doc.paragraphs:
            for run in para.runs:
                if run.font.size is not None:
                    pt = run.font.size.pt
                    if 6.0 <= pt <= 50.0:
                        sizes.append(pt)
        if not sizes:
            return 10.5
        return sum(sizes) / len(sizes)

    @staticmethod
    def _max_run_size(para) -> Optional[float]:
        max_size: Optional[float] = None
        for run in para.runs:
            if run.font.size is None:
                continue
            pt = run.font.size.pt
            if max_size is None or pt > max_size:
                max_size = pt
        return max_size

    @staticmethod
    def _bold_ratio(para) -> float:
        total = 0
        bold = 0
        for run in para.runs:
            text = run.text or ""
            if not text.strip():
                continue
            total += 1
            if run.bold:
                bold += 1
        if total == 0:
            return 0.0
        return bold / total


# ------------------------- Markdown -------------------------
class MarkdownParser(DocumentParser):
    """基于行扫描的 Markdown 解析器。

    - 识别 # / ## / ### ... 开头的标题，记录层级。
    - 连续正文行（去除空行）作为一个 section；line_start / line_end 记录行号范围。
    - 正文继承最近出现的标题（最近一个任意层级标题）。
    """

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def parse(self, path: Path) -> list[RawDocumentSection]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 部分 MD 文件可能含 BOM
            text = path.read_text(encoding="utf-8-sig")

        lines = text.splitlines()
        sections: list[RawDocumentSection] = []
        current_heading: Optional[str] = None

        # 临时缓存一个 section 的起始行
        buf: list[str] = []
        line_start: Optional[int] = None

        def flush(end_line: int) -> None:
            nonlocal buf, line_start
            if buf and line_start is not None:
                content = _clean_text("\n".join(buf))
                if content:
                    sections.append(
                        RawDocumentSection(
                            content=content,
                            source_name=path.name,
                            heading=current_heading,
                            line_start=line_start,
                            line_end=end_line,
                        )
                    )
            buf = []
            line_start = None

        for i, line in enumerate(lines, start=1):
            m = self._HEADING_RE.match(line.strip())
            if m:
                # 标题行：先 flush 当前缓存，再更新 heading
                flush(i - 1)
                current_heading = m.group(2).strip()
                continue

            if not line.strip():
                # 空行：作为一个 section 的自然边界
                flush(i - 1)
                continue

            if line_start is None:
                line_start = i
            buf.append(line)

        # 文件末尾 flush
        flush(len(lines))

        if not sections:
            raise EmptyDocumentError(
                f"Markdown 解析后无有效内容：{path.name}"
            )
        return sections


# ------------------------- TXT -------------------------
class TxtParser(DocumentParser):
    """TXT 解析器。

    自动尝试 UTF-8 / UTF-8-SIG / GB18030；全部失败 → :class:`TextEncodingError`。
    按行扫描，将连续非空行合并为一个 section；保留 line_start / line_end。
    """

    CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030")

    def parse(self, path: Path) -> list[RawDocumentSection]:
        raw = path.read_bytes()
        text: Optional[str] = None
        last_err: Optional[Exception] = None
        for enc in self.CANDIDATE_ENCODINGS:
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError as e:
                last_err = e
        if text is None:
            raise TextEncodingError(
                f"TXT 解码失败：尝试 {list(self.CANDIDATE_ENCODINGS)} 均不匹配。"
                f" 最后错误：{last_err}"
            )

        lines = text.splitlines()
        sections: list[RawDocumentSection] = []
        buf: list[str] = []
        line_start: Optional[int] = None

        def flush(end_line: int) -> None:
            nonlocal buf, line_start
            if buf and line_start is not None:
                content = _clean_text("\n".join(buf))
                if content:
                    sections.append(
                        RawDocumentSection(
                            content=content,
                            source_name=path.name,
                            line_start=line_start,
                            line_end=end_line,
                        )
                    )
            buf = []
            line_start = None

        for i, line in enumerate(lines, start=1):
            if not line.strip():
                flush(i - 1)
                continue
            if line_start is None:
                line_start = i
            buf.append(line)
        flush(len(lines))

        if not sections:
            raise EmptyDocumentError(
                f"TXT 解析后无有效行：{path.name}"
            )
        return sections


# ======================================================================
# 内部工具
# ======================================================================
_MULTI_SPACE = re.compile(r"[ \t　]+")  # 含全角空格
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """通用文本清理：
    - 合并连续空格 / 制表符 / 全角空格为单空格
    - 合并 3+ 连续换行为 2 个
    - 去掉首尾空白
    - 保留单字符内容（不主动删除）
    """
    if not text:
        return ""
    s = _MULTI_SPACE.sub(" ", text)
    s = _MULTI_NEWLINE.sub("\n\n", s)
    return s.strip()


# ======================================================================
# 统一入口
# ======================================================================
_PARSERS: dict[str, DocumentParser] = {
    ".pdf": PdfParser(),
    ".docx": DocxParser(),
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".txt": TxtParser(),
}


def parse_document(path: Path) -> list[RawDocumentSection]:
    """按扩展名分发到对应解析器，返回 :class:`RawDocumentSection` 列表。

    Raises:
        UnsupportedFileTypeError: 文件扩展名不在白名单中。
        FileNotFoundError: 文件不存在。
        DocumentParseError: 解析失败（子类含具体原因）。
        EmptyDocumentError: 解析成功但无内容。
        UnsupportedScannedPDFError: PDF 无文本层。
        TextEncodingError: TXT 解码失败。
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型 '{ext}'；支持的扩展名: {sorted(_PARSERS)}"
        )
    return parser.parse(path)


def supported_extensions() -> list[str]:
    """返回当前支持的扩展名列表（仅用于 UI 提示）。"""
    return sorted(_PARSERS.keys())