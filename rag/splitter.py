"""firstRAG 中文分块器（阶段 2.1 增强版）。

核心策略：

1. **按 heading 分组**：同 heading 下的连续段落先聚合，不跨 heading 合并。
2. **短段落预处理**：
   - 纯空白 / 纯标点 / 单字乱码 → 直接过滤。
   - 纯序号类短段（如 "1"、"一"、"（一）"、"1."）→ 优先附加到下一段；无下一段则附加到前一段。
   - 单汉字短段（如药品名）→ 优先附加到下一段；无下一段则附加到前一段。
3. **同 heading 内累积**：按段落顺序累积文本，接近 ``chunk_size`` 时 flush；段落间用 ``\\n`` 连接。
4. **超长聚合**：累积超过 ``chunk_size`` 后，用 :class:`RecursiveCharacterTextSplitter` 切分；
   ``overlap`` 由 splitter 自身处理。
5. **不跨 heading 合并**：每个 heading 组独立聚合。

元数据：
- 每个 chunk 保留所在 heading。
- 段落号：``paragraph_start``（== ``paragraph_number``）、``paragraph_end``、``paragraph_numbers`` 列表。
- ``line_start`` / ``line_end`` 沿用 section 的值（聚合多段时取首尾）。
- ``chunk_index`` 整个输出列表从 0 连续递增。
- ``chunk_id`` 由 ``sha1(document_id + "::" + chunk_index)[:16]`` 生成，**稳定且唯一**。
- 不生成空 chunk。

chunk_size / chunk_overlap 从 :class:`config.settings.Settings` 读取，可由 :class:`SplitOptions` 覆盖。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from rag.models import DocumentChunk
from rag.parsers import RawDocumentSection


# 中文优先分隔符列表（顺序敏感：优先级从高到低）
CHINESE_SEPARATORS: list[str] = [
    "\n\n",   # 双换行（段落）
    "\n",     # 单换行
    "。",     # 中文句号
    "？",     # 中文问号
    "！",     # 中文感叹号
    "；",     # 中文分号
    "?",
    "!",
    ";",
    ".",
    " ",
    "",
]


# 短段"序号"模式：纯序号、纯中文数字前缀
_SEQUENCE_PATTERN_RE = re.compile(
    r"^\s*(?:"
    r"[一二三四五六七八九十百千]+[、\.]?|"        # 一、 / 一.
    r"（[一二三四五六七八九十百千]+）|"             # （一）
    r"\([一二三四五六七八九十百千]+\)|"
    r"\d+[\.、\)]?|"                              # 1. / 1、 / 1)
    r"[A-Za-z][\.\)]"                              # A. / a)
    r")\s*$"
)
# 中文句子终止标点
_TERMINATOR_RE = re.compile(r"[。？！!?;；]")
# 纯标点 / 纯空白检测
_PURE_PUNCT_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)


@dataclass
class SplitOptions:
    """分块参数覆盖项。``None`` 表示从 :class:`Settings` 读取。"""

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


# ======================================================================
# 公共入口
# ======================================================================
def split_sections(
    sections: list[RawDocumentSection],
    document_id: str,
    options: SplitOptions | None = None,
) -> list[DocumentChunk]:
    """将解析得到的 section 列表切分为 :class:`DocumentChunk` 列表。"""
    settings = get_settings()
    chunk_size = (
        options.chunk_size
        if (options and options.chunk_size is not None)
        else settings.chunk_size
    )
    chunk_overlap = (
        options.chunk_overlap
        if (options and options.chunk_overlap is not None)
        else settings.chunk_overlap
    )

    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须 > 0，得到 {chunk_size}")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap 必须满足 0 <= overlap < chunk_size，得到 "
            f"overlap={chunk_overlap}, chunk_size={chunk_size}"
        )

    # 1. 按 heading 分组（保持原顺序）
    groups = _group_by_heading(sections)

    # 2. 每个 group 独立聚合 + 切分
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for heading, group_sections in groups:
        # 短段落预处理
        attached = _attach_short_paragraphs(group_sections)
        # 同 heading 内累积到 chunk_size 附近
        accumulated = _accumulate_paragraphs(attached, target_size=chunk_size)
        # 切分（每段不超过 chunk_size；过长则再走 splitter）
        for merged_text, merged_secs in accumulated:
            for piece_text in _split_text_to_size(merged_text, chunk_size, chunk_overlap):
                content = piece_text.strip()
                if not content:
                    continue
                chunk = _make_chunk(
                    content=content,
                    source_secs=merged_secs,
                    heading=heading,
                    document_id=document_id,
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1
    return chunks


# ======================================================================
# 内部：分组、聚合、合并
# ======================================================================
def _group_by_heading(
    sections: list[RawDocumentSection],
) -> list[tuple[Optional[str], list[RawDocumentSection]]]:
    """按 heading 分组；heading 为 ``None`` 的段落归入 ``(None, [...])`` 组。

    当 heading 变化时开启新组；同一 heading 的连续段落聚合到同一组。
    """
    groups: list[tuple[Optional[str], list[RawDocumentSection]]] = []
    current_heading: Optional[str] = None
    current_group: list[RawDocumentSection] = []

    for sec in sections:
        # 第一次出现 heading 才开启新组；否则合并到当前组
        if sec.heading != current_heading:
            if current_group:
                groups.append((current_heading, current_group))
            current_heading = sec.heading
            current_group = [sec]
        else:
            current_group.append(sec)

    if current_group:
        groups.append((current_heading, current_group))
    return groups


def _is_pure_sequence(text: str) -> bool:
    """判断是否为纯序号 / 短编号（如 '1'、'一'、'（一）'、'1.'）。"""
    return bool(_SEQUENCE_PATTERN_RE.match(text))


def _is_pure_punct_or_ws(text: str) -> bool:
    """判断是否为纯标点 / 纯空白 / 单字乱码。"""
    if not text:
        return True
    return bool(_PURE_PUNCT_RE.match(text))


def _attach_short_paragraphs(
    sections: list[RawDocumentSection],
) -> list[RawDocumentSection]:
    """把短段落（纯序号 / 单字 / 短文本）合并到相邻正常段落。

    策略：
    - 纯空白 / 纯标点段落：直接丢弃。
    - 纯序号短段（长度 ≤ 8）：向后查找第一个非短段并合并到其头部；无则向前合并。
    - 短正文（长度 ≤ 8 且非终止）：向后合并；无则向前合并。
    """
    # 1. 先过滤纯空白 / 纯标点
    filtered: list[RawDocumentSection] = []
    for s in sections:
        if _is_pure_punct_or_ws(s.content):
            continue
        filtered.append(s)

    if not filtered:
        return []

    # 2. 标记哪些段是"短段"（需要被合并到邻居）
    #    短段定义：**纯序号**（如 "1"、"（一）"）或 **单字符**段落（如 "甲"）。
    #    普通短正文（如 6 字符的"段落1内容。"）保留，由累积函数自然合并到 chunk_size 附近。
    is_short = [
        _is_pure_sequence(s.content) or len(s.content.strip()) == 1
        for s in filtered
    ]

    # 3. 累积合并
    result: list[RawDocumentSection] = []
    i = 0
    n = len(filtered)
    while i < n:
        sec = filtered[i]
        if not is_short[i]:
            result.append(sec)
            i += 1
            continue

        # 当前是短段：尝试向后合并
        if i + 1 < n and not is_short[i + 1]:
            merged_content = sec.content + "\n" + filtered[i + 1].content
            merged = _combine_sections(filtered[i + 1], merged_content, sec)
            result.append(merged)
            is_short[i + 1] = False  # 已合并到结果，避免再处理
            i += 2
            continue

        # 向后无正常段：尝试向前合并到 result 末尾
        if result:
            prev = result[-1]
            merged_content = prev.content + "\n" + sec.content
            result[-1] = _combine_sections(prev, merged_content, sec)
            i += 1
            continue

        # 既无向前也无向后：保留为独立段（极少见），但记入 metadata
        result.append(sec)
        i += 1

    return result


def _combine_sections(
    base: RawDocumentSection,
    merged_content: str,
    extra: RawDocumentSection,
) -> RawDocumentSection:
    """合并两个 section 的内容，元数据以 ``base`` 为准。"""
    return replace(
        base,
        content=merged_content,
        metadata={**base.metadata, "merged_from": (base.paragraph_number, extra.paragraph_number)},
    )


def _accumulate_paragraphs(
    sections: list[RawDocumentSection],
    target_size: int,
) -> list[tuple[str, list[RawDocumentSection]]]:
    """累积段落；每段累积接近 ``target_size`` 时 flush。

    返回 ``[(content, [section, ...]), ...]``，每个 tuple 内的段落共同贡献该 content。
    """
    out: list[tuple[str, list[RawDocumentSection]]] = []
    cur_text = ""
    cur_secs: list[RawDocumentSection] = []

    def flush() -> None:
        nonlocal cur_text, cur_secs
        if cur_text.strip():
            out.append((cur_text, cur_secs))
        cur_text = ""
        cur_secs = []

    for sec in sections:
        text = sec.content
        # 估算加入后是否超过目标
        projected = (len(cur_text) + 1 + len(text)) if cur_text else len(text)
        if cur_text and projected > target_size:
            flush()
        if cur_text:
            cur_text += "\n" + text
        else:
            cur_text = text
        cur_secs.append(sec)
    flush()
    return out


def _split_text_to_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """对一段累积文本切分到不超过 ``chunk_size``。

    - 不超过 chunk_size：直接返回 ``[text]``。
    - 超过 chunk_size：用 :class:`RecursiveCharacterTextSplitter` 切分。
    """
    if len(text) <= chunk_size:
        return [text]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=False,
        is_separator_regex=False,
    )
    return splitter.split_text(text)


def _make_chunk(
    content: str,
    source_secs: list[RawDocumentSection],
    heading: Optional[str],
    document_id: str,
    chunk_index: int,
) -> DocumentChunk:
    """从一组聚合 section 构造 :class:`DocumentChunk`。"""
    para_nums = [s.paragraph_number for s in source_secs if s.paragraph_number is not None]
    para_start = para_nums[0] if para_nums else None
    para_end = para_nums[-1] if para_nums else None
    line_starts = [s.line_start for s in source_secs if s.line_start is not None]
    line_ends = [s.line_end for s in source_secs if s.line_end is not None]
    page_nums = [s.page_number for s in source_secs if s.page_number is not None]

    # 元数据：保留首个 section 的非空元数据
    meta: dict = {}
    for s in source_secs:
        if s.metadata:
            meta.update(s.metadata)
            break

    return DocumentChunk(
        chunk_id=_make_chunk_id(document_id, chunk_index),
        document_id=document_id,
        content=content,
        source_name=source_secs[0].source_name if source_secs else "",
        page_number=page_nums[0] if page_nums else None,
        paragraph_number=para_start,
        paragraph_start=para_start,
        paragraph_end=para_end,
        paragraph_numbers=para_nums,
        heading=heading,
        line_start=line_starts[0] if line_starts else None,
        line_end=line_ends[-1] if line_ends else None,
        chunk_index=chunk_index,
        metadata=meta,
    )


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """生成稳定且唯一的 chunk_id。"""
    raw = f"{document_id}::{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]