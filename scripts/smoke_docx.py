"""DOCX 真实文件冒烟测试脚本（阶段 2.2）。

输出：
1. 解析前后字符数与覆盖率
2. 块类型分布（paragraph / table / textbox）
3. heading 数量
4. chunk 数量、平均 / 中位 / P90 / 最短 / 最长 chunk 长度
5. < 10 / < 50 字符 chunk 数量
6. 前 5 个 chunk 元数据 + 随机 5 个主体内容 chunk 元数据

不输出大段药品正文。

运行：
    python scripts/smoke_docx.py [path-to-docx]
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DOCX = ROOT / "国家基本药物目录（2026年版）(OCR).docx"


def _fmt(n):
    if isinstance(n, int):
        return f"{n:,}"
    if isinstance(n, float):
        return f"{n:.2f}"
    return str(n)


def _short_sample(s: str, max_len: int = 80) -> str:
    s = s.replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def main() -> int:
    from rag.parsers import DocxParser
    from rag.splitter import split_sections

    docx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCX
    if not docx_path.exists():
        print(f"文件不存在: {docx_path}")
        return 2

    print("=" * 72)
    print(f"DOCX 冒烟测试：{docx_path.name}")
    print("=" * 72)

    # 1. 解析
    parser = DocxParser()
    sections = parser.parse(docx_path)
    stats = parser.last_filter_stats

    para_secs = [s for s in sections if s.block_type == "paragraph"]
    table_secs = [s for s in sections if s.block_type == "table"]
    textbox_secs = [s for s in sections if s.block_type == "textbox"]
    headings = {s.heading for s in sections if s.heading}

    total_chars = sum(len(s.content) for s in sections)
    print("\n[1] 解析结果")
    print(f"  文件大小          = {_fmt(docx_path.stat().st_size)} 字节")
    print(f"  有效 section 总数 = {_fmt(len(sections))}")
    print(f"  解析总字符数      = {_fmt(total_chars)}")
    print(f"  普通段落块        = {_fmt(len(para_secs))} (字符 {_fmt(sum(len(s.content) for s in para_secs))})")
    print(f"  表格块            = {_fmt(len(table_secs))} (字符 {_fmt(sum(len(s.content) for s in table_secs))})")
    print(f"  文本框块          = {_fmt(len(textbox_secs))} (字符 {_fmt(sum(len(s.content) for s in textbox_secs))})")
    print(f"  heading 数量      = {_fmt(len(headings))}")
    print(f"  过滤统计          = {stats}")

    # 2. 分块
    document_id = "smoke-docx-001"
    chunks = split_sections(sections, document_id=document_id)
    lengths = [len(c.content) for c in chunks]
    lengths_sorted = sorted(lengths)
    avg_len = statistics.mean(lengths) if lengths else 0
    median_len = statistics.median(lengths_sorted) if lengths_sorted else 0
    p90_idx = max(0, int(len(lengths_sorted) * 0.9) - 1)
    p90_len = lengths_sorted[p90_idx] if lengths_sorted else 0
    short_10 = sum(1 for n in lengths if n < 10)
    short_50 = sum(1 for n in lengths if n < 50)

    print("\n[2] 分块统计")
    print(f"  chunk 总数        = {_fmt(len(chunks))}")
    print(f"  平均长度          = {_fmt(avg_len)}")
    print(f"  中位长度          = {_fmt(median_len)}")
    print(f"  P90 长度          = {_fmt(p90_len)}")
    print(f"  最短 chunk        = {_fmt(min(lengths) if lengths else 0)}")
    print(f"  最长 chunk        = {_fmt(max(lengths) if lengths else 0)}")
    print(f"  < 10 字符 chunk   = {_fmt(short_10)}")
    print(f"  < 50 字符 chunk   = {_fmt(short_50)}")

    # 3. 前 5 个 chunk
    print("\n[3] 前 5 个 chunk 元数据")
    for c in chunks[:5]:
        print(f"  [{c.chunk_index}] {c.chunk_id} 块类型={c.block_type or '-'}")
        print(f"      heading = {c.heading!r}")
        print(f"      table   = table_index={c.table_index} rows={c.row_start}~{c.row_end}")
        print(f"      para    = {c.paragraph_start}~{c.paragraph_end}")
        print(f"      sample  = {_short_sample(c.content)}")

    # 4. 随机 5 个主体内容 chunk
    body_chunks = [c for c in chunks if len(c.content) >= 30]
    sample = random.sample(body_chunks, k=min(5, len(body_chunks))) if body_chunks else []
    print("\n[4] 随机 5 个主体内容 chunk 元数据")
    for c in sample:
        print(f"  [{c.chunk_index}] {c.chunk_id} 块类型={c.block_type or '-'}")
        print(f"      heading = {c.heading!r}")
        print(f"      table   = table_index={c.table_index} rows={c.row_start}~{c.row_end}")
        print(f"      sample  = {_short_sample(c.content)}")

    # 5. 覆盖率（与原始 XML 可见字符对比）
    import zipfile
    from xml.etree import ElementTree as ET

    W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(docx_path, "r") as zf:
        with zf.open("word/document.xml") as f:
            root = ET.parse(f).getroot()
        all_visible = sum(
            len(t.text.strip())
            for t in root.iter(f"{W_NS}t")
            if t.text and t.text.strip()
        )
    coverage = total_chars / all_visible * 100 if all_visible else 0
    print("\n[5] 覆盖率")
    print(f"  document.xml 可见字符 = {_fmt(all_visible)}")
    print(f"  parser 提取字符       = {_fmt(total_chars)}")
    print(f"  覆盖率                = {coverage:.2f}%")

    # 6. 不变量检查
    print("\n[6] 不变量检查")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index 不连续"
    print("  ✓ chunk_index 连续")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 有重复"
    print("  ✓ chunk_id 唯一")
    empty = [c for c in chunks if not c.content.strip()]
    assert not empty, f"存在空 chunk: {len(empty)}"
    print("  ✓ 无空 chunk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
