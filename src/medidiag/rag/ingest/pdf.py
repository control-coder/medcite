"""PDF 解析器：pdfplumber 按页抽文本与表格。

处理规则（《RAG升级计划.md》第五节）：
- 跨页重复出现的行判为页眉页脚并去除；
- 合并中文断行（行尾无句末标点且下一行不以编号开头）；
- 按「一、（一）1.」等编号与字号识别标题；
- 表格单独抽取为表格元素；
- 每页可抽取字符少于阈值判为扫描页，记入报告并跳过。

页眉页脚检测、断行合并、标题识别是纯函数，可单测；pdfplumber 只在
``parse_pdf`` 里做 IO。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medidiag.rag.ingest.document import (
    Document,
    DocumentMetadata,
    Element,
    Section,
    serialize_table_row,
)

# 每页可抽取字符少于此值判为扫描页
SCANNED_PAGE_MIN_CHARS = 20
# 一行至少在这么多页出现才判为页眉页脚
HEADER_FOOTER_MIN_PAGES = 3
# 标题编号：一、（一）1. （1） 1.1 等
# 数字编号后不接数字，避免把「75.8号决议」「2024年」这类数值误判为编号
_HEADING_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"^第[一二三四五六七八九十百\d]+[章节篇部分]"), 1),
    (re.compile(r"^[一二三四五六七八九十]+、\S"), 1),
    (re.compile(r"^（[一二三四五六七八九十]+）\s*\S"), 2),
    (re.compile(r"^\([一二三四五六七八九十]+\)\s*\S"), 2),
    (re.compile(r"^\d{1,2}\.\d+(\.\d+)*[、\s]+(?!\d)\S"), 3),
    (re.compile(r"^\d{1,2}[、.．]\s*(?![\d.．])\S"), 3),
    (re.compile(r"^（\d{1,2}）\S"), 4),
    (re.compile(r"^\(\d{1,2}\)\S"), 4),
)
# 句末标点：行尾没有它说明可能被断行
_SENTENCE_END = "。！？；：.!?;:"
# 列表/编号开头：下一行以它开头时不合并
_LIST_START = re.compile(
    r"^([一二三四五六七八九十]+、|（[一二三四五六七八九十\d]+）|\([一二三四五六七八九十\d]+\)|"
    r"\d+[、.．]|[•·●○▪-]\s)"
)
# 整行只是编号（PDF 常把编号与标题文字断成两行）：「1.1」「一、」「（一）」
_BARE_NUMBER = re.compile(
    r"^(\d+\.\d+(\.\d+)*|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|"
    r"\([一二三四五六七八九十]+\)|第[一二三四五六七八九十\d]+[章节])$"
)

# 标题最长长度；超过则即使匹配编号也当正文
_HEADING_MAX_LEN = 60


@dataclass
class PageContent:
    """一页的抽取结果。"""

    page_no: int
    lines: list[str] = field(default_factory=list)
    font_sizes: list[float] = field(default_factory=list)  # 每行的代表字号
    tables: list[Element] = field(default_factory=list)
    is_scanned: bool = False


@dataclass
class PdfReport:
    """解析报告：扫描页、跳过的页、表格数等。"""

    total_pages: int = 0
    scanned_pages: list[int] = field(default_factory=list)
    table_count: int = 0
    element_count: int = 0


def detect_header_footer(pages_lines: list[list[str]],
                         min_pages: int = HEADER_FOOTER_MIN_PAGES) -> set[str]:
    """找出跨页重复出现的行（页眉页脚）。

    一行至少在 ``min_pages`` 个不同页面出现才判定；短文档（不足 min_pages 页）
    不做判定，避免误删正文。页码类行（纯数字、「第 X 页」）直接判定。
    """
    if len(pages_lines) < min_pages:
        return set()
    page_of_line: dict[str, set[int]] = {}
    for page_no, lines in enumerate(pages_lines):
        for line in set(lines):
            page_of_line.setdefault(line, set()).add(page_no)
    repeated = {line for line, pages in page_of_line.items() if len(pages) >= min_pages}
    page_num = re.compile(r"^(第\s*\d+\s*页(共\s*\d+\s*页)?|[-—–]?\s*\d+\s*[-—–]?)$")
    for lines in pages_lines:
        for line in lines:
            if page_num.match(line):
                repeated.add(line)
    return repeated


def remove_header_footer(pages_lines: list[list[str]], repeated: set[str]) -> list[list[str]]:
    """把判定为页眉页脚的行从各页删除，保持行序。"""
    return [[line for line in lines if line not in repeated] for lines in pages_lines]


def _is_numbered_start(line: str) -> bool:
    """行是否以编号/标题模式开头（不合并到上一行）。"""
    return bool(_LIST_START.match(line)) or classify_heading(line) > 0


def merge_broken_lines(lines: list[str]) -> list[str]:
    """合并中文断行：行尾无句末标点且下一行不以编号或标题模式开头时合并。

    标题行自身也不吸收后续行（「一、病原学」与正文断开）。
    合并时直接拼接（中文断行不需要空格）；行尾是 ASCII 字母或数字时补空格，
    避免英文单词粘连。
    """
    merged: list[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue
        prev = merged[-1]
        if not prev:
            merged.append(line)
            continue
        # 前一行是裸编号且下一行不是编号开头时，合并为完整标题行；
        # 两个编号连续出现说明是目录残余或排版碎片，不合并
        if (_BARE_NUMBER.match(prev) and prev[-1] not in _SENTENCE_END
                and not _BARE_NUMBER.match(line) and not _LIST_START.match(line)):
            merged[-1] = prev + " " + line if prev[-1].isascii() else prev + line
            continue
        if prev[-1] in _SENTENCE_END or _is_numbered_start(line) or classify_heading(prev):
            merged.append(line)
            continue
        # 前一行以半角字符结尾时补空格，否则直接拼接
        if prev[-1].isascii() and (prev[-1].isalnum()):
            merged[-1] = prev + " " + line
        else:
            merged[-1] = prev + line
    return merged


# 目录条目：编号+标题+页码（末尾是空格加纯数字页码），不是正文标题
_TOC_ENTRY = re.compile(r"\s\d{1,4}$")


def classify_heading(line: str, font_size: float | None = None,
                     body_size: float | None = None) -> int:
    """判断一行是不是标题，返回层级（1–4），不是标题返回 0。

    依据：编号模式；或字号明显大于正文字号（>=1.15 倍）且行较短。
    编号后以页码（空格加纯数字）结尾的行是目录条目，不是正文标题。
    """
    if not line or len(line) > _HEADING_MAX_LEN:
        return 0
    if _TOC_ENTRY.search(line):
        return 0
    for pattern, level in _HEADING_PATTERNS:
        match = pattern.match(line)
        if match:
            # 数字编号（3、4 级）的标题文字部分很长或是完整句子时，
            # 是正文编号段落不是标题；中文编号（一、（一））与「第X章」不受此限
            if level >= 3:
                # 编号前缀之后的文字
                rest = re.sub(r"^[\d.．、（）()\s]+", "", line)
                if len(rest) > 25 or line.endswith(("。", "；", ";")):
                    return 0
            return level
    if font_size and body_size and font_size >= body_size * 1.15 and len(line) <= 30:
        return 2
    return 0


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def merge_broken_lines_with_sizes(lines: list[str],
                                  sizes: list[float]) -> list[tuple[str, float]]:
    """merge_broken_lines 的字号版：合并行时保留首行字号（标题通常不因断行产生）。"""
    merged: list[tuple[str, float]] = []
    for i, line in enumerate(lines):
        size = sizes[i] if i < len(sizes) else 0.0
        if not merged:
            merged.append((line, size))
            continue
        prev_text, prev_size = merged[-1]
        if not prev_text:
            merged.append((line, size))
            continue
        # 前一行是裸编号且下一行不是编号开头时，合并为完整标题行；
        # 两个编号连续出现说明是目录残余或排版碎片，不合并
        if (_BARE_NUMBER.match(prev_text) and prev_text[-1] not in _SENTENCE_END
                and not _BARE_NUMBER.match(line) and not _LIST_START.match(line)):
            sep = " " if prev_text[-1].isascii() else ""
            merged[-1] = (prev_text + sep + line, prev_size)
            continue
        if (prev_text[-1] in _SENTENCE_END or _is_numbered_start(line)
                or classify_heading(prev_text)):
            merged.append((line, size))
            continue
        if prev_text[-1].isascii() and prev_text[-1].isalnum():
            merged[-1] = (prev_text + " " + line, prev_size)
        else:
            merged[-1] = (prev_text + line, prev_size)
    return merged


def _drop_toc_lines(lines_with_sizes: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """识别并去掉目录区：连续 ≥3 行的「编号+标题+页码」条目。

    目录页里的条目不该作为正文段落，也不该作为标题。连续 3 行以上匹配
    目录模式的片段整段丢弃；正文里偶尔出现的一两行目录式引用保留。
    """
    out: list[tuple[str, float]] = []
    run: list[tuple[str, float]] = []  # 当前目录条目连续段

    def flush() -> None:
        if len(run) < 3:
            out.extend(run)
        run.clear()

    for line, size in lines_with_sizes:
        # 目录条目：末尾是页码（空格加纯数字），且行长在标题范围内
        is_toc = bool(_TOC_ENTRY.search(line)) and len(line) <= _HEADING_MAX_LEN
        if is_toc:
            run.append((line, size))
        else:
            flush()
            out.append((line, size))
    flush()
    return out


def pages_to_sections(pages: list[PageContent]) -> list[Section]:
    """把清洗后的页内容组装为章节列表。"""
    all_sizes = [s for page in pages for s in page.font_sizes if s > 0]
    body_size = _median(all_sizes)

    sections: list[Section] = [Section(title="", level=1)]
    for page in pages:
        if page.is_scanned:
            continue
        merged = merge_broken_lines_with_sizes(page.lines, page.font_sizes)
        for line, size in _drop_toc_lines(merged):
            level = classify_heading(line, font_size=size or None, body_size=body_size)
            if level:
                sections.append(Section(title=line, level=level))
            else:
                sections[-1].elements.append(Element(kind="paragraph", text=line,
                                                     metadata={"page": page.page_no}))
        for table in page.tables:
            sections[-1].elements.append(table)
    return [s for s in sections if s.title or s.elements]


def _line_sizes(page: Any, lines: list[str]) -> list[float | None]:
    """按 pdfplumber 的单词坐标把字号归到每一行。取每行单词字号的中位数。

    行是按 extract_text 的文本行切分的，与单词坐标不完全一一对应；这里用
    行首词匹配做近似对齐，对不上时该行字号为 None，标题识别退化为只用编号模式。
    """
    words = page.extract_words(extra_attrs=["size"])
    if not words or not lines:
        return [None] * len(lines)
    # 词按 (top, x0) 排序后聚成行（top 差小于 3pt 视为同一行）
    rows: list[list[float]] = []
    current: list[float] = []
    last_top: float | None = None
    for word in sorted(words, key=lambda w: (round(float(w["top"]), 1), float(w["x0"]))):
        top = float(word["top"])
        if last_top is None or abs(top - last_top) <= 3.0:
            current.append(float(word["size"]))
        else:
            rows.append(current)
            current = [float(word["size"])]
        last_top = top
    if current:
        rows.append(current)
    row_sizes = [_median(r) for r in rows]
    # 行数通常等于文本行数；不一致时按顺序对齐，多余/不足补 None
    sizes: list[float | None] = []
    for i in range(len(lines)):
        sizes.append(row_sizes[i] if i < len(row_sizes) else None)
    return sizes


def _extract_page(page: Any, page_no: int) -> PageContent:
    """用 pdfplumber 抽取一页。page 类型为 pdfplumber.page.Page（无类型存根）。"""
    text = page.extract_text() or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sizes = [s if s is not None else 0.0 for s in _line_sizes(page, lines)]

    tables: list[Element] = []
    for raw in page.extract_tables():
        if not raw:
            continue
        header = [str(c or "").strip() for c in raw[0]]
        rows = [[str(c or "").strip() for c in row] for row in raw[1:] if any(row)]
        if not rows and not any(header):
            continue
        text_rows = [serialize_table_row(header, row) for row in rows]
        tables.append(Element(kind="table", text="\n".join(text_rows), header=header,
                              rows=rows, metadata={"page": page_no}))

    char_count = len(text.strip())
    return PageContent(
        page_no=page_no,
        lines=lines,
        font_sizes=sizes[: len(lines)],
        tables=tables,
        is_scanned=char_count < SCANNED_PAGE_MIN_CHARS,
    )


def parse_pdf(path: Path | str, doc_id: str,
              metadata: DocumentMetadata | None = None) -> tuple[Document, PdfReport]:
    """解析 PDF 文件，返回 Document 与报告。扫描页记入报告并跳过。"""
    import pdfplumber

    report = PdfReport()
    page_contents: list[PageContent] = []
    with pdfplumber.open(str(path)) as pdf:
        report.total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            content = _extract_page(page, i)
            if content.is_scanned:
                report.scanned_pages.append(i)
            page_contents.append(content)

    repeated = detect_header_footer([p.lines for p in page_contents if not p.is_scanned])
    cleaned = remove_header_footer([p.lines for p in page_contents], repeated)
    for content, lines in zip(page_contents, cleaned, strict=True):
        content.lines = lines

    sections = pages_to_sections(page_contents)
    meta = metadata or DocumentMetadata()
    document = Document(doc_id=doc_id, metadata=meta, sections=sections)
    report.element_count = document.element_count()
    report.table_count = document.table_count()
    return document, report
