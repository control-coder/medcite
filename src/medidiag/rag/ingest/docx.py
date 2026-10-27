"""DOCX 解析器：python-docx 按段落样式识别标题层级，表格转表格元素。

DOC/WPS 老格式：本机有 LibreOffice（soffice）时用无界面模式转为 DOCX 再解析；
没有就在报告中记为跳过，不静默丢弃。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medidiag.rag.ingest.document import (
    Document,
    DocumentMetadata,
    Element,
    Section,
    serialize_table_row,
)
from medidiag.rag.ingest.pdf import classify_heading

# 段落样式名 → 标题层级；中英文都认
_STYLE_LEVELS: dict[str, int] = {}
for _i in range(1, 5):
    _STYLE_LEVELS[f"heading {_i}"] = _i
    _STYLE_LEVELS[f"标题 {_i}"] = _i
    _STYLE_LEVELS[f"标题{_i}"] = _i


@dataclass
class DocxReport:
    """解析报告：元素数、表格数、跳过的转换。"""

    element_count: int = 0
    table_count: int = 0
    converted_from: str | None = None  # 老格式转换来源（.doc/.wps）
    skipped: str | None = None  # 跳过原因


def heading_level_of_paragraph(paragraph: Any) -> int:
    """按段落样式与编号模式识别标题层级；不是标题返回 0。

    paragraph 类型为 docx.text.paragraph.Paragraph（无类型存根）。
    """
    style_name = (paragraph.style.name or "").strip().lower() if paragraph.style else ""
    if style_name in _STYLE_LEVELS:
        return _STYLE_LEVELS[style_name]
    text = paragraph.text.strip()
    # 政府文件常不用样式，只靠编号
    return classify_heading(text)


def _iter_block_items(parent: Any) -> Any:
    """按文档顺序产出段落与表格（python-docx 默认分开遍历会乱序）。"""
    from docx.document import Document as _DocxDocument
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if isinstance(parent, _DocxDocument):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._element
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def parse_docx(
    path: Path | str, doc_id: str, metadata: DocumentMetadata | None = None
) -> tuple[Document, DocxReport]:
    """解析 DOCX 文件。"""
    import docx

    report = DocxReport()
    doc = docx.Document(str(path))
    sections: list[Section] = [Section(title="", level=1)]

    for block in _iter_block_items(doc):
        if block.__class__.__name__ == "Paragraph":
            text = block.text.strip()
            if not text:
                continue
            level = heading_level_of_paragraph(block)
            if level:
                sections.append(Section(title=text, level=level))
            else:
                sections[-1].elements.append(Element(kind="paragraph", text=text))
        else:  # Table
            header: list[str] = []
            rows: list[list[str]] = []
            for r_idx, row in enumerate(block.rows):
                cells = [cell.text.strip() for cell in row.cells]
                # 合并单元格会产生重复文本，去掉与左格相同的重复
                deduped = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
                if r_idx == 0:
                    header = deduped
                else:
                    rows.append(deduped)
            text_rows = [serialize_table_row(header, row) for row in rows]
            sections[-1].elements.append(
                Element(kind="table", text="\n".join(text_rows), header=header, rows=rows)
            )

    meta = metadata or DocumentMetadata()
    document = Document(
        doc_id=doc_id, metadata=meta, sections=[s for s in sections if s.title or s.elements]
    )
    report.element_count = document.element_count()
    report.table_count = document.table_count()
    return document, report


def find_soffice() -> str | None:
    """返回本机 LibreOffice soffice 路径，没有返回 None。"""
    found = shutil.which("soffice")
    if found:
        return found
    # Windows 常见安装位置
    for candidate in (
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def convert_to_docx(path: Path, work_dir: Path) -> Path | None:
    """用 soffice 无界面模式把 .doc/.wps 转为 .docx；失败返回 None。"""
    soffice = find_soffice()
    if not soffice:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 本机 LibreOffice 转换公开文档
        [soffice, "--headless", "--convert-to", "docx", "--outdir", str(work_dir), str(path)],
        check=False,
        capture_output=True,
        timeout=120,
    )
    converted = work_dir / (path.stem + ".docx")
    return converted if converted.exists() else None


def parse_doc_like(
    path: Path | str,
    doc_id: str,
    metadata: DocumentMetadata | None = None,
    work_dir: Path | None = None,
) -> tuple[Document | None, DocxReport]:
    """解析 .docx/.doc/.wps。老格式无 soffice 时返回 (None, 报告记 skipped)。"""
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".docx":
        return parse_docx(src, doc_id, metadata)
    if suffix in {".doc", ".wps"}:
        report = DocxReport()
        converted = convert_to_docx(src, work_dir or src.parent)
        if converted is None:
            report.skipped = "本机未安装 LibreOffice（soffice），无法转换老格式，已跳过"
            return None, report
        document, report = parse_docx(converted, doc_id, metadata)
        report.converted_from = suffix
        return document, report
    report = DocxReport(skipped=f"不支持的格式：{suffix}")
    return None, report
