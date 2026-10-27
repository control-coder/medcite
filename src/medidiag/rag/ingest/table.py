"""XLSX / CSV 解析器：按工作表读入，识别表头，行序列化为检索文本。

每张工作表产出一个章节；每行一个段落元素（「列名：值；列名：值」），
另生成一个整表摘要块作为表格元素。记录工作表名与行号。
"""

from __future__ import annotations

import csv
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

# 单元格字符上限：疫情统计表可能有超长单元格，截断避免检索文本爆炸
_CELL_MAX_CHARS = 500


@dataclass
class TableReport:
    """解析报告。"""

    sheet_count: int = 0
    row_count: int = 0
    element_count: int = 0


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) > _CELL_MAX_CHARS:
        return text[:_CELL_MAX_CHARS]
    return text


def _looks_like_header(row: list[str]) -> bool:
    """第一行像表头：非空、没有纯数字单元格占多数。"""
    cells = [c for c in row if c]
    if not cells:
        return False
    numeric = sum(1 for c in cells if c.replace(".", "").replace("%", "").isdigit())
    return numeric <= len(cells) // 2


def _sheet_to_section(name: str, rows: list[list[str]]) -> Section:
    """一张工作表 → 一个章节：表头识别后每行一个段落，外加整表摘要。"""
    section = Section(title=name, level=1)
    if not rows:
        return section
    header: list[str] = []
    data_rows = rows
    if _looks_like_header(rows[0]):
        header = rows[0]
        data_rows = rows[1:]

    row_elements: list[Element] = []
    serialized_rows: list[str] = []
    for idx, row in enumerate(data_rows, start=2 if header else 1):
        if not any(row):
            continue
        text = serialize_table_row(header, row)
        if not text:
            continue
        serialized_rows.append(text)
        row_elements.append(Element(
            kind="paragraph", text=text,
            metadata={"sheet": name, "row": idx, "content_kind": "table_row"}))
    section.elements.extend(row_elements)

    # 整表摘要：行列规模与列名，供概览检索
    summary = f"表格「{name}」共 {len(serialized_rows)} 行"
    if header:
        summary += f"，列：{'、'.join(h for h in header if h)}"
    section.elements.append(Element(
        kind="table", text=summary, header=header, rows=data_rows,
        metadata={"sheet": name, "content_kind": "table_summary"}))
    return section


def parse_xlsx(path: Path | str, doc_id: str,
               metadata: DocumentMetadata | None = None) -> tuple[Document, TableReport]:
    """解析 XLSX，每张工作表一个章节。"""
    import openpyxl

    report = TableReport()
    workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    sections: list[Section] = []
    for sheet in workbook.worksheets:
        rows = [[_cell_text(cell) for cell in row] for row in sheet.iter_rows(values_only=True)]
        rows = [row for row in rows if any(row)]
        if not rows:
            continue
        sections.append(_sheet_to_section(sheet.title, rows))
        report.sheet_count += 1
        report.row_count += sum(1 for e in sections[-1].elements if e.kind == "paragraph")
    workbook.close()

    meta = metadata or DocumentMetadata()
    document = Document(doc_id=doc_id, metadata=meta, sections=sections)
    report.element_count = document.element_count()
    return document, report


def parse_csv(path: Path | str, doc_id: str,
              metadata: DocumentMetadata | None = None,
              encoding: str = "utf-8") -> tuple[Document, TableReport]:
    """解析 CSV，整个文件视作一张工作表。"""
    report = TableReport(sheet_count=1)
    with open(path, encoding=encoding, newline="") as fh:
        rows = [[_cell_text(cell) for cell in row] for row in csv.reader(fh)]
    rows = [row for row in rows if any(row)]
    name = Path(path).stem
    section = _sheet_to_section(name, rows)
    report.row_count = sum(1 for e in section.elements if e.kind == "paragraph")

    meta = metadata or DocumentMetadata()
    document = Document(doc_id=doc_id, metadata=meta, sections=[section] if rows else [])
    report.element_count = document.element_count()
    return document, report
