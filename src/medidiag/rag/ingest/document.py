"""统一文档模型：Document → Section → Element。

所有解析器（HTML、PDF、DOCX、表格、JSON）都输出这一结构，后续分块、
元数据与索引只读这个模型，不关心原始格式。设计参考 Unstructured 的元素划分。

元数据字段对应《RAG升级计划.md》第五节：source_type、evidence_level、
published_at、accessed_at、language、section_path、topic、license、content_hash，
另加 encoding、period 等解析期信息。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Self

# 元素类型：标题、段落、列表、表格
ELEMENT_KINDS = ("heading", "paragraph", "list", "table")


@dataclass
class Element:
    """文档中的最小内容单元。

    表格元素把表头与数据行分开保存；``text`` 保存行序列化后的检索文本
    （「列名：值；列名：值」），引用校验对比的就是这段序列化文本。
    """

    kind: str  # ELEMENT_KINDS 之一
    text: str
    level: int = 0  # 仅 heading 使用：1–4
    items: list[str] = field(default_factory=list)  # 仅 list 使用
    header: list[str] = field(default_factory=list)  # 仅 table 使用：表头
    rows: list[list[str]] = field(default_factory=list)  # 仅 table 使用：数据行
    metadata: dict[str, Any] = field(default_factory=dict)  # 工作表名、行号、页码等

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            kind=str(data["kind"]),
            text=str(data.get("text", "")),
            level=int(data.get("level", 0)),
            items=[str(x) for x in data.get("items", [])],
            header=[str(x) for x in data.get("header", [])],
            rows=[[str(c) for c in row] for row in data.get("rows", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Section:
    """由标题（h1–h4 或等价编号）划分出的章节。"""

    title: str
    level: int  # 1–4
    elements: list[Element] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "level": self.level,
                "elements": [e.to_dict() for e in self.elements]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            title=str(data["title"]),
            level=int(data["level"]),
            elements=[Element.from_dict(e) for e in data.get("elements", [])],
        )


@dataclass
class DocumentMetadata:
    """文档级元数据。解析器填写自己知道的字段，其余由清单或编排层补。"""

    source_type: str = ""  # who_factsheet / who_publication / nhc_document / epidemic_report / wikipedia
    evidence_level: str = ""
    published_at: str | None = None
    accessed_at: str | None = None
    language: str = "zh-CN"
    topic: str | None = None
    license: str | None = None
    content_hash: str = ""  # 原始快照字节内容的 sha256
    encoding: str | None = None  # 检测出的原始编码
    period: str | None = None  # 疫情月报的统计周期
    source_url: str | None = None
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Document:
    """解析后的完整文档。section_path 由 elements/章节层级在读侧派生。"""

    doc_id: str
    metadata: DocumentMetadata
    sections: list[Section] = field(default_factory=list)

    def iter_section_paths(self) -> list[tuple[Section, list[str]]]:
        """返回每个章节及其标题路径（h1 > h2 > …），供元数据 section_path 使用。"""
        out: list[tuple[Section, list[str]]] = []
        stack: list[tuple[int, str]] = []
        for section in self.sections:
            while stack and stack[-1][0] >= section.level:
                stack.pop()
            stack.append((section.level, section.title))
            out.append((section, [t for _, t in stack]))
        return out

    def element_count(self) -> int:
        return sum(len(s.elements) for s in self.sections)

    def table_count(self) -> int:
        return sum(1 for s in self.sections for e in s.elements if e.kind == "table")

    def full_text(self) -> str:
        """拼接全部元素的检索文本，供逐字核对与去重使用。"""
        parts: list[str] = []
        for section in self.sections:
            if section.title:
                parts.append(section.title)
            for element in section.elements:
                if element.kind == "list":
                    parts.extend(element.items)
                else:
                    parts.append(element.text)
        return "\n".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "metadata": self.metadata.to_dict(),
                "sections": [s.to_dict() for s in self.sections]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            doc_id=str(data["doc_id"]),
            metadata=DocumentMetadata.from_dict(dict(data.get("metadata", {}))),
            sections=[Section.from_dict(s) for s in data.get("sections", [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Self:
        return cls.from_dict(json.loads(line))


def serialize_table_row(header: list[str], row: list[str]) -> str:
    """把一行表格序列化为「列名：值；列名：值」检索文本。

    列数多于表头时多余单元格按「列N」命名；空单元格跳过。
    """
    parts: list[str] = []
    for i, cell in enumerate(row):
        cell = cell.strip()
        if not cell:
            continue
        name = header[i].strip() if i < len(header) and header[i].strip() else f"列{i + 1}"
        parts.append(f"{name}：{cell}")
    return "；".join(parts)
