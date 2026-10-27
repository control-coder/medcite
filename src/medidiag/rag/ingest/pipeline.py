"""解析编排：读 sources.json → 按 format 选解析器 → 清洗 → 去重 → 输出 JSONL 与报告。

只处理清单中 license_conclusion 为「纳入」的文档。解析后的文档写到
``<cache>/parsed/<doc_id>.json``（每行一个 Document JSON），报告写到
``<cache>/parsed/report.json``。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medidiag.rag.ingest import clean, dedup
from medidiag.rag.ingest.document import Document, DocumentMetadata
from medidiag.rag.ingest.encoding import decode_bytes

_FORMAT_TO_PARSER = {"html", "pdf", "docx", "xlsx", "csv", "json", "doc", "wps"}


@dataclass
class FileReport:
    """单个文件的解析报告。"""

    doc_id: str
    format: str
    status: str  # ok / failed / skipped
    element_count: int = 0
    table_count: int = 0
    scanned_pages: list[int] = field(default_factory=list)
    encoding: str | None = None
    failure_reason: str | None = None
    content_hash: str = ""


@dataclass
class IngestReport:
    """整轮解析报告。"""

    files: list[FileReport] = field(default_factory=list)
    dedup_pairs: list[dict[str, Any]] = field(default_factory=list)
    kept_doc_ids: list[str] = field(default_factory=list)
    dropped_doc_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [asdict(f) for f in self.files],
            "dedup_pairs": self.dedup_pairs,
            "kept_doc_ids": self.kept_doc_ids,
            "dropped_doc_ids": self.dropped_doc_ids,
        }


def _metadata_from_source(entry: dict[str, Any], source_types: dict[str, Any]) -> DocumentMetadata:
    """从清单条目构造文档元数据。"""
    stype = str(entry.get("source_type", ""))
    type_info = source_types.get(stype, {})
    return DocumentMetadata(
        source_type=stype,
        evidence_level=str(entry.get("evidence_level") or type_info.get("evidence_level", "")),
        published_at=entry.get("published_at"),
        accessed_at=entry.get("accessed_at"),
        language=str(entry.get("language", "zh-CN")),
        license=entry.get("license") or type_info.get("license"),
        period=entry.get("period"),
        source_url=entry.get("url"),
        title=str(entry.get("title", "")),
    )


def _parse_one(path: Path, doc_id: str, fmt: str, meta: DocumentMetadata,
               cache_dir: Path) -> tuple[Document | None, FileReport]:
    """按格式解析单个文件。失败不抛异常，记入报告。"""
    from medidiag.rag.ingest import docx as docx_mod
    from medidiag.rag.ingest import html as html_mod
    from medidiag.rag.ingest import pdf as pdf_mod
    from medidiag.rag.ingest import table as table_mod

    report = FileReport(doc_id=doc_id, format=fmt, status="ok")
    raw = path.read_bytes()
    report.content_hash = hashlib.sha256(raw).hexdigest()
    meta.content_hash = report.content_hash
    document: Document | None

    try:
        if fmt in {"html", "json"}:
            result = decode_bytes(raw)
            meta.encoding = result.encoding
            report.encoding = result.encoding
            if fmt == "html":
                document = html_mod.parse_html(result.text, doc_id, meta, meta.source_type)
            else:
                payload = json.loads(result.text)
                document = html_mod.parse_mediawiki_json(payload, doc_id, meta)
            report.element_count = document.element_count()
            report.table_count = document.table_count()
            return document, report
        if fmt == "pdf":
            document, pdf_report = pdf_mod.parse_pdf(path, doc_id, meta)
            report.element_count = pdf_report.element_count
            report.table_count = pdf_report.table_count
            report.scanned_pages = pdf_report.scanned_pages
            return document, report
        if fmt in {"docx", "doc", "wps"}:
            work_dir = cache_dir / "converted"
            document, docx_report = docx_mod.parse_doc_like(path, doc_id, meta, work_dir)
            if document is None:
                report.status = "skipped"
                report.failure_reason = docx_report.skipped
                return None, report
            report.element_count = docx_report.element_count
            report.table_count = docx_report.table_count
            return document, report
        if fmt == "xlsx":
            document, table_report = table_mod.parse_xlsx(path, doc_id, meta)
            report.element_count = table_report.element_count
            return document, report
        if fmt == "csv":
            result = decode_bytes(raw)
            meta.encoding = result.encoding
            report.encoding = result.encoding
            # CSV 先统一转成 UTF-8 文本再解析，兼容 GBK 来源
            tmp = cache_dir / "converted" / f"{doc_id}.csv"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(result.text, encoding="utf-8")
            document, table_report = table_mod.parse_csv(tmp, doc_id, meta)
            report.element_count = table_report.element_count
            return document, report
        report.status = "failed"
        report.failure_reason = f"不支持的格式：{fmt}"
        return None, report
    except Exception as exc:  # 单文件失败不中断整轮
        report.status = "failed"
        report.failure_reason = f"{type(exc).__name__}: {exc}"
        return None, report


def run_ingest(sources_path: Path, cache_dir: Path) -> IngestReport:
    """执行整轮解析、清洗与去重。

    输出：
    - ``<cache>/parsed/documents.jsonl``：每行一个 Document JSON；
    - ``<cache>/parsed/report.json``：解析报告。
    """
    # utf-8-sig 兼容 Windows 工具写出的带 BOM 清单
    sources = json.loads(sources_path.read_text(encoding="utf-8-sig"))
    source_types = sources.get("source_types", {})
    raw_dir = cache_dir / "raw"
    parsed_dir = cache_dir / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    report = IngestReport()
    documents: list[Document] = []

    for entry in sources.get("documents", []):
        if entry.get("license_conclusion") != "纳入":
            continue
        doc_id = str(entry["doc_id"])
        fmt = str(entry.get("format", "")).lower()
        if fmt not in _FORMAT_TO_PARSER:
            report.files.append(FileReport(
                doc_id=doc_id, format=fmt, status="failed",
                failure_reason=f"未知格式：{fmt}"))
            continue
        raw_path = raw_dir / str(entry.get("source_type", "")) / f"{doc_id}.{fmt}"
        if not raw_path.exists():
            report.files.append(FileReport(
                doc_id=doc_id, format=fmt, status="failed",
                failure_reason=f"快照不存在：{raw_path}"))
            continue
        meta = _metadata_from_source(entry, source_types)
        document, file_report = _parse_one(raw_path, doc_id, fmt, meta, cache_dir)
        report.files.append(file_report)
        if document is not None:
            clean.clean_document(document)
            documents.append(document)

    # 文档级去重
    doc_dedup = dedup.dedup_documents(documents)
    kept_ids = set(doc_dedup.kept_doc_ids)
    kept_documents = [d for d in documents if d.doc_id in kept_ids]
    report.kept_doc_ids = doc_dedup.kept_doc_ids
    report.dropped_doc_ids = doc_dedup.dropped_doc_ids
    report.dedup_pairs.extend(asdict(p) for p in doc_dedup.pairs)

    # 段落级跨来源去重（在保留的文档之间）
    para_dedup = dedup.dedup_paragraphs(kept_documents)
    report.dedup_pairs.extend(asdict(p) for p in para_dedup.pairs)

    # 写 JSONL：每行一个 Document
    out_path = parsed_dir / "documents.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for document in kept_documents:
            fh.write(document.to_json() + "\n")

    report_path = parsed_dir / "report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
