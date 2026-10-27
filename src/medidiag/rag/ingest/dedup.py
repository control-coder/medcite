"""近重复检测：基于字符 5-gram 的 MinHash，Jaccard ≥ 0.85 判重复。

粒度与取舍：
- 文档级：整篇文档的全文做 MinHash，跨来源重复的整篇只保留一份；
- 段落级：各文档的段落元素两两之间做跨来源判断（同一文档内不判，
  因为同文档内重复段落通常是正文需要的重申，删了会破坏结构）。

保留哪份：``evidence_level`` 数字越小权威越高（level_1_guideline 高于
level_5_other）；同级时保留元素更多的那份。重复关系记录在返回的报告里，
不修改文档内容本身，由编排层决定剔除哪些。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medidiag.rag.ingest.document import Document

# 计划第五节：Jaccard 相似度不低于 0.85 判为重复
JACCARD_THRESHOLD = 0.85
# MinHash 排列数；128 是精度与成本的常见折中
_NUM_PERM = 128
# 证据等级权威排序：数字越小越权威；未知等级排最后
_LEVEL_ORDER = {
    "level_1_guideline": 1,
    "level_2_review": 2,
    "level_3_rct": 3,
    "level_4_observational": 4,
    "level_5_other": 5,
}

_CJK_SHINGLE = re.compile(r".")


def _authority(document: Document) -> tuple[int, int]:
    """返回 (权威等级, 内容量)，排序时取小者。"""
    level = _LEVEL_ORDER.get(document.metadata.evidence_level, 99)
    return (level, -document.element_count())


def _shingles(text: str, n: int = 5) -> set[str]:
    """字符 n-gram；短文本（不足 n 字）整体作为一个 shingle。"""
    text = re.sub(r"\s+", "", text)
    if len(text) <= n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _minhash(shingles: set[str]) -> object:
    """构造 MinHash。返回类型为 datasketch.MinHash（无类型存根）。"""
    from datasketch import MinHash

    mh = MinHash(num_perm=_NUM_PERM)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh


@dataclass
class DuplicatePair:
    """一对重复：保留 kept_id，去掉 dropped_id。"""

    kept_id: str
    dropped_id: str
    similarity: float
    granularity: str  # document / paragraph


@dataclass
class DedupReport:
    """去重结果：要保留的文档、要剔除的文档、重复关系。"""

    kept_doc_ids: list[str] = field(default_factory=list)
    dropped_doc_ids: list[str] = field(default_factory=list)
    pairs: list[DuplicatePair] = field(default_factory=list)
    # 段落级：需要从文档里剔除的段落元素文本（按文档分组）
    dropped_paragraphs: dict[str, list[str]] = field(default_factory=dict)


def dedup_documents(documents: list[Document],
                    threshold: float = JACCARD_THRESHOLD) -> DedupReport:
    """文档级去重。返回保留与剔除清单，不修改输入。"""
    if len(documents) < 2:
        return DedupReport(kept_doc_ids=[d.doc_id for d in documents])

    hashes = {d.doc_id: _minhash(_shingles(d.full_text())) for d in documents}
    # 按权威排序，逐个与已保留的比较；重复的被剔除
    ordered = sorted(documents, key=_authority)
    kept: list[Document] = []
    report = DedupReport()
    for doc in ordered:
        duplicate_of: tuple[str, float] | None = None
        for other in kept:
            sim = float(hashes[doc.doc_id].jaccard(hashes[other.doc_id]))  # type: ignore[attr-defined]
            if sim >= threshold:
                duplicate_of = (other.doc_id, sim)
                break
        if duplicate_of:
            report.dropped_doc_ids.append(doc.doc_id)
            report.pairs.append(DuplicatePair(
                kept_id=duplicate_of[0], dropped_id=doc.doc_id,
                similarity=duplicate_of[1], granularity="document"))
        else:
            kept.append(doc)
            report.kept_doc_ids.append(doc.doc_id)
    return report


def dedup_paragraphs(documents: list[Document],
                     threshold: float = JACCARD_THRESHOLD) -> DedupReport:
    """段落级跨来源去重：同一内容出现在多个文档时只保留权威文档的那份。

    只在不同文档之间判断；同一文档内的重复段落不动。返回每篇文档需要
    剔除的段落文本清单（不直接改文档，避免破坏章节结构）。
    """
    # 收集所有 (doc_id, 段落文本)，短段落（<30 字）不去重，误伤率高
    entries: list[tuple[Document, str]] = []
    for doc in documents:
        for section in doc.sections:
            for element in section.elements:
                if element.kind == "paragraph" and len(element.text) >= 30:
                    entries.append((doc, element.text))

    hashes = [(doc, text, _minhash(_shingles(text))) for doc, text in entries]
    # 按文档权威排序段落；同一文本先遇到权威文档的
    order = sorted(range(len(hashes)), key=lambda i: _authority(hashes[i][0]))
    kept: list[tuple[Document, str, object]] = []
    report = DedupReport()
    for idx in order:
        doc, text, mh = hashes[idx]
        duplicate_of: tuple[str, float] | None = None
        for kept_doc, _kept_text, kept_mh in kept:
            if kept_doc.doc_id == doc.doc_id:
                continue  # 同文档内不判
            sim = float(mh.jaccard(kept_mh))  # type: ignore[attr-defined]
            if sim >= threshold:
                duplicate_of = (kept_doc.doc_id, sim)
                break
        if duplicate_of:
            report.dropped_paragraphs.setdefault(doc.doc_id, []).append(text)
            report.pairs.append(DuplicatePair(
                kept_id=duplicate_of[0], dropped_id=f"{doc.doc_id}#para",
                similarity=duplicate_of[1], granularity="paragraph"))
        else:
            kept.append((doc, text, mh))
    return report
