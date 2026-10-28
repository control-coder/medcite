"""M6 分块与 v3 标注迁移的单元测试。不联网、不依赖 .cache。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medidiag.rag.ingest.chunking import (
    CHILD_MAX_CHARS,
    PARENT_MAX_CHARS,
    RECURSIVE_MAX_CHARS,
    RECURSIVE_OVERLAP_CHARS,
    _group_child_sentences,
    _split_sentences,
    chunk_corpus,
    chunk_document,
)
from medidiag.rag.ingest.document import Document, DocumentMetadata, Element, Section
from medidiag.schemas import chunk_to_dict, load_knowledge_chunks, write_jsonl
from scripts.migrate_v3_gold import map_excerpt, squash


def _doc(sections: list[Section], doc_id: str = "d1") -> Document:
    return Document(
        doc_id=doc_id,
        metadata=DocumentMetadata(
            source_type="who_factsheet", title="流感", evidence_level="level_5_other",
            published_at="2026-01-01", license="CC", period="2026-01"),
        sections=sections,
    )


# ---------- 递归切分 ----------


def test_recursive_respects_max_chars_and_overlap() -> None:
    # 句长约 48 字 × 12 = 576 字 > 512，迫使切块
    sentence = "流感主要通过飞沫传播，咳嗽打喷嚏时病毒随飞沫排出，密切接触者容易感染，应当注意个人防护。"
    text = sentence * 12
    doc = _doc([Section(title="传播", level=2, elements=[Element(kind="paragraph", text=text)])])
    out = chunk_document(doc, "recursive")
    assert len(out.children) >= 2
    for chunk in out.children:
        # 块体不超过 512（重叠前缀另加，总长不超过 512+76）
        assert len(chunk.text) <= RECURSIVE_MAX_CHARS + RECURSIVE_OVERLAP_CHARS
    # 第 2 块起都有前一块末尾的重叠前缀
    first, second = out.children[0].text, out.children[1].text
    assert second[:RECURSIVE_OVERLAP_CHARS] == first[-RECURSIVE_OVERLAP_CHARS:]
    # 去掉各块重叠前缀后拼接能还原原文本（重叠只是前缀复制，不丢内容）
    pieces = [out.children[0].text] + [c.text[RECURSIVE_OVERLAP_CHARS:] for c in out.children[1:]]
    assert "".join(pieces) == text


def test_recursive_short_text_single_chunk_no_overlap() -> None:
    doc = _doc([Section(title="", level=1, elements=[Element(kind="paragraph", text="短段落。")])])
    out = chunk_document(doc, "recursive")
    assert len(out.children) == 1
    assert out.children[0].text == "短段落。"


def test_recursive_does_not_cross_sections() -> None:
    doc = _doc([
        Section(title="一、概述", level=1, elements=[Element(kind="paragraph", text="第一段内容。")]),
        Section(title="二、预防", level=1, elements=[Element(kind="paragraph", text="第二段内容。")]),
    ])
    out = chunk_document(doc, "recursive")
    assert len(out.children) == 2
    assert out.children[0].metadata["section_path"] == ["一、概述"]
    assert out.children[1].metadata["section_path"] == ["二、预防"]
    assert "第一段" in out.children[0].text and "第二段" not in out.children[0].text


def test_recursive_skips_dropped_paragraphs() -> None:
    kept = "保留的段落，内容足够长，超过三十个字符的低权威重复检查阈值要求。"
    dropped = "被剔除的重复段落，来自低权威来源，内容同样超过三十个字符的长度要求。"
    doc = _doc([Section(title="", level=1, elements=[
        Element(kind="paragraph", text=kept),
        Element(kind="paragraph", text=dropped),
    ])])
    out = chunk_document(doc, "recursive", dropped_paragraphs={"d1": [dropped]})
    assert len(out.children) == 1
    assert dropped not in out.children[0].text


# ---------- 父子分块 ----------


def test_parent_child_child_limits_and_parent_id() -> None:
    sentences = ["第一句内容，说明传播途径。", "第二句内容，说明高危人群。", "第三句内容，说明预防措施。"]
    doc = _doc([Section(title="概述", level=2, elements=[
        Element(kind="paragraph", text="".join(sentences))])])
    out = chunk_document(doc, "parent_child")
    assert out.parents and out.children
    parent_ids = {c.metadata["parent_id"] for c in out.children}
    assert parent_ids == {out.parents[0].chunk_id}
    for chunk in out.children:
        assert len(chunk.text) <= CHILD_MAX_CHARS
        # 每个子块 1–2 句；短碎片合并后（数据质量修复）最多 3 句
        assert len(_split_sentences(chunk.text)) <= 3
    assert len(out.parents[0].text) <= PARENT_MAX_CHARS
    assert out.parents[0].metadata["role"] == "parent"


def test_parent_child_long_paragraph_splits_parents() -> None:
    sentence = "流感病毒属于正黏病毒科，分为甲、乙、丙、丁四型，其中甲型最容易发生变异。"  # 39 字
    text = sentence * 20  # 780 字 > 600
    doc = _doc([Section(title="病原学", level=2, elements=[Element(kind="paragraph", text=text)])])
    out = chunk_document(doc, "parent_child")
    assert len(out.parents) >= 2
    for parent in out.parents:
        assert len(parent.text) <= PARENT_MAX_CHARS
    # 每个子块的 parent_id 都指向存在的父块
    parent_ids = {p.chunk_id for p in out.parents}
    for child in out.children:
        assert child.metadata["parent_id"] in parent_ids


def test_group_child_sentences_hard_split_long_sentence() -> None:
    long_sentence = "无逗号长句" + "甲" * (CHILD_MAX_CHARS + 30) + "。"
    children = _group_child_sentences([long_sentence])
    assert len(children) >= 2
    for child in children:
        assert len(child) <= CHILD_MAX_CHARS


# ---------- 表格 ----------


def test_table_rows_are_standalone_chunks_both_strategies() -> None:
    element = Element(kind="table", text="地区：广东；病例：120\n地区：北京；病例：80",
                      header=["地区", "病例"], rows=[["广东", "120"], ["北京", "80"]])
    doc = _doc([Section(title="统计", level=2, elements=[
        Element(kind="paragraph", text="疫情统计如下表所示，各地区数据详见下表。"),
        element,
    ])])
    for strategy in ("recursive", "parent_child"):
        out = chunk_document(doc, strategy)
        row_chunks = [c for c in out.children if c.metadata["content_kind"] == "table_row"]
        assert [c.text for c in row_chunks] == ["地区：广东；病例：120", "地区：北京；病例：80"]
        # 行块不与正文混切
        assert all("疫情统计" not in c.text for c in row_chunks)
    # 父子策略下整表摘要作为父块
    out = chunk_document(doc, "parent_child")
    assert any(p.metadata["content_kind"] == "table_summary" for p in out.parents)
    summary_parent = next(p for p in out.parents if p.metadata["content_kind"] == "table_summary")
    row_chunks = [c for c in out.children if c.metadata["content_kind"] == "table_row"]
    assert {c.metadata["parent_id"] for c in row_chunks} == {summary_parent.chunk_id}
    # 递归策略下没有父块
    out_rc = chunk_document(doc, "recursive")
    assert out_rc.parents == []


# ---------- chunk_id 稳定性 ----------


def test_chunk_id_format_and_stability() -> None:
    doc = _doc([
        Section(title="概述", level=1, elements=[Element(kind="paragraph", text="流感是急性呼吸道传染病。")]),
        Section(title="预防", level=2, elements=[
            Element(kind="paragraph", text="接种疫苗是最有效的预防手段。"),
            Element(kind="table", text="措施：疫苗", header=["措施"], rows=[["疫苗"]]),
        ]),
    ])
    out1 = chunk_corpus([doc], "parent_child")
    out2 = chunk_corpus([doc], "parent_child")
    ids1 = [c.chunk_id for c in out1.children] + [p.chunk_id for p in out1.parents]
    ids2 = [c.chunk_id for c in out2.children] + [p.chunk_id for p in out2.parents]
    assert ids1 == ids2  # 幂等
    # 格式：{doc_id}:{章节序号}:{策略}:{序号}
    for chunk in out1.children:
        parts = chunk.chunk_id.split(":")
        assert parts[0] == "d1" and parts[2] == "pc" and parts[3].isdigit()
        assert int(parts[1]) in (0, 1)
        assert chunk.metadata["chunk_hash"]
    table_chunk = next(c for c in out1.children if c.metadata["content_kind"] == "table_row")
    assert table_chunk.chunk_id == "d1:1:pc:1"  # 第二个章节内第 2 个子块
    assert table_chunk.metadata["section_path"] == ["概述", "预防"]


# ---------- KnowledgeChunk 兼容 ----------


def test_load_knowledge_chunks_reads_new_and_old(tmp_path: Path) -> None:
    doc = _doc([Section(title="概述", level=1, elements=[
        Element(kind="paragraph", text="流感通过飞沫传播。")])])
    out = chunk_document(doc, "parent_child")
    new_path = tmp_path / "new.jsonl"
    write_jsonl((chunk_to_dict(c) for c in out.children + out.parents), new_path)
    chunks = load_knowledge_chunks(new_path)
    assert len(chunks) == len(out.children) + len(out.parents)
    assert chunks[0].source_id == "d1"
    assert chunks[0].metadata["strategy"] == "parent_child"

    old = {
        "chunk_id": "kb_flu_01", "source": "世界卫生组织", "source_id": "flu",
        "text": "旧格式片段。", "evidence_level": "level_5_other",
        "metadata": {"title": "流感"},
    }
    old_path = tmp_path / "old.jsonl"
    old_path.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_knowledge_chunks(old_path)
    assert loaded[0].chunk_id == "kb_flu_01"


# ---------- 数据质量修复：短碎片合并与控件文字 ----------


def test_child_control_lines_dropped() -> None:
    """纯日期、纯控件文字（「实况报道」、页码）的独立行/块直接丢弃，不进检索。"""
    doc = _doc([Section(title="概述", level=2, elements=[
        Element(kind="paragraph", text="2025年12月8日"),
        Element(kind="paragraph", text="实况报道"),
        Element(kind="paragraph", text="vi"),
        Element(kind="paragraph", text="麻疹是一种传染性很强的病毒性疾病，可接种疫苗预防。"),
    ])])
    out = chunk_document(doc, "parent_child")
    texts = [c.text for c in out.children]
    assert not any("2025年12月8日" in t for t in texts)
    assert not any(t == "实况报道" for t in texts)
    assert not any(t == "vi" for t in texts)
    assert any("传染性很强" in t for t in texts)


def test_child_short_fragments_merged() -> None:
    """不足 15 字的短句与相邻句合并；合并后仍 ≤130 字。"""
    sentences = ["麻疹会致命。", "它是一种传染性很强的病毒性疾病，", "可通过接种疫苗有效预防。"]
    doc = _doc([Section(title="概述", level=2, elements=[
        Element(kind="paragraph", text="".join(sentences))])])
    out = chunk_document(doc, "parent_child")
    assert out.children
    for chunk in out.children:
        assert len(chunk.text) <= 130
    # 短句「麻疹会致命。」（6 字）与相邻句合并，不单独成块
    assert not any(c.text == "麻疹会致命。" for c in out.children)
    assert any("麻疹会致命" in c.text for c in out.children)


def test_child_merge_respects_merge_limit() -> None:
    """合并上限放宽到 130 字只用于合并；超过 110 字的单句硬切仍按 110 字。"""
    long_sentence = "甲" * 120 + "。"  # 121 字单句，硬切
    doc = _doc([Section(title="概述", level=2, elements=[
        Element(kind="paragraph", text=long_sentence)])])
    out = chunk_document(doc, "parent_child")
    for chunk in out.children:
        assert len(chunk.text) <= CHILD_MAX_CHARS


def test_recursive_drops_control_lines() -> None:
    """递归切分同样丢弃纯日期/控件文字行。"""
    doc = _doc([Section(title="概述", level=2, elements=[
        Element(kind="paragraph", text="实况报道"),
        Element(kind="paragraph", text="麻疹是一种传染性很强的病毒性疾病，可接种疫苗预防。"),
    ])])
    out = chunk_document(doc, "recursive")
    texts = [c.text for c in out.children]
    assert not any(t == "实况报道" for t in texts)
    assert any("传染性很强" in t for t in texts)


# ---------- 标注迁移 ----------


def _children(texts: list[str]) -> tuple[list[str], list[str]]:
    # 与迁移脚本一致：用同一 squash（含全半角归一）预处理子块文本
    ids = [f"d1:0:pc:{i}" for i in range(len(texts))]
    return ids, [squash(t) for t in texts]


def test_map_excerpt_single_chunk() -> None:
    ids, texts = _children(["流感通过飞沫传播。", "预防靠接种疫苗。"])
    mapping, gold, reason = map_excerpt("飞沫传播", ids, texts)
    assert mapping == "exact" and gold == [ids[0]] and reason is None


def test_map_excerpt_span_across_children() -> None:
    ids, texts = _children(["流感通过飞沫传播", "，也可接触传播。", "无关内容。"])
    mapping, gold, reason = map_excerpt("流感通过飞沫传播，也可接触传播", ids, texts)
    assert mapping == "span" and gold == ids[:2] and reason is None


def test_map_excerpt_failed() -> None:
    ids, texts = _children(["流感通过飞沫传播。"])
    mapping, gold, reason = map_excerpt("疟疾由按蚊叮咬传播", ids, texts)
    assert mapping == "failed" and gold == [] and reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
