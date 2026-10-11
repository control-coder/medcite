"""schema 与术语表文件测试。

验证:
1. EvalSample / KnowledgeChunk 可创建、可序列化
2. MeSH synonym map 存在
"""

from __future__ import annotations

import json
from pathlib import Path

from medidiag.schemas import (
    EvalSample,
    KnowledgeChunk,
    chunk_to_dict,
    read_jsonl,
    sample_to_dict,
    write_jsonl,
)

# ===== schema 单元测试 =====


def test_eval_sample_defaults() -> None:
    s = EvalSample(
        sample_id="test_001",
        source="manual",
        question="Is this a test?",
        gold_answer="yes",
    )
    assert s.sample_id == "test_001"
    assert s.gold_evidence_ids == []
    assert s.label_source == "dataset"
    assert s.labeler == "dataset"
    assert s.review_status == "single_checked"
    assert s.options is None


def test_knowledge_chunk_defaults() -> None:
    c = KnowledgeChunk(
        chunk_id="kb_001",
        source="test",
        source_id="book1",
        text="some content",
    )
    assert c.evidence_level == "level_5_other"
    assert c.metadata == {}


def test_serialization_roundtrip(tmp_path: Path) -> None:
    samples = [
        EvalSample(
            sample_id="s1",
            source="PubMedQA",
            question="q1",
            gold_answer="yes",
            gold_evidence_ids=["kb_001"],
        )
    ]
    path = tmp_path / "test.jsonl"
    write_jsonl([sample_to_dict(s) for s in samples], path)
    loaded = read_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0]["sample_id"] == "s1"
    assert loaded[0]["gold_evidence_ids"] == ["kb_001"]


def test_chunk_serialization(tmp_path: Path) -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="kb_001",
            source="MedQA_textbook",
            source_id="Anatomy_Gray",
            text="content",
        )
    ]
    path = tmp_path / "chunks.jsonl"
    write_jsonl([chunk_to_dict(c) for c in chunks], path)
    loaded = read_jsonl(path)
    assert loaded[0]["source"] == "MedQA_textbook"


# ===== 术语表文件测试 =====

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MESH_SYNONYMS = (
    _PROJECT_ROOT / "src" / "medidiag" / "rag" / "medical_terms" / "mesh_synonyms.json"
)


def test_mesh_synonyms_exist() -> None:
    assert _MESH_SYNONYMS.exists(), "mesh_synonyms.json not found"
    with _MESH_SYNONYMS.open(encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) > 100, f"too few mesh terms: {len(data)}"
