"""阶段 1 schema 与数据集测试。

验证:
1. EvalSample / KnowledgeChunk 可创建、可序列化
2. 评测集与知识库文件存在
3. 公开题占比 >= 40%（PLAN.md 要求）
4. sample_id 带前缀（数据泄露防护）
5. chunk source_id 不等于 sample_id（数据泄露防护）
6. MeSH synonym map 存在
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


# ===== 数据集完整性测试 =====

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EVAL_SET = _PROJECT_ROOT / "eval" / "datasets" / "eval_set.jsonl"
_KB_CHUNKS = _PROJECT_ROOT / "eval" / "datasets" / "knowledge_chunks.jsonl"
_MESH_SYNONYMS = (
    _PROJECT_ROOT / "src" / "medidiag" / "rag" / "medical_terms" / "mesh_synonyms.json"
)


@pytest.fixture(scope="module")
def eval_samples() -> list[dict]:
    return read_jsonl(_EVAL_SET)


@pytest.fixture(scope="module")
def kb_chunks() -> list[dict]:
    return read_jsonl(_KB_CHUNKS)


def test_eval_set_exists() -> None:
    assert _EVAL_SET.exists(), "eval_set.jsonl not found"


def test_knowledge_chunks_exist() -> None:
    assert _KB_CHUNKS.exists(), "knowledge_chunks.jsonl not found"


def test_eval_set_non_empty(eval_samples: list[dict]) -> None:
    assert len(eval_samples) > 0, "eval_set is empty"


def test_kb_chunks_non_empty(kb_chunks: list[dict]) -> None:
    assert len(kb_chunks) > 0, "knowledge_chunks is empty"


def test_public_ratio_meets_plan(eval_samples: list[dict]) -> None:
    """PLAN.md 要求公开题占比 >= 40%。"""
    public = sum(1 for s in eval_samples if s["source"] in ("PubMedQA", "MedQA"))
    ratio = public / len(eval_samples)
    assert ratio >= 0.40, f"public ratio {ratio:.2%} < 40%"


def test_sample_id_has_prefix(eval_samples: list[dict]) -> None:
    """sample_id 应带来源前缀（数据泄露防护：不直接用 PubMed ID）。"""
    for s in eval_samples:
        assert "_" in s["sample_id"], f"sample_id without prefix: {s['sample_id']}"
        assert s["sample_id"].startswith(("pubmedqa_", "medqa_", "manual_")), (
            f"unexpected sample_id prefix: {s['sample_id']}"
        )


def test_chunk_source_id_not_sample_id(
    eval_samples: list[dict], kb_chunks: list[dict]
) -> None:
    """数据泄露防护：chunk source_id 不应等于任何 sample_id。"""
    sample_ids = {s["sample_id"] for s in eval_samples}
    for c in kb_chunks:
        assert c["source_id"] not in sample_ids, (
            f"chunk source_id == sample_id: {c['source_id']}"
        )


def test_chunk_has_required_fields(kb_chunks: list[dict]) -> None:
    """每个 chunk 必须有 chunk_id / source / source_id / text。"""
    for c in kb_chunks:
        assert "chunk_id" in c
        assert "source" in c
        assert "source_id" in c
        assert "text" in c
        assert len(c["text"]) > 0


def test_eval_sample_has_required_fields(eval_samples: list[dict]) -> None:
    """每个评测样本必须有 sample_id / source / question / gold_answer。"""
    for s in eval_samples:
        assert "sample_id" in s
        assert "source" in s
        assert "question" in s
        assert "gold_answer" in s
        assert "gold_evidence_ids" in s
        assert "label_source" in s


def test_pubmedqa_samples_have_evidence(eval_samples: list[dict]) -> None:
    """PubMedQA 样本应有 gold_evidence（来自 CONTEXTS）。"""
    pmc = [s for s in eval_samples if s["source"] == "PubMedQA"]
    has_evidence = sum(1 for s in pmc if s["gold_evidence_ids"])
    # 至少 80% 的 PubMedQA 样本应有 evidence
    assert has_evidence / len(pmc) >= 0.80, (
        f"only {has_evidence}/{len(pmc)} PubMedQA samples have evidence"
    )


def test_medqa_samples_no_evidence(eval_samples: list[dict]) -> None:
    """MedQA 样本 gold_evidence 为空（无 explanation），label_source=dataset_no_evidence。"""
    medqa = [s for s in eval_samples if s["source"] == "MedQA"]
    for s in medqa:
        assert s["gold_evidence_ids"] == [], (
            f"MedQA sample {s['sample_id']} should have empty gold_evidence"
        )
        assert s["label_source"] == "dataset_no_evidence"


def test_mesh_synonyms_exist() -> None:
    assert _MESH_SYNONYMS.exists(), "mesh_synonyms.json not found"
    with _MESH_SYNONYMS.open(encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) > 100, f"too few mesh terms: {len(data)}"
