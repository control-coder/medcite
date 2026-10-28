"""扩充测试集构造与池化标注的离线测试：抽样约束、重复片段、人工复核结果的合并，以及标注池的构成。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import pooled_judgments as pj
from scripts import build_paraphrase_extension as bx

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def v3() -> tuple[list[dict], list[dict]]:
    return _read(ROOT / "examples/public_health_v3/chunks.jsonl"), _read(ROOT / "examples/public_health_v3/queries.jsonl")


def test_committed_extension_set_is_reproducible(v3) -> None:
    chunks, queries = v3
    target = ROOT / "examples/public_health_v3_ext"
    judgments = _read(target / "pooled_judgments.jsonl") if (target / "pooled_judgments.jsonl").exists() else []
    rows = bx.build(chunks, queries, judgments)
    assert "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows) == (target / "queries.jsonl").read_text(encoding="utf-8")
    assert (target / "chunks.jsonl").read_bytes() == (ROOT / "examples/public_health_v3/chunks.jsonl").read_bytes()


def test_anchors_avoid_v3_gold_and_respect_the_overlap_cap(v3) -> None:
    chunks, queries = v3
    text = {c["chunk_id"]: c["text"] for c in chunks}
    v3_gold = {g for q in queries for g in q["gold_evidence_ids"]}
    order = bx.sample_order(chunks, queries)
    assert not set(order) & v3_gold and len(order) == len(set(order))
    for i, question in bx.ANSWERABLE.items():
        if question is not None:
            assert bx.overlap(question, text[order[i]]) <= bx.MAX_OVERLAP, i


def test_identical_text_chunks_are_both_gold(v3) -> None:
    chunks, queries = v3
    rows = {r["sample_id"]: r for r in bx.build(chunks, queries, [])}
    text = {c["chunk_id"]: c["text"] for c in chunks}
    for row in rows.values():
        anchor_texts = {text[g] for g in row["gold_evidence_ids"]}
        assert len(anchor_texts) == (1 if row["gold_evidence_ids"] else 0), row["sample_id"]  # 未合并复核时只有锚定段及其重复
        twins = [c["chunk_id"] for c in chunks if c["text"] in anchor_texts]
        assert sorted(twins) == row["gold_evidence_ids"]


def test_overlap_above_the_cap_is_refused(v3, monkeypatch) -> None:
    chunks, queries = v3
    order = bx.sample_order(chunks, queries)
    copied = next(c["text"] for c in chunks if c["chunk_id"] == order[0])
    monkeypatch.setitem(bx.ANSWERABLE, 0, copied)  # 原样照抄片段
    with pytest.raises(ValueError, match="重合率"):
        bx.build(chunks, queries, [])


def test_human_review_adds_gold_only_when_confirmed(v3) -> None:
    chunks, queries = v3
    judgments = [
        {"sample_id": "pv3x_P_001", "chunk_id": "kb_covid_risk_01", "human_label": "answers"},
        {"sample_id": "pv3x_P_002", "chunk_id": "kb_covid_risk_01", "human_label": "does_not_answer"},
        {"sample_id": "pv3x_P_003", "chunk_id": "kb_covid_risk_01", "human_label": None},
    ]
    rows = {r["sample_id"]: r for r in bx.build(chunks, queries, judgments)}
    assert "kb_covid_risk_01" in rows["pv3x_P_001"]["gold_evidence_ids"] and "人工复核补充" in rows["pv3x_P_001"]["rationale"]
    assert "kb_covid_risk_01" not in rows["pv3x_P_002"]["gold_evidence_ids"]
    assert "kb_covid_risk_01" not in rows["pv3x_P_003"]["gold_evidence_ids"]


def test_confirmed_evidence_for_a_near_miss_question_is_an_error(v3) -> None:
    chunks, queries = v3
    near_miss = f"pv3x_N_{min(bx.NEAR_MISS):03d}"
    with pytest.raises(ValueError, match="没有对应证据"):
        bx.build(chunks, queries, [{"sample_id": near_miss, "chunk_id": "kb_covid_risk_01", "human_label": "answers"}])


def test_pool_collects_non_gold_top3_across_conditions_and_reports() -> None:
    queries = [{"sample_id": "a", "gold_evidence_ids": ["g"]}, {"sample_id": "b", "gold_evidence_ids": []}]
    first = {"per_query": [
        {"sample_id": "a", "conditions": {"bm25": {"retrieved": ["g", "x", "y"]}, "dense": {"retrieved": ["x", "z"]}}},
        {"sample_id": "b", "conditions": {"bm25": {"retrieved": ["x"]}}}]}
    second = {"per_query": [{"sample_id": "a", "conditions": {"bm25_rerank": {"retrieved": ["y", "g"]}}}]}
    assert pj.pool(queries, [first, second]) == [
        {"sample_id": "a", "chunk_id": "x", "conditions": ["bm25", "dense"]},
        {"sample_id": "a", "chunk_id": "y", "conditions": ["bm25", "bm25_rerank"]},
        {"sample_id": "a", "chunk_id": "z", "conditions": ["dense"]},
        {"sample_id": "b", "chunk_id": "x", "conditions": ["bm25"]},
    ]


class _Result:
    def __init__(self, parsed: Any) -> None:
        self.parsed_json = parsed


class _Provider:
    def __init__(self, parsed: Any) -> None:
        self.parsed = parsed
        self.requests: list[Any] = []

    def generate(self, request: Any, *, timeout_s: float, idempotency_key: str) -> _Result:
        self.requests.append(request)
        return _Result(self.parsed)


@pytest.mark.parametrize("parsed,label", [
    ({"answers": True, "reason": "原文给出了数字"}, "answers"),
    ({"answers": False, "reason": "只是话题相同"}, "does_not_answer"),
    ({"answers": "yes"}, "invalid"),
    (None, "invalid"),
])
def test_judge_accepts_only_a_boolean_verdict(parsed, label) -> None:
    provider = _Provider(parsed)
    assert pj.judge(provider, "问题", "原文", "mimo-v2.6-flash")["model_label"] == label
    user = json.loads(provider.requests[0].messages[1]["content"])
    assert user == {"question": "问题", "passage": "原文"}  # 每次只给一段原文


def test_rerun_keeps_existing_human_review() -> None:
    from eval.retrieval_benchmark import Dataset

    dataset = Dataset(chunks=[{"chunk_id": "x", "source_id": "s", "text": "原文"}],
                      queries=[{"sample_id": "a", "split": "test", "bucket": "paraphrase", "question": "问题",
                                "gold_evidence_ids": ["g"]}])
    report = {"per_query": [{"sample_id": "a", "conditions": {"bm25": {"retrieved": ["x"]}}}]}
    existing = {("a", "x"): {"human_label": "does_not_answer", "human_note": "只是话题相同"}}
    rows = pj.run(dataset, [report], _Provider({"answers": True}), "mimo-v2.6-flash", 1, existing)
    assert rows == [{"sample_id": "a", "chunk_id": "x", "conditions": ["bm25"], "model_label": "answers",
                     "model_reason": "", "human_label": "does_not_answer", "human_note": "只是话题相同"}]


def test_confirmed_pairs_stay_in_the_pool_after_merging_into_gold() -> None:
    """复核确认的片段并入正确片段后，重跑池化仍要保留这条记录，否则下次构造会把它从正确片段里丢掉。"""
    from eval.retrieval_benchmark import Dataset

    dataset = Dataset(chunks=[{"chunk_id": "x", "source_id": "s", "text": "原文"}],
                      queries=[{"sample_id": "a", "split": "test", "bucket": "paraphrase", "question": "问题",
                                "gold_evidence_ids": ["g", "x"]}])  # x 已由复核并入
    report = {"per_query": [{"sample_id": "a", "conditions": {"bm25": {"retrieved": ["g", "x"]}}}]}
    existing = {("a", "x"): {"human_label": "answers", "human_note": "原文给出了答案"}}
    rows = pj.run(dataset, [report], _Provider({"answers": True}), "mimo-v2.6-flash", 1, existing)
    assert [(r["chunk_id"], r["human_label"]) for r in rows] == [("x", "answers")]


def test_committed_judgments_cover_every_model_positive() -> None:
    """模型判为回答了或判断无效的，都有人工结论；被确认的都已并入正确片段。"""
    target = ROOT / "examples/public_health_v3_ext"
    rows = _read(target / "pooled_judgments.jsonl")
    gold = {q["sample_id"]: set(q["gold_evidence_ids"]) for q in _read(target / "queries.jsonl")}
    for row in rows:
        if row["model_label"] in {"answers", "invalid"}:
            assert row["human_label"] in {"answers", "does_not_answer"} and row["human_note"], row
        assert (row["chunk_id"] in gold[row["sample_id"]]) == (row["human_label"] == "answers"), row
