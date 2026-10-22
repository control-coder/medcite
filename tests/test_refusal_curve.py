"""拒答取舍曲线：合成数据上的口径检查，以及对入库报告的零模型核对（CI 可运行）。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from eval import refusal_curve as rc

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "artifacts/reports/llm/refusal-curve-v3-20261022.json"
SOURCES = {"v3": ROOT / "artifacts/reports/llm/llm-pipeline-v3-rerank-20261017.json",
           "v3_ext": ROOT / "artifacts/reports/llm/llm-pipeline-v3-ext-20261017.json"}


def _item(sid: str, gold: list[str], score: float, decision: str = "answered", selected: list[str] | None = None,
          evidence: list[str] | None = None, split: str = "test") -> dict:
    return {"sample_id": sid, "split": split, "bucket": "direct" if gold else "near_miss", "gold": gold,
            "evidence": evidence if evidence is not None else (gold[:1] or ["x"]),
            "rerank_top1": score, "dense_top1": score / 2, "decision": decision,
            "selected": selected if selected is not None else (gold[:1] if decision == "answered" else [])}


ITEMS = [
    _item("a1", ["g1"], 0.9),
    _item("a2", ["g2"], 0.6),
    _item("a3", ["g3"], 0.2, selected=["other"]),  # 作答但引用的不是正确片段
    _item("a4", ["g4"], 0.4, decision="abstained"),
    _item("u1", [], 0.5, decision="abstained"),
    _item("u2", [], 0.1, decision="answered", selected=["x"]),  # 误答
]


def test_operating_point_is_the_recorded_pipeline() -> None:
    point = rc.gated_point(ITEMS, "rerank_top1", -math.inf)
    assert point["tau"] == "-inf" and point["gate_blocked"] == 0
    assert point["false_answers"]["k"] == 1 and point["false_answers"]["n"] == 2
    assert point["over_refusals"]["k"] == 1 and point["useful_answers"]["k"] == 2
    assert point["answered_without_gold"]["k"] == 1 and point["answered"] == 4


def test_gate_only_removes_answers_and_counts_scores_equal_to_tau_as_answered() -> None:
    point = rc.gated_point(ITEMS, "rerank_top1", 0.2)  # 只挡住 u2（0.1）；a3 恰好等于 τ，仍作答
    assert point["gate_blocked"] == 1 and point["false_answers"]["k"] == 0
    assert point["answered_without_gold"]["k"] == 1
    high = rc.gated_point(ITEMS, "rerank_top1", 0.7)
    assert high["useful_answers"]["k"] == 1 and high["over_refusals"]["k"] == 3
    assert rc.gated_point(ITEMS, "rerank_top1", math.inf)["answered"] == 0


def test_curve_is_monotone_in_answers() -> None:
    points = rc.curve(ITEMS, "rerank_top1", rc.gated_point)
    assert points[0]["tau"] == "-inf" and points[-1]["tau"] == "+inf"
    answered = [p["answered"] for p in points]
    assert answered == sorted(answered, reverse=True)


def test_score_only_ignores_model_decisions() -> None:
    point = rc.score_only_point(ITEMS, "rerank_top1", 0.3)
    assert point["false_answers"]["k"] == 1  # u1 0.5 ≥ 0.3，即使模型拒答，只看分数也算作答
    assert point["useful_answers"]["k"] == 3 and point["over_refusals"]["k"] == 1


def test_auroc_and_score_selection() -> None:
    assert rc.score_auroc(ITEMS, "rerank_top1") == pytest.approx(0.75)
    tied = [{**i, "dense_top1": i["rerank_top1"]} for i in ITEMS]
    assert rc.select_score(tied) == "rerank_top1"  # 并列取 rerank_top1
    flipped = [{**i, "dense_top1": (1.0 if i["gold"] else 0.0)} for i in ITEMS]
    assert rc.select_score(flipped) == "dense_top1"


def test_budget_threshold_blocks_at_most_the_budget() -> None:
    dev = [_item(f"d{k}", [f"g{k}"], k / 20) for k in range(1, 21)] + [_item("du", [], 0.01)]
    tau = rc.select_budget_tau(dev, "rerank_top1", 0.10)  # 20 条可回答，最多挡 2 条
    assert sum(1 for i in dev if i["gold"] and i["rerank_top1"] < tau) == 2
    assert rc.select_budget_tau(dev, "rerank_top1", 0.01) == -math.inf
    ties = [_item(f"t{k}", [f"g{k}"], 0.5) for k in range(10)]
    assert sum(1 for i in ties if i["rerank_top1"] < rc.select_budget_tau(ties, "rerank_top1", 0.2)) == 0


def test_youden_threshold_prefers_lower_tau_on_ties() -> None:
    dev = [_item("a", ["g"], 0.8), _item("u", [], 0.2, decision="abstained")]
    tau = rc.select_youden_tau(dev, "rerank_top1")
    assert 0.2 < tau <= 0.8


@dataclass(frozen=True)
class _Hit:
    chunk_id: str
    text: str


def test_reconstruct_reranks_the_union_of_agent_searches() -> None:
    from medidiag.rag.rerank import Reranker

    corpus = {"c1": "甲", "c2": "乙", "c3": "丙", "c4": "丁"}
    results = {"原问题": ["c1", "c2"], "换说法": ["c3", "c2"]}
    scores = {"甲": 0.1, "乙": 0.5, "丙": 0.9, "丁": 0.0}
    reranker = Reranker(lambda pairs: [scores[t] for _, t in pairs], model_name="m", model_revision="r", candidates=2)
    trace = {"steps": [{"action": "search", "query": "原问题"}, {"action": "search", "query": "换说法"},
                       {"action": "stop"}, {"action": "merge", "method": "rerank"}]}
    out = rc.reconstruct("原问题", trace, lambda q, d: [_Hit(c, corpus[c]) for c in results[q][:d]], reranker)
    assert out == [("c3", 0.9), ("c2", 0.5), ("c1", 0.1)]


def test_build_items_rejects_evidence_that_differs_from_the_report() -> None:
    from eval.retrieval_benchmark import Dataset
    from medidiag.rag.rerank import Reranker

    dataset = Dataset(chunks=[{"chunk_id": "c1", "text": "甲"}, {"chunk_id": "c2", "text": "乙"}],
                      queries=[{"sample_id": "q", "split": "test", "bucket": "direct", "question": "问",
                                "gold_evidence_ids": ["c1"]}])
    report = {"agent": {"dense_agent_rerank": {"traces": {"q": {"steps": [
                  {"action": "search", "query": "问"}, {"action": "merge", "method": "rerank"}]}}}},
              "per_query": [{"sample_id": "q", "conditions": {"dense_agent_rerank": {
                  "retrieved": ["c2", "c1"], "trials": [{"decision": "answered", "selected": ["c1"]}]}}}]}
    reranker = Reranker(lambda pairs: [1.0 if t == "甲" else 0.0 for _, t in pairs],
                        model_name="m", model_revision="r", candidates=2)
    hits = [_Hit("c1", "甲"), _Hit("c2", "乙")]
    with pytest.raises(ValueError, match="不一致"):
        rc.build_items(dataset, report, "dense_agent_rerank", lambda q, d: hits, reranker, lambda q, p: 0.5)


# ----------------------------------------------------------------- 入库报告（不需要模型）

@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_committed_items_carry_the_recorded_pipeline_outcomes(committed: dict) -> None:
    for name, path in SOURCES.items():
        source = json.loads(path.read_text(encoding="utf-8"))
        rows = {r["sample_id"]: r for r in source["per_query"]}
        test_items = [i for i in committed["per_query"][name] if i["sample_id"] in rows]
        assert len(test_items) == len(rows)
        for item in test_items:
            row = rows[item["sample_id"]]
            assert item["evidence"] == row["conditions"][committed["condition"]]["retrieved"]
            outcome = rc.final_outcome(row, committed["condition"])
            assert (item["decision"], item["selected"]) == (outcome["decision"], outcome.get("selected", []))
            assert item["rerank_scores"] == sorted(item["rerank_scores"], reverse=True)


def test_committed_curve_recomputes_from_stored_scores(committed: dict) -> None:
    items = committed["per_query"]
    rebuilt = rc.build([i for i in items["v3"] if i["split"] == "dev"],
                       {"v3_test": [i for i in items["v3"] if i["split"] == "test"], "v3_ext": items["v3_ext"]})
    assert rebuilt["selection"] == committed["selection"]
    assert rebuilt["evaluations"] == committed["evaluations"]


def test_operating_point_matches_the_documented_end_to_end_numbers(committed: dict) -> None:
    v3 = committed["evaluations"]["v3_test"]["operating_point"]
    ext = committed["evaluations"]["v3_ext"]["operating_point"]
    assert (v3["useful_answers"]["k"], v3["over_refusals"]["k"], v3["false_answers"]["k"]) == (54, 1, 0)
    assert (ext["useful_answers"]["k"], ext["over_refusals"]["k"], ext["false_answers"]["k"]) == (111, 15, 0)
