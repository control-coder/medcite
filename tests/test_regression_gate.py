"""回归门槛与回归集：全部零网络，只读仓库内的报告、阈值和语料。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from eval import failure_attribution as fa
from eval import regression_gate as gate
from medidiag.rag.retrieval import Bm25Retriever
from medidiag.schemas import KnowledgeChunk, read_jsonl

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = json.loads(gate.THRESHOLDS.read_text(encoding="utf-8"))
REPORT = json.loads((ROOT / THRESHOLDS["report"]).read_text(encoding="utf-8"))
ATTRIBUTION = json.loads((ROOT / "artifacts/reports/llm/failure-attribution-v3-20261011.json").read_text(encoding="utf-8"))
CASES = read_jsonl(ROOT / "examples/rag_regression/cases.jsonl")
DATASET = ROOT / "examples/public_health_v3"


@pytest.fixture(scope="module")
def bm25() -> Bm25Retriever:
    retriever = Bm25Retriever([KnowledgeChunk(**row) for row in read_jsonl(DATASET / "chunks.jsonl")], "cjk_bigram")
    retriever.build_index()
    return retriever


def test_committed_report_meets_thresholds() -> None:
    assert gate.check_report(REPORT, THRESHOLDS) == []


def test_gate_flags_each_kind_of_regression() -> None:
    worse = copy.deepcopy(REPORT)
    worse["generation_with_feedback_retry"]["dense"]["trial_0"]["false_answer_rate"]["k"] = 1
    worse["generation_with_feedback_retry"]["dense_agent"]["trial_0"]["useful_answer_rate"]["k"] = 50
    worse["generation_with_feedback_retry"]["bm25_rewrite"]["trial_0"]["contract_violations_or_errors"] = 2
    worse["generation"]["bm25_agent"]["trial_0"]["contract_violations_or_errors"] = 4
    messages = gate.check_report(worse, THRESHOLDS)
    assert len(messages) == 4
    assert any("dense/trial_0" in m and "误答" in m for m in messages)
    assert any("dense_agent" in m and "有用回答" in m for m in messages)
    assert any("bm25_rewrite" in m and "被拦截" in m for m in messages)
    assert any("bm25_agent" in m and "一次结果" in m for m in messages)


def test_gate_flags_missing_condition() -> None:
    partial = copy.deepcopy(REPORT)
    del partial["generation_with_feedback_retry"]["dense"]
    assert any("缺少条件" in m for m in gate.check_report(partial, THRESHOLDS))


def test_attribution_report_is_up_to_date_with_rules() -> None:
    """入库的归因报告必须等于用当前规则对入库评测报告重新归因的结果，规则改了就得重新生成。"""
    assert fa.attribute(REPORT)["failures"] == ATTRIBUTION["failures"]
    assert ATTRIBUTION["failure_totals_by_condition"] == fa.attribute(REPORT)["failure_totals_by_condition"]


def test_regression_cases_are_the_first_trial_failures() -> None:
    failing = {f["sample_id"] for f in ATTRIBUTION["failures"] if f["trial"] == 0}
    assert {c["sample_id"] for c in CASES} == failing
    assert len(CASES) == ATTRIBUTION["distinct_failing_queries"]
    queries = {q["sample_id"]: q for q in read_jsonl(DATASET / "queries.jsonl")}
    for case in CASES:
        assert case["question"] == queries[case["sample_id"]]["question"]
        assert case["gold_evidence_ids"] == queries[case["sample_id"]]["gold_evidence_ids"]
        assert all(item["label"] in fa.LABELS for item in case["failures"])


def test_application_bm25_still_returns_recorded_top3_for_regression_cases(bm25: Bm25Retriever) -> None:
    for case in CASES:
        got = [hit.chunk_id for hit in bm25.search(case["question"], top_k=3)]
        assert got == case["bm25_top3"], case["sample_id"]
        assert sorted(set(case["gold_evidence_ids"]) & set(got)) == case["bm25_gold_in_top3"]


def test_application_bm25_equals_evaluated_bm25_on_all_test_queries(bm25: Bm25Retriever) -> None:
    """应用里的检索器必须就是评测过的那个：84 条 test 查询的前 3 个片段与报告里记录的逐条一致。"""
    queries = {q["sample_id"]: q for q in read_jsonl(DATASET / "queries.jsonl")}
    assert len(REPORT["per_query"]) == 84
    for row in REPORT["per_query"]:
        got = [hit.chunk_id for hit in bm25.search(queries[row["sample_id"]]["question"], top_k=3)]
        assert got == row["conditions"]["bm25"]["retrieved"], row["sample_id"]
