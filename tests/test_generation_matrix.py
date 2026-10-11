"""生成层/端到端矩阵：口径与评测函数一致，区间可复现；只读入库报告。"""

from __future__ import annotations

import json
from pathlib import Path

from eval import generation_matrix as gm

ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads((ROOT / "artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json").read_text(encoding="utf-8"))
COMMITTED = json.loads((ROOT / "artifacts/reports/llm/generation-matrix-v3-20261011.json").read_text(encoding="utf-8"))


def _cell(retrieved: list[str], first: dict) -> dict:
    return {"retrieved": retrieved, "trials": [first]}


def _tiny_report() -> dict:
    ok = {"decision": "answered", "selected": ["g1"], "model_called": True}
    blocked = {"decision": "invalid", "selected": [], "detail": "摘录不是本轮完整原文", "model_called": True,
               "feedback_retry": {"decision": "answered", "selected": ["g2"], "model_called": True}}
    abstain = {"decision": "abstained", "selected": [], "model_called": True}
    rows = [
        {"sample_id": "a", "bucket": "direct", "gold": ["g1"], "conditions": {"c": _cell(["g1"], ok)}},
        {"sample_id": "b", "bucket": "direct", "gold": ["g2"], "conditions": {"c": _cell(["g2"], blocked)}},
        {"sample_id": "n", "bucket": "near_miss", "gold": [], "conditions": {"c": _cell(["x"], abstain)}},
    ]
    return {"config": {"conditions": ["c"], "model": "m"}, "per_query": rows}


def test_row_counts_binding_and_retry_separately() -> None:
    row = gm.condition_row(_tiny_report(), "c")
    assert (row["binding_pass_first"]["k"], row["binding_pass_first"]["n"]) == (2, 3)
    assert (row["binding_pass_after_retry"]["k"], row["binding_pass_after_retry"]["n"]) == (3, 3)
    assert (row["useful_answer"]["k"], row["useful_answer"]["n"]) == (2, 2)
    assert (row["useful_answer_first_pass"]["k"]) == 0 + 1  # 首次只有 a 选对，b 被拦截
    assert row["false_answer"]["k"] == 0 and row["blocked_reasons"] == ["摘录不是本轮完整原文"]
    assert row["f1"] == 1.0


def test_committed_matrix_matches_report_and_evaluator_headline() -> None:
    assert gm.build_matrix(REPORT) == COMMITTED["matrix"]
    dense_agent = COMMITTED["matrix"]["dense_agent"]["trial_0"]
    assert (dense_agent["useful_answer"]["k"], dense_agent["false_answer"]["k"]) == (54, 0)
    assert dense_agent["f1"] == 0.973
