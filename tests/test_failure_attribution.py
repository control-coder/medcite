"""失败归因规则的测试：六类标签互斥、全部失败都打标签、无法由程序判定的拦截必须有人工复核。"""

from __future__ import annotations

import pytest

from eval import failure_attribution as fa


def _ans(selected: list[str]) -> dict:
    return {"decision": "answered", "selected": selected}


ABSTAIN = {"decision": "abstained", "selected": []}
BLOCKED = {"decision": "invalid", "selected": [], "detail": "摘录不是本轮完整原文"}


def test_ideal_outcomes_are_not_failures() -> None:
    assert fa.classify(["g1"], ["g1", "x"], _ans(["g1"]), None) is None
    assert fa.classify([], ["x"], ABSTAIN, None) is None


@pytest.mark.parametrize(
    "gold,retrieved,outcome,label",
    [
        (["g1"], ["x", "y"], ABSTAIN, "检索缺失"),  # 缺正确片段，模型弃答
        (["g1", "g2"], ["g1", "x"], ABSTAIN, "检索缺失"),  # 多片段问题缺一个
        (["g1"], ["x", "y"], _ans(["x"]), "检索噪声"),  # 缺正确片段还拿噪声硬答
        (["g1"], ["g1", "x"], _ans(["x"]), "检索噪声"),  # 正确片段在，却只引用了噪声
        ([], ["x"], _ans(["x"]), "充分性误判"),  # 不可回答却作答
        (["g1"], ["g1", "x"], ABSTAIN, "弃答误判"),  # 正确片段齐全却弃答
        (["g1"], ["x"], BLOCKED, "检索缺失"),  # 缺正确片段，同时被拦截：先算检索
    ],
)
def test_each_failure_gets_exactly_one_label(gold, retrieved, outcome, label) -> None:
    result = fa.classify(gold, retrieved, outcome, None)
    assert result is not None and result[0] == label and result[0] in fa.LABELS


def test_blocked_output_needs_manual_review() -> None:
    with pytest.raises(ValueError, match="人工复核"):
        fa.classify(["g1"], ["g1"], BLOCKED, None)
    assert fa.classify(["g1"], ["g1"], BLOCKED, ("校验误判", "忠实但被拒"))[0] == "校验误判"
    with pytest.raises(ValueError, match="人工复核"):
        fa.classify([], ["x"], BLOCKED, None)


def test_final_outcome_follows_retry_only_when_present() -> None:
    first = {**BLOCKED, "feedback_retry": _ans(["g1"])}
    assert fa.final_outcome(first, "feedback_retry")["decision"] == "answered"
    assert fa.final_outcome(first, None)["decision"] == "invalid"
    assert fa.final_outcome(ABSTAIN, "feedback_retry") is ABSTAIN


def _report() -> dict:
    def cell(retrieved: list[str], *trials: dict) -> dict:
        return {"retrieved": retrieved, "trials": list(trials)}

    rows = [
        {"sample_id": "q1", "bucket": "direct", "gold": ["g1"],
         "conditions": {"a": cell(["g1"], _ans(["g1"])), "b": cell(["x"], ABSTAIN)}},
        {"sample_id": "q2", "bucket": "multi", "gold": ["g1", "g2"],
         "conditions": {"a": cell(["g1"], ABSTAIN), "b": cell(["g1"], ABSTAIN)}},
        {"sample_id": "q3", "bucket": "near_miss", "gold": [],
         "conditions": {"a": cell(["x"], ABSTAIN), "b": cell(["x"], _ans(["x"]))}},
    ]
    return {"config": {"conditions": ["a", "b"]}, "per_query": rows}


def test_attribute_counts_every_failure_per_condition_and_dedups_queries() -> None:
    result = fa.attribute(_report())
    table = result["failures_by_condition_and_label"]
    assert table["a"]["检索缺失"] == 1 and sum(table["a"].values()) == 1  # a 只有 q2 失败
    assert table["b"]["检索缺失"] == 2 and table["b"]["充分性误判"] == 1
    assert result["failure_totals_by_condition"] == {"a": 1, "b": 3}
    assert result["retrieval_missing_partial_by_condition"] == {"a": 1, "b": 1}  # q2 的 g1 在片段里
    assert result["distinct_failing_queries"] == 3  # q1(b)、q2(a,b)、q3(b)
    assert result["queries_failing_in_all_conditions"] == ["q2"]
    assert len(result["failures"]) == sum(result["failure_totals_by_condition"].values())
