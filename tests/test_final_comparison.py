"""M15 最终对照：配置校验与汇总纯函数的离线用例。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eval.final_comparison import (
    CONDITIONS,
    comparison_table,
    condition_config,
    summarize_non_popular,
    summarize_popular,
)
from medidiag.workflow.application import load_application_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["application_v4_recommended.yaml", "application_v4_recommended_generative.yaml"])
def test_recommended_configs_pass_validation(name: str) -> None:
    config = load_application_config(ROOT / "configs" / name)
    generation = config["generation"]
    assert generation["agent"] is True and generation["agent_version"] == 2 and generation["query_router"] is True
    assert generation["on_invalid"] == "feedback"
    assert config["retrieval"]["method"] == "hybrid" and config["retrieval"]["rerank"]["candidates"] == 20
    assert generation["answer_mode"] == ("generative" if "generative" in name else "extractive")


def test_baseline_condition_uses_v3_retrieval_on_v4_corpus() -> None:
    config = condition_config(CONDITIONS[0])
    assert config["retrieval"]["method"] == "dense" and config["retrieval"]["rerank"]["candidates"] == 5
    assert config["dataset"]["version"] == "public-health-v4"
    assert "query_router" not in config["generation"] and config["generation"].get("agent_version", 1) == 1


def _row(intent: str, answerable: bool | None, classification: str, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"intent": intent, "answerable": answerable, "partial_only": False, "hit3": False,
                           "routed_out": False, "classification": classification, "answered": False, "route": None,
                           "generation_error": None, "context_precision": None}
    row.update(extra)
    return row


def test_summarize_popular_counts_router_caused_over_refusal() -> None:
    rows = [
        _row("popular_science", True, "correct", hit3=True, answered=True),
        _row("popular_science", True, "over_refusal", routed_out=True),
        _row("popular_science", True, "over_refusal"),
        _row("popular_science", False, "answered_unanswerable", answered=True, partial_only=True),
        _row("emergency", True, "correct"),  # 非科普不计入科普指标
    ]
    summary = summarize_popular(rows)
    assert summary["n"] == 4 and summary["n_answerable"] == 3
    assert summary["evidence_hit@3"]["count"] == "1/3"
    assert summary["over_refusal"]["count"] == "2/3" and summary["over_refusal_by_router"]["count"] == "1/3"
    assert summary["unanswerable_answered_partial_only"]["count"] == "1/1"


def test_summarize_non_popular_and_table() -> None:
    emergency_route = {"intent": "emergency", "refused": True}
    rows = [
        _row("emergency", True, "x", routed_out=True, route=emergency_route),
        _row("emergency", True, "x", answered=True),
        _row("out_of_scope", False, "x"),
    ]
    summary = summarize_non_popular(rows)
    assert summary["not_answered"]["count"] == "2/3" and summary["template_given"]["count"] == "1/3"
    assert summary["emergency_prompt_given"]["count"] == "1/2" and summary["emergency_answered"]["count"] == "1/2"
    table = comparison_table({CONDITIONS[0]: {"popular": summarize_popular(rows), "non_popular": summary,
                                              "cost": {"model_calls_total": 0, "input_tokens": 0, "output_tokens": 0}}})
    assert CONDITIONS[0] in table and "急症给出急症提示" in table
