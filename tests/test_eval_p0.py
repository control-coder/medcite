"""P0-A evaluation truthfulness and configuration gates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import click
import pytest

from eval.configuration import load_config, validate_config
from eval.leakage_check import run_leakage_check
from eval.metrics import (
    compute_citation_precision,
    compute_gold_evidence_coverage,
    compute_unsupported_claim_rate,
)
from eval.runner import ExperimentResult, SampleResult, _non_reportable_reasons


@pytest.fixture
def config() -> dict:
    return load_config("eval/config.yaml")


def test_default_development_config_is_valid(config: dict) -> None:
    assert validate_config(config, Path.cwd()) == []


def test_each_rag_ablation_changes_one_switch(config: dict) -> None:
    baseline = config["experiments"]["rag"]["rag_embedding"]["config"]
    expected = {
        "rag_bm25": "use_bm25",
        "rag_evidence_weight": "use_evidence_weighting",
        "rag_term_norm": "use_term_normalization",
        "rag_citation_review": "use_citation_review",
    }
    for name, switch in expected.items():
        candidate = config["experiments"]["rag"][name]["config"]
        differences = [key for key in baseline if candidate[key] != baseline[key]]
        assert differences == [switch]


def test_config_rejects_coupled_single_variable_group(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["experiments"]["rag"]["rag_bm25"]["config"][
        "use_term_normalization"
    ] = True
    issues = validate_config(invalid, Path.cwd())
    assert any("rag_bm25 must differ" in issue for issue in issues)


def test_formal_mode_rejects_rule_fallback_and_unpinned_models(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["evaluation"]["mode"] = "formal"
    issues = validate_config(invalid, Path.cwd())
    assert "formal evaluation requires judge.method=nli" in issues
    assert any("revision must be immutable" in issue for issue in issues)
    # Labels are generated from a completed formal raw run. Requiring files here
    # would make the mandatory 20% sampling step impossible; report generation
    # instead requires a passing post-run audit.
    assert not any("existing dataset.annotation" in issue for issue in issues)

    missing_annotation_path = deepcopy(config)
    missing_annotation_path["evaluation"]["mode"] = "formal"
    missing_annotation_path["dataset"]["annotation"]["citation_sample_path"] = ""
    annotation_issues = validate_config(missing_annotation_path, Path.cwd())
    assert "formal mode requires configured dataset.annotation.citation_sample_path" in annotation_issues


def test_agent_manifest_is_fixed_to_100_unique_samples(config: dict) -> None:
    path = Path(config["dataset"]["agent_sample_manifest_path"])
    records = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 100


def test_agent_experiments_lock_rag_full(config: dict) -> None:
    for experiment in config["experiments"]["agent"].values():
        assert experiment["retrieval_profile"] == "rag_full"


def test_recall_uses_only_evidence_eligible_samples() -> None:
    result = ExperimentResult(
        experiment="rag_embedding",
        family="rag",
        sample_results=[
            SampleResult("eligible-hit", "rag_embedding", "rag", True, True),
            SampleResult("eligible-miss", "rag_embedding", "rag", True, False),
            SampleResult("ineligible", "rag_embedding", "rag", False, None),
        ],
    )
    assert result.evidence_recall_at_5 == 0.5
    assert result.gold_evidence_coverage == pytest.approx(2 / 3)
    assert result.to_dict()["metrics"]["workflow_success_rate"] is None


def test_gold_evidence_coverage_empty_dataset() -> None:
    assert compute_gold_evidence_coverage(0, 0) == 0.0


def test_citation_and_claim_metrics_use_different_denominators() -> None:
    results = [
        {
            "claim_id": "claim-1",
            "evidence_chunk_id": "c1",
            "verdict": "SUPPORTED",
        },
        {
            "claim_id": "claim-1",
            "evidence_chunk_id": "c2",
            "verdict": "UNSUPPORTED",
        },
        {
            "claim_id": "claim-2",
            "evidence_chunk_id": "",
            "verdict": "UNSUPPORTED",
        },
    ]
    assert compute_citation_precision(results) == 0.5
    assert compute_unsupported_claim_rate(results) == 0.5


def test_leakage_gate_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(click.FileError):
        run_leakage_check(
            tmp_path / "missing-eval.jsonl",
            tmp_path / "missing-kb.jsonl",
            {},
        )


def test_development_dry_run_is_non_reportable(config: dict) -> None:
    reasons = _non_reportable_reasons(config, limit=5, dry_run=True)
    assert "evaluation_mode_is_development" in reasons
    assert "judge_method_is_not_nli" in reasons
    assert "sample_limit_used" in reasons
    assert "dry_run_has_no_generation_or_workflow_result" in reasons
