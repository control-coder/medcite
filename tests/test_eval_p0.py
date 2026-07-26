"""P0-A evaluation truthfulness and configuration gates."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import click
import pytest

from eval.configuration import load_config, validate_config
from eval.leakage_check import run_leakage_check
from eval.metrics import (
    compute_citation_precision,
    compute_gold_evidence_coverage,
    compute_recall_at_k,
    compute_unsupported_claim_rate,
)
from eval.runner import (
    AgentOutputCache,
    ExperimentResult,
    SampleResult,
    _assert_formal_response_id_provenance,
    _build_manifest,
    _non_reportable_reasons,
    select_experiments,
)


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


def test_rag_retrieval_selection_excludes_generation_groups(config: dict) -> None:
    """纯检索选择只包含不需要 generation/judge 的单变量组。"""
    names = select_experiments(config, "rag_retrieval")
    assert names == [
        "rag_embedding",
        "rag_bm25",
        "rag_evidence_weight",
        "rag_term_norm",
    ]
    assert all(
        not config["experiments"]["rag"][name]["config"]["use_citation_review"]
        for name in names
    )

def test_config_rejects_coupled_single_variable_group(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["experiments"]["rag"]["rag_bm25"]["config"][
        "use_term_normalization"
    ] = True
    issues = validate_config(invalid, Path.cwd())
    assert any("rag_bm25 must differ" in issue for issue in issues)


def _valid_formal_config(config: dict) -> dict:
    """构造具有可核验 snapshot 的合法 formal 配置。"""
    formal = deepcopy(config)
    formal["evaluation"]["mode"] = "formal"
    formal["generation"].update(
        {
            "revision": "deepseek-v4-flash",
            "provenance_mode": "provider_snapshot",
            "snapshot_id": "provider-system-fingerprint-20260717-a1b2c3",
        }
    )
    formal["generation"].pop("response_id_source", None)
    for section in ("embedding", "rerank", "judge"):
        formal[section]["revision"] = "a" * 40
    formal["judge"]["method"] = "nli"
    return formal


def _valid_response_id_formal_config(config: dict) -> dict:
    """构造使用响应 ID 且不声明 snapshot 的合法 formal 配置。"""
    formal = _valid_formal_config(config)
    formal["generation"].update(
        {
            "revision": "deepseek-v4-flash",
            "provenance_mode": "provider_response_id",
            "response_id_source": "response.id",
        }
    )
    formal["generation"].pop("snapshot_id", None)
    return formal


def test_formal_mode_rejects_rule_fallback_and_unverifiable_model_locks(
    config: dict,
) -> None:
    invalid = deepcopy(config)
    invalid["evaluation"]["mode"] = "formal"
    issues = validate_config(invalid, Path.cwd())
    assert "formal evaluation requires judge.method=nli" in issues
    assert (
        "generation.revision must be a declared provider model identifier in formal mode"
        in issues
    )
    assert (
        "formal evaluation requires generation.snapshot_id when "
        "provenance_mode=provider_snapshot"
        in issues
    )
    for section in ("embedding", "rerank", "judge"):
        assert not any(
            issue.startswith(f"{section}.revision must be") for issue in issues
        )
    # 现有 dataset annotation 不能替代 formal raw run 后实际输出的 20% 人工复核。
    assert not any("existing dataset.annotation" in issue for issue in issues)

    missing_annotation_path = deepcopy(config)
    missing_annotation_path["evaluation"]["mode"] = "formal"
    missing_annotation_path["dataset"]["annotation"]["citation_sample_path"] = ""
    annotation_issues = validate_config(missing_annotation_path, Path.cwd())
    assert "formal mode requires configured dataset.annotation.citation_sample_path" in annotation_issues


def test_formal_mode_accepts_full_hf_commits_and_provider_snapshot(config: dict) -> None:
    assert validate_config(_valid_formal_config(config), Path.cwd()) == []


def test_formal_mode_accepts_response_id_provenance_without_snapshot(config: dict) -> None:
    assert validate_config(_valid_response_id_formal_config(config), Path.cwd()) == []


def test_formal_mode_rejects_short_sha_and_placeholder_snapshot(config: dict) -> None:
    invalid = _valid_formal_config(config)
    invalid["embedding"]["revision"] = "a" * 12
    invalid["generation"]["snapshot_id"] = "REPLACE_WITH_PROVIDER_SNAPSHOT"

    issues = validate_config(invalid, Path.cwd())

    assert (
        "embedding.revision must be a full 40-character Hugging Face commit SHA "
        "in formal mode"
    ) in issues
    assert (
        "formal evaluation requires generation.snapshot_id when "
        "provenance_mode=provider_snapshot"
        in issues
    )


def test_formal_mode_rejects_invalid_response_id_source(config: dict) -> None:
    invalid = _valid_response_id_formal_config(config)
    invalid["generation"]["response_id_source"] = "header.x-request-id"

    assert (
        "formal evaluation requires generation.response_id_source=response.id "
        "when provenance_mode=provider_response_id"
    ) in validate_config(invalid, Path.cwd())


def test_formal_template_is_runnable_with_response_id_provenance() -> None:
    template = load_config("eval/config.formal.template.yaml")
    assert validate_config(template, Path.cwd()) == []


def test_formal_response_id_provenance_requires_sample_request_id(config: dict) -> None:
    formal = _valid_response_id_formal_config(config)
    missing_id = SampleResult("sample-1", "agent_single", "agent", True)

    with pytest.raises(click.ClickException, match="FORMAL_GENERATION_RESPONSE_ID_MISSING"):
        _assert_formal_response_id_provenance(formal, missing_id)

    recorded_id = SampleResult(
        "sample-1",
        "agent_single",
        "agent",
        True,
        provider_request_ids=["chatcmpl-test-id"],
    )
    _assert_formal_response_id_provenance(formal, recorded_id)


def test_formal_manifest_records_generation_provenance(config: dict) -> None:
    formal = _valid_response_id_formal_config(config)
    now = datetime.now(UTC)

    manifest = _build_manifest(
        run_id="run-with-model-lock",
        config=formal,
        experiments=["rag_embedding"],
        started_at=now,
        finished_at=now,
        limit=None,
        dry_run=False,
        cache=AgentOutputCache(),
        git_commit="test-commit",
        dirty_diff_hash="test-diff",
    )

    assert manifest["models"]["generation"] == {
        "model": "deepseek-v4-flash",
        "revision": "deepseek-v4-flash",
        "provenance_mode": "provider_response_id",
        "response_id_source": "response.id",
    }


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
    """空分母的指标是"未定义"，必须与实测 0.0 区分开。"""
    assert compute_gold_evidence_coverage(0, 0) is None
    assert compute_gold_evidence_coverage(0, 4) == 0.0


def test_undefined_metrics_are_null_not_zero() -> None:
    """MedQA 这类样本没有 gold evidence，Recall 未定义而非"检索全失败"。"""
    assert compute_recall_at_k([]) is None
    assert compute_recall_at_k([False, False]) == 0.0
    assert compute_citation_precision([]) is None
    assert compute_citation_precision(
        [{"claim_id": "c", "evidence_chunk_id": "", "verdict": "UNSUPPORTED"}]
    ) is None

    result = ExperimentResult(experiment="agent_only", family="agent", sample_results=[])
    payload = result.to_dict()["metrics"]
    assert payload["evidence_recall_at_5"] is None
    assert payload["gold_evidence_coverage"] is None
    assert payload["citation_precision"] is None


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
