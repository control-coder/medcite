"""Generate formal evaluation reports only after the human citation audit passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _metric(result: Any, name: str) -> float | None:
    if isinstance(result, dict):
        return result.get("metrics", {}).get(name)
    value = getattr(result, name, None)
    return value() if callable(value) else value


def _table(results: dict[str, Any]) -> str:
    lines = [
        "| Experiment | Family | Recall@5 | Gold coverage | Citation precision | Unsupported claims | Pipeline approval | P95(ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        family = result.get("family", "") if isinstance(result, dict) else result.family
        values = [
            _metric(result, "evidence_recall_at_5"),
            _metric(result, "gold_evidence_coverage"),
            _metric(result, "citation_precision"),
            _metric(result, "unsupported_claim_rate"),
            _metric(result, "pipeline_approval_rate"),
            _metric(result, "p95_latency_ms"),
        ]
        formatted = ["N/A" if value is None else f"{value:.4f}" for value in values]
        lines.append(f"| {name} | {family} | " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def _validate_audit(manifest: dict[str, Any], audit: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest.get("formal_candidate"):
        reasons = ", ".join(manifest.get("non_reportable_reasons", []))
        raise ValueError(f"refusing to generate formal report from non-formal run: {reasons}")
    if not audit:
        raise ValueError("refusing to generate formal report before citation human-calibration audit")
    if audit.get("status") != "PASSED" or not audit.get("report_eligible"):
        reasons = ", ".join(audit.get("failure_reasons", []))
        raise ValueError(f"refusing to generate formal report from failed citation audit: {reasons}")
    if audit.get("run_id") != manifest.get("run_id"):
        raise ValueError("citation audit run_id does not match raw-result manifest")
    return audit


def generate_report(
    results: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    annotation_audit: dict[str, Any] | None = None,
) -> str:
    audit = _validate_audit(manifest, annotation_audit)
    agreement = audit["agreement"]
    calibration = audit["calibration"]
    return "\n".join(
        [
            "# MediDiag Formal Evaluation Report",
            "",
            "> Generated from immutable raw results and a passing independent human citation-review audit. Do not edit metrics manually.",
            "",
            "## Run provenance",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- git_commit: `{manifest['git_commit']}`",
            f"- dirty_diff_hash: `{manifest['dirty_diff_hash']}`",
            f"- config_hash: `{manifest['config_hash']}`",
            f"- judge: `{config['judge']['model']}@{config['judge']['revision']}` (`{config['judge']['method']}`)",
            f"- dataset_version: `{config['dataset']['version']}`",
            "",
            "## Human citation-review gate",
            "",
            f"- independently double-labeled pairs: `{audit['sample']['count']}/{audit['sample']['population_count']}` ({audit['sample']['ratio']:.2%})",
            f"- Cohen's Kappa: `{agreement['cohen_kappa']:.4f}`; observed agreement: `{agreement['observed_agreement']:.4f}`",
            f"- Kappa disposition: {agreement['verdict']}",
            f"- disagreements/adjudications: `{agreement['disagreement_count']}/{agreement['adjudication_count']}`",
            f"- fixed NLI judge agreement with adjudicated human labels: `{calibration['judge_agreement']:.4f}`",
            "",
            "## Results",
            "",
            _table(results),
            "",
            "`workflow_success_rate` is intentionally omitted here until the API/worker runner records persisted `CLOSED_SUCCESS` terminal states.",
            "",
            "## Metric denominators",
            "",
            "- Evidence Recall@5: eligible samples with at least one gold evidence hit / samples with non-empty `gold_evidence_ids`.",
            "- Gold Evidence Coverage: samples with non-empty `gold_evidence_ids` / all samples.",
            "- Citation Precision: supported emitted claim-citation pairs / all emitted claim-citation pairs, judged by the fixed NLI model; the human-review section reports its calibration against adjudicated labels.",
            "- Unsupported Claim Rate: claims whose best citation verdict is unsupported / all claims.",
        ]
    )


def write_report(
    results: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    output_path: Path,
    annotation_audit: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_report(results, config, manifest, annotation_audit), encoding="utf-8"
    )
