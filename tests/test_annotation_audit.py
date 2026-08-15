"""Tests for the post-run human citation-review gate.

The fixtures are synthetic only to test audit mechanics; they are never used as
medical evaluation results or included in reports/raw.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.annotation_audit import (
    AnnotationAuditError,
    audit_annotation_package,
    prepare_annotation_sample,
)
from eval.report import generate_report
from medidiag.schemas import read_jsonl, write_jsonl


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_run(tmp_path: Path, pair_count: int = 10) -> Path:
    run_dir = tmp_path / "run-001"
    run_dir.mkdir()
    knowledge_path = tmp_path / "knowledge.jsonl"
    write_jsonl(
        (
            {"chunk_id": f"chunk-{index}", "text": f"evidence {index}", "source": "public", "metadata": {}}
            for index in range(pair_count)
        ),
        knowledge_path,
    )
    _write_json(run_dir / "manifest.json", {"run_id": "run-001", "formal_candidate": True})
    _write_json(
        run_dir / "config.snapshot.json",
        {"dataset": {"knowledge_base_path": str(knowledge_path)}},
    )
    citations = [
        {
            "claim_id": f"claim-{index}",
            "claim_text": f"claim text {index}",
            "evidence_chunk_id": f"chunk-{index}",
            "verdict": "SUPPORTED" if index < 8 else "PARTIAL",
            "method": "nli",
            "model_name": "fixed-nli",
            "model_revision": "revision-1",
        }
        for index in range(pair_count)
    ]
    _write_json(
        run_dir / "rag_full.json",
        {
            "experiment": "rag_full",
            "family": "rag",
            "metrics": {},
            "samples": [{"sample_id": "sample-1", "citation_results": citations}],
        },
    )
    return run_dir


def _label_records(sample_path: Path, annotator: str, labels: list[str]) -> list[dict]:
    sample = read_jsonl(sample_path)
    return [
        {
            "annotation_id": record["annotation_id"],
            "label": label,
            "annotator_id": annotator,
            "annotated_at": "2026-07-17T12:00:00+08:00",
            "rationale": "Reviewed the emitted claim against the cited public evidence.",
            "annotation_method": "human_independent",
            "reviewer_type": "human",
            "assistance_disclosure": "none",
            "independence_attestation": True,
        }
        for record, label in zip(sample, labels, strict=True)
    ]


def _write_review_package(tmp_path: Path, sample_path: Path, *, include_adjudication: bool = True) -> tuple[Path, Path, Path]:
    # 8/2 vs 7/3 gives Kappa in [0.6, 0.8); the one disagreement must be adjudicated.
    labels_a = ["SUPPORTED"] * 8 + ["PARTIAL"] * 2
    labels_b = ["SUPPORTED"] * 7 + ["PARTIAL"] * 3
    a_path = tmp_path / "annotator-a.jsonl"
    b_path = tmp_path / "annotator-b.jsonl"
    adjudication_path = tmp_path / "adjudication.jsonl"
    write_jsonl(_label_records(sample_path, "reviewer-a", labels_a), a_path)
    write_jsonl(_label_records(sample_path, "reviewer-b", labels_b), b_path)
    sample = read_jsonl(sample_path)
    disagreement = sample[7]["annotation_id"]
    adjudications = (
        [
            {
                "annotation_id": disagreement,
                "final_label": "PARTIAL",
                "adjudicator_id": "senior-reviewer",
                "adjudicated_at": "2026-07-17T13:00:00+08:00",
                "reason": "The evidence supports only part of the claim.",
                "modified_fields": ["final_label"],
                "reviewer_type": "human",
                "assistance_disclosure": "none",
            }
        ]
        if include_adjudication
        else []
    )
    write_jsonl(adjudications, adjudication_path)
    return a_path, b_path, adjudication_path


def test_prepare_creates_immutable_20_percent_or_larger_template(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sample_path = tmp_path / "citation-sample.jsonl"

    prepared = prepare_annotation_sample(run_dir, sample_path, ratio=0.20, seed=42)

    records = read_jsonl(sample_path)
    assert prepared["selected_pair_count"] == 2
    assert prepared["all_pair_count"] == 10
    assert len(records) == 2
    assert all(record["evidence_text"].startswith("evidence") for record in records)
    assert sample_path.with_suffix(".jsonl.manifest.json").is_file()


def test_audit_requires_all_disagreements_and_writes_calibration(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sample_path = tmp_path / "citation-sample.jsonl"
    prepare_annotation_sample(run_dir, sample_path, ratio=1.0)
    a_path, b_path, adjudication_path = _write_review_package(tmp_path, sample_path)
    audit_path = run_dir / "citation_annotation_audit.json"

    audit = audit_annotation_package(
        run_dir, sample_path, a_path, b_path, adjudication_path, audit_path
    )

    assert audit["status"] == "PASSED"
    assert audit["report_eligible"] is True
    assert 0.60 <= audit["agreement"]["cohen_kappa"] < 0.80
    assert audit["agreement"]["disagreement_count"] == 1
    assert audit["agreement"]["adjudication_count"] == 1
    assert audit_path.is_file()
    assert audit["human_review_attestations"]["annotator_a"] == {
        "annotator_ids": ["reviewer-a"],
        "annotation_methods": ["human_independent"],
        "reviewer_types": ["human"],
        "assistance_disclosures": ["none"],
        "independence_attested": True,
    }


def test_audit_rejects_missing_adjudication_for_disagreement(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sample_path = tmp_path / "citation-sample.jsonl"
    prepare_annotation_sample(run_dir, sample_path, ratio=1.0)
    a_path, b_path, adjudication_path = _write_review_package(
        tmp_path, sample_path, include_adjudication=False
    )

    with pytest.raises(AnnotationAuditError, match="every and only annotator disagreement"):
        audit_annotation_package(
            run_dir, sample_path, a_path, b_path, adjudication_path, tmp_path / "audit.json"
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("annotation_method", "model_assisted_prelabel", "non-human annotation_method"),
        ("reviewer_type", "model", "non-human reviewer_type"),
        ("assistance_disclosure", "model_assisted", "disallowed assistance_disclosure"),
        ("independence_attestation", False, "must attest independent human review"),
    ],
)
def test_audit_rejects_nonhuman_or_nonindependent_labels(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    run_dir = _make_run(tmp_path)
    sample_path = tmp_path / "citation-sample.jsonl"
    prepare_annotation_sample(run_dir, sample_path, ratio=1.0)
    a_path, b_path, adjudication_path = _write_review_package(tmp_path, sample_path)
    labels_a = read_jsonl(a_path)
    labels_a[0][field] = value
    write_jsonl(labels_a, a_path)

    with pytest.raises(AnnotationAuditError, match=message):
        audit_annotation_package(
            run_dir, sample_path, a_path, b_path, adjudication_path, tmp_path / "audit.json"
        )


def test_report_refuses_missing_audit_and_accepts_matching_passing_audit(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sample_path = tmp_path / "citation-sample.jsonl"
    prepare_annotation_sample(run_dir, sample_path, ratio=1.0)
    a_path, b_path, adjudication_path = _write_review_package(tmp_path, sample_path)
    audit = audit_annotation_package(
        run_dir, sample_path, a_path, b_path, adjudication_path, run_dir / "citation_annotation_audit.json"
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"git_commit": "abc", "dirty_diff_hash": "def", "config_hash": "ghi"})
    config = {
        "judge": {"model": "fixed-nli", "revision": "revision-1", "method": "nli"},
        "dataset": {"version": "test-v1"},
    }
    results = json.loads((run_dir / "rag_full.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="human-calibration audit"):
        generate_report({"rag_full": results}, config, manifest)

    report = generate_report({"rag_full": results}, config, manifest, audit)
    assert "Human citation-review gate" in report
    assert "Cohen's Kappa" in report
