"""Human citation-review sampling, agreement audit, and formal-report gate.

This module intentionally creates review templates from a completed formal run but
never fabricates labels. Two independently authored label files and, when needed,
an explicit adjudication file are required before a result can be reported.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from eval.kappa import cohen_kappa, kappa_verdict
from medidiag.schemas import read_jsonl, write_jsonl

_VALID_LABELS = {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}
_REQUIRED_LABEL_FIELDS = {"annotation_id", "label", "annotator_id", "annotated_at", "rationale"}
_REQUIRED_ADJUDICATION_FIELDS = {
    "annotation_id",
    "final_label",
    "adjudicator_id",
    "adjudicated_at",
    "reason",
    "modified_fields",
}


class AnnotationAuditError(ValueError):
    """Raised when human-review artifacts cannot support a formal report."""


@dataclass(frozen=True)
class CitationPair:
    annotation_id: str
    run_id: str
    experiment: str
    family: str
    sample_id: str
    claim_id: str
    claim_text: str
    evidence_chunk_id: str
    evidence_text: str
    judge_verdict: str
    judge_method: str
    judge_model: str
    judge_revision: str

    def to_dict(self) -> dict[str, str]:
        return {
            "annotation_id": self.annotation_id,
            "run_id": self.run_id,
            "experiment": self.experiment,
            "family": self.family,
            "sample_id": self.sample_id,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "evidence_chunk_id": self.evidence_chunk_id,
            "evidence_text": self.evidence_text,
            "judge_verdict": self.judge_verdict,
            "judge_method": self.judge_method,
            "judge_model": self.judge_model,
            "judge_revision": self.judge_revision,
        }


def prepare_annotation_sample(
    run_dir: str | Path,
    output_path: str | Path,
    *,
    ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a deterministic, verdict-stratified double-review template.

    The output contains public evidence text and judge provenance so annotators
    can assess the exact claim-citation pair. It deliberately contains no human
    labels. A sidecar manifest binds the template to the originating raw run.
    """
    if not 0 < ratio <= 1:
        raise AnnotationAuditError("ratio must be in (0, 1]")

    run_path = Path(run_dir)
    pairs, run_manifest = _load_citation_pairs(run_path)
    if not pairs:
        raise AnnotationAuditError("run contains no emitted claim-citation pairs")

    target = max(1, math.ceil(len(pairs) * ratio))
    selected = _stratified_select(pairs, target, seed)
    output = Path(output_path)
    write_jsonl((pair.to_dict() for pair in selected), output)
    sample_manifest = {
        "schema_version": 1,
        "run_id": run_manifest["run_id"],
        "source_run_manifest_hash": _sha256_file(run_path / "manifest.json"),
        "all_pair_count": len(pairs),
        "selected_pair_count": len(selected),
        "selected_ratio": len(selected) / len(pairs),
        "requested_ratio": ratio,
        "seed": seed,
        "all_pair_hash": _hash_records(pair.to_dict() for pair in pairs),
        "selected_annotation_ids_hash": _hash_values(pair.annotation_id for pair in selected),
    }
    _write_json_atomic(_sample_manifest_path(output), sample_manifest)
    return sample_manifest


def audit_annotation_package(
    run_dir: str | Path,
    sample_path: str | Path,
    annotator_a_path: str | Path,
    annotator_b_path: str | Path,
    adjudication_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate dual labels and adjudication, then write a report-gate artifact."""
    run_path = Path(run_dir)
    pairs, run_manifest = _load_citation_pairs(run_path)
    sample = _load_sample(sample_path, pairs, run_manifest, _sha256_file(run_path / "manifest.json"))
    labels_a = _load_labels(annotator_a_path, "annotator_a")
    labels_b = _load_labels(annotator_b_path, "annotator_b")
    sample_ids = {pair.annotation_id for pair in sample}
    _validate_label_coverage(labels_a, sample_ids, "annotator_a")
    _validate_label_coverage(labels_b, sample_ids, "annotator_b")
    _validate_independent_annotators(labels_a, labels_b)

    a_by_id = {record["annotation_id"]: record for record in labels_a}
    b_by_id = {record["annotation_id"]: record for record in labels_b}
    kappa, observed, expected, labels, confusion, count = cohen_kappa(
        {key: record["label"] for key, record in a_by_id.items()},
        {key: record["label"] for key, record in b_by_id.items()},
    )
    disagreements = sorted(
        annotation_id
        for annotation_id in sample_ids
        if a_by_id[annotation_id]["label"] != b_by_id[annotation_id]["label"]
    )
    adjudications = _load_adjudications(adjudication_path)
    _validate_adjudications(adjudications, disagreements)
    adjudication_by_id = {record["annotation_id"]: record for record in adjudications}
    final_labels = {
        annotation_id: (
            a_by_id[annotation_id]["label"]
            if annotation_id not in adjudication_by_id
            else adjudication_by_id[annotation_id]["final_label"]
        )
        for annotation_id in sample_ids
    }
    judge_agreement = sum(
        pair.judge_verdict == final_labels[pair.annotation_id] for pair in sample
    ) / len(sample)
    human_supported_rate = sum(label == "SUPPORTED" for label in final_labels.values()) / len(sample)
    passed = kappa >= 0.60
    audit = {
        "schema_version": 1,
        "status": "PASSED" if passed else "FAILED",
        "report_eligible": passed,
        "failure_reasons": [] if passed else ["COHEN_KAPPA_BELOW_0_60"],
        "run_id": run_manifest["run_id"],
        "source_run_manifest_hash": _sha256_file(run_path / "manifest.json"),
        "sample": {
            "path": str(Path(sample_path)),
            "hash": _sha256_file(Path(sample_path)),
            "count": len(sample),
            "population_count": len(pairs),
            "ratio": len(sample) / len(pairs),
            "annotation_ids_hash": _hash_values(pair.annotation_id for pair in sample),
        },
        "review_files": {
            "annotator_a": {"path": str(Path(annotator_a_path)), "hash": _sha256_file(Path(annotator_a_path))},
            "annotator_b": {"path": str(Path(annotator_b_path)), "hash": _sha256_file(Path(annotator_b_path))},
            "adjudication": {"path": str(Path(adjudication_path)), "hash": _sha256_file(Path(adjudication_path))},
        },
        "agreement": {
            "common_sample_count": count,
            "labels": labels,
            "observed_agreement": observed,
            "expected_agreement": expected,
            "cohen_kappa": kappa,
            "verdict": kappa_verdict(kappa),
            "confusion": confusion,
            "disagreement_count": len(disagreements),
            "adjudication_count": len(adjudications),
        },
        "calibration": {
            "judge_agreement": judge_agreement,
            "human_supported_rate": human_supported_rate,
            "label_counts": {label: sum(value == label for value in final_labels.values()) for label in sorted(_VALID_LABELS)},
        },
    }
    _write_json_atomic(Path(output_path), audit)
    return audit


def _load_citation_pairs(run_dir: Path) -> tuple[list[CitationPair], dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.snapshot.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise AnnotationAuditError("run_dir must contain manifest.json and config.snapshot.json")
    manifest = _read_json(manifest_path)
    if not manifest.get("run_id"):
        raise AnnotationAuditError("run manifest has no run_id")
    config = _read_json(config_path)
    chunk_path = _resolve_from_project(config["dataset"]["knowledge_base_path"])
    chunks = {str(item["chunk_id"]): str(item.get("text", "")) for item in read_jsonl(chunk_path)}
    pairs: list[CitationPair] = []
    seen: set[str] = set()
    for result_path in sorted(run_dir.glob("*.json")):
        if result_path.name in {"manifest.json", "config.snapshot.json", "citation_annotation_audit.json"}:
            continue
        result = _read_json(result_path)
        experiment = str(result.get("experiment", result_path.stem))
        family = str(result.get("family", ""))
        for sample in result.get("samples", []):
            sample_id = str(sample.get("sample_id", ""))
            for citation in sample.get("citation_results", []):
                claim_id = str(citation.get("claim_id", ""))
                evidence_chunk_id = str(citation.get("evidence_chunk_id", ""))
                if not claim_id:
                    raise AnnotationAuditError(f"{result_path.name} contains citation without claim_id")
                if str(citation.get("verdict", "")) not in _VALID_LABELS:
                    raise AnnotationAuditError(f"{result_path.name} contains invalid judge verdict")
                key = f"{manifest['run_id']}|{experiment}|{sample_id}|{claim_id}|{evidence_chunk_id}"
                annotation_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
                if annotation_id in seen:
                    raise AnnotationAuditError(f"duplicate claim-citation pair in raw run: {key}")
                seen.add(annotation_id)
                pairs.append(
                    CitationPair(
                        annotation_id=annotation_id,
                        run_id=str(manifest["run_id"]),
                        experiment=experiment,
                        family=family,
                        sample_id=sample_id,
                        claim_id=claim_id,
                        claim_text=str(citation.get("claim_text", "")),
                        evidence_chunk_id=evidence_chunk_id,
                        evidence_text=chunks.get(evidence_chunk_id, ""),
                        judge_verdict=str(citation.get("verdict", "")),
                        judge_method=str(citation.get("method", "")),
                        judge_model=str(citation.get("model_name", "")),
                        judge_revision=str(citation.get("model_revision", "")),
                    )
                )
    return sorted(pairs, key=lambda item: item.annotation_id), manifest


def _stratified_select(pairs: list[CitationPair], target: int, seed: int) -> list[CitationPair]:
    if target >= len(pairs):
        return list(pairs)
    buckets: dict[str, list[CitationPair]] = defaultdict(list)
    for pair in pairs:
        buckets[pair.judge_verdict].append(pair)
    quota: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    for verdict, items in buckets.items():
        exact = target * len(items) / len(pairs)
        quota[verdict] = min(len(items), math.floor(exact))
        fractions.append((exact - math.floor(exact), verdict))
    remaining = target - sum(quota.values())
    for _, verdict in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if quota[verdict] < len(buckets[verdict]):
            quota[verdict] += 1
            remaining -= 1
    selected: list[CitationPair] = []
    for verdict in sorted(buckets):
        ranked = sorted(
            buckets[verdict],
            key=lambda item: hashlib.sha256(f"{seed}|{item.annotation_id}".encode()).hexdigest(),
        )
        selected.extend(ranked[: quota[verdict]])
    return sorted(selected, key=lambda item: item.annotation_id)


def _load_sample(
    sample_path: str | Path,
    pairs: list[CitationPair],
    manifest: dict[str, Any],
    source_manifest_hash: str,
) -> list[CitationPair]:
    path = Path(sample_path)
    if not path.is_file():
        raise AnnotationAuditError(f"citation sample does not exist: {path}")
    sidecar = _sample_manifest_path(path)
    if not sidecar.is_file():
        raise AnnotationAuditError(f"citation sample manifest does not exist: {sidecar}")
    sample_manifest = _read_json(sidecar)
    records = read_jsonl(path)
    source = {pair.annotation_id: pair for pair in pairs}
    ids = [str(record.get("annotation_id", "")) for record in records]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise AnnotationAuditError("citation sample must contain non-empty unique annotation_id values")
    if any(value not in source for value in ids):
        raise AnnotationAuditError("citation sample contains pairs not found in raw run")
    for record in records:
        annotation_id = str(record["annotation_id"])
        if record != source[annotation_id].to_dict():
            raise AnnotationAuditError("citation sample content was modified after preparation")
    if sample_manifest.get("run_id") != manifest["run_id"]:
        raise AnnotationAuditError("citation sample belongs to a different run_id")
    if sample_manifest.get("source_run_manifest_hash") != source_manifest_hash:
        raise AnnotationAuditError("citation sample was created from a different run manifest")
    expected_hash = _hash_values(ids)
    if sample_manifest.get("selected_annotation_ids_hash") != expected_hash:
        raise AnnotationAuditError("citation sample manifest does not match citation sample contents")
    selected = [source[annotation_id] for annotation_id in ids]
    if sample_manifest.get("selected_pair_count") != len(selected):
        raise AnnotationAuditError("citation sample count does not match its manifest")
    if len(selected) / len(pairs) < 0.20:
        raise AnnotationAuditError("citation sample covers less than the required 20% of citation pairs")
    return selected


def _load_labels(path: str | Path, role: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise AnnotationAuditError(f"{role} file does not exist: {file_path}")
    records = read_jsonl(file_path)
    if not records:
        raise AnnotationAuditError(f"{role} file is empty")
    ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = _REQUIRED_LABEL_FIELDS - set(record)
        if missing:
            raise AnnotationAuditError(f"{role} line {index} missing fields: {sorted(missing)}")
        annotation_id = str(record["annotation_id"])
        if not annotation_id or annotation_id in ids:
            raise AnnotationAuditError(f"{role} has duplicate/empty annotation_id: {annotation_id!r}")
        ids.add(annotation_id)
        if record["label"] not in _VALID_LABELS:
            raise AnnotationAuditError(f"{role} has invalid label: {record['label']!r}")
        if not str(record["annotator_id"]).strip() or not str(record["rationale"]).strip():
            raise AnnotationAuditError(f"{role} requires non-empty annotator_id and rationale")
        _validate_timestamp(str(record["annotated_at"]), f"{role}.annotated_at")
    return records


def _validate_label_coverage(records: list[dict[str, Any]], expected_ids: set[str], role: str) -> None:
    actual_ids = {str(record["annotation_id"]) for record in records}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise AnnotationAuditError(f"{role} IDs do not exactly match citation sample; missing={missing[:3]}, extra={extra[:3]}")


def _validate_independent_annotators(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> None:
    ids_a = {str(record["annotator_id"]).strip() for record in a}
    ids_b = {str(record["annotator_id"]).strip() for record in b}
    if len(ids_a) != 1 or len(ids_b) != 1:
        raise AnnotationAuditError("each annotator file must contain exactly one annotator_id")
    if ids_a == ids_b:
        raise AnnotationAuditError("annotator_a and annotator_b must be different people")


def _load_adjudications(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise AnnotationAuditError(f"adjudication file does not exist: {file_path}")
    records = read_jsonl(file_path)
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = _REQUIRED_ADJUDICATION_FIELDS - set(record)
        if missing:
            raise AnnotationAuditError(f"adjudication line {index} missing fields: {sorted(missing)}")
        annotation_id = str(record["annotation_id"])
        if not annotation_id or annotation_id in seen:
            raise AnnotationAuditError("adjudication contains duplicate/empty annotation_id")
        seen.add(annotation_id)
        if record["final_label"] not in _VALID_LABELS:
            raise AnnotationAuditError(f"adjudication has invalid final_label: {record['final_label']!r}")
        if not str(record["adjudicator_id"]).strip() or not str(record["reason"]).strip():
            raise AnnotationAuditError("adjudication requires non-empty adjudicator_id and reason")
        if not isinstance(record["modified_fields"], list) or not record["modified_fields"]:
            raise AnnotationAuditError("adjudication.modified_fields must be a non-empty list")
        _validate_timestamp(str(record["adjudicated_at"]), "adjudication.adjudicated_at")
    return records


def _validate_adjudications(records: list[dict[str, Any]], disagreements: list[str]) -> None:
    expected = set(disagreements)
    actual = {str(record["annotation_id"]) for record in records}
    if actual != expected:
        raise AnnotationAuditError("every and only annotator disagreement requires an adjudication record")


def _validate_timestamp(value: str, field_name: str) -> None:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnnotationAuditError(f"{field_name} must be ISO-8601: {value!r}") from exc


def _sample_manifest_path(sample_path: Path) -> Path:
    return sample_path.with_suffix(sample_path.suffix + ".manifest.json")


def _resolve_from_project(configured_path: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationAuditError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnnotationAuditError(f"JSON object expected in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise AnnotationAuditError(f"file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hash_records(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in records):
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@click.group()
def cli() -> None:
    """Prepare and audit manual citation calibration artifacts."""


@cli.command("prepare")
@click.option("--run-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--ratio", type=float, default=0.20, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def prepare_command(run_dir: Path, output: Path, ratio: float, seed: int) -> None:
    """Create an unlabeled 20% double-review citation sample."""
    result = prepare_annotation_sample(run_dir, output, ratio=ratio, seed=seed)
    click.echo(f"prepared={result['selected_pair_count']} population={result['all_pair_count']} sample={output}")


@cli.command("audit")
@click.option("--run-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--sample", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--annotator-a", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--annotator-b", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--adjudication", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def audit_command(run_dir: Path, sample: Path, annotator_a: Path, annotator_b: Path, adjudication: Path, output: Path) -> None:
    """Validate double labels, Kappa, and all disagreement adjudications."""
    result = audit_annotation_package(run_dir, sample, annotator_a, annotator_b, adjudication, output)
    click.echo(f"status={result['status']} kappa={result['agreement']['cohen_kappa']:.4f} output={output}")


@cli.command("report")
@click.option("--run-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--audit", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def report_command(run_dir: Path, audit: Path, output: Path) -> None:
    """Render the formal Markdown report after the audit gate passes."""
    from eval.report import write_report

    manifest = _read_json(run_dir / "manifest.json")
    config = _read_json(run_dir / "config.snapshot.json")
    results: dict[str, dict[str, Any]] = {}
    for result_path in sorted(run_dir.glob("*.json")):
        if result_path.name in {"manifest.json", "config.snapshot.json", "citation_annotation_audit.json"}:
            continue
        result = _read_json(result_path)
        results[str(result.get("experiment", result_path.stem))] = result
    if not results:
        raise click.ClickException("run contains no experiment result JSON files")
    write_report(results, config, manifest, output, _read_json(audit))
    click.echo(f"report={output}")


if __name__ == "__main__":
    cli()
