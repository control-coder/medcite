"""Evaluation configuration loading and validation.

The YAML file is the only source of truth for experiment switches, model
identifiers, retrieval weights, dataset paths, and execution mode.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import click
import yaml

RAG_EXPERIMENTS = (
    "rag_embedding",
    "rag_bm25",
    "rag_evidence_weight",
    "rag_term_norm",
    "rag_citation_review",
    "rag_full",
)
AGENT_EXPERIMENTS = (
    "agent_single",
    "agent_fixed_pair",
    "agent_dynamic_pair",
)
EXPERIMENTS = RAG_EXPERIMENTS + AGENT_EXPERIMENTS

_RAG_SWITCHES = (
    "use_bm25",
    "use_evidence_weighting",
    "use_term_normalization",
    "use_rerank",
    "use_citation_review",
)
_UNPINNED_REVISIONS = {"", "main", "latest", "development-unpinned", "unpinned"}
_HF_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_FORMAL_PLACEHOLDER_MARKERS = ("replace", "placeholder", "todo", "example")
_GENERATION_PROVENANCE_MODES = {"provider_snapshot", "provider_response_id"}


def _is_formal_placeholder(value: object) -> bool:
    """Return whether a formal-lock value is missing or clearly non-verifiable."""
    normalized = str(value or "").strip().lower()
    return normalized in _UNPINNED_REVISIONS or any(
        marker in normalized for marker in _FORMAL_PLACEHOLDER_MARKERS
    )


def _is_full_hf_commit_sha(revision: str) -> bool:
    """Hugging Face revisions are reproducible only when pinned to a full commit."""
    return bool(_HF_COMMIT_SHA_PATTERN.fullmatch(revision))


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise click.FileError(str(path), hint="eval config not found")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise click.UsageError(f"eval config must be a mapping, got {type(config)}")
    return config


def validate_config(
    config: dict[str, Any], project_root: str | Path | None = None
) -> list[str]:
    issues: list[str] = []
    required = (
        "evaluation",
        "generation",
        "embedding",
        "rerank",
        "judge",
        "dataset",
        "retrieval",
        "experiments",
        "workflow",
        "metrics",
        "reproduction",
    )
    for key in required:
        if key not in config:
            issues.append(f"missing top-level key: {key}")
    if issues:
        return issues

    mode = config["evaluation"].get("mode")
    if mode not in {"development", "formal"}:
        issues.append("evaluation.mode must be 'development' or 'formal'")

    for section in ("generation", "embedding", "rerank", "judge"):
        model_config = config[section]
        if not model_config.get("model"):
            issues.append(f"{section}.model must be locked")
        revision = str(model_config.get("revision", ""))
        if not revision:
            issues.append(f"{section}.revision must be declared")

        if mode != "formal":
            continue
        if section == "generation":
            provenance_mode = str(
                model_config.get("provenance_mode", "provider_snapshot")
            )
            if provenance_mode not in _GENERATION_PROVENANCE_MODES:
                issues.append(
                    "generation.provenance_mode must be 'provider_snapshot' or "
                    "'provider_response_id' in formal mode"
                )
            if _is_formal_placeholder(revision):
                issues.append(
                    "generation.revision must be a declared provider model identifier in formal mode"
                )
            if provenance_mode == "provider_snapshot" and _is_formal_placeholder(
                model_config.get("snapshot_id")
            ):
                issues.append(
                    "formal evaluation requires generation.snapshot_id when "
                    "provenance_mode=provider_snapshot"
                )
            if (
                provenance_mode == "provider_response_id"
                and model_config.get("response_id_source") != "response.id"
            ):
                issues.append(
                    "formal evaluation requires generation.response_id_source=response.id "
                    "when provenance_mode=provider_response_id"
                )
        elif not _is_full_hf_commit_sha(revision):
            issues.append(
                f"{section}.revision must be a full 40-character Hugging Face commit SHA in formal mode"
            )

    generation = config["generation"]
    if not isinstance(generation.get("seed"), int):
        issues.append("generation.seed must be an integer")
    if not isinstance(generation.get("temperature"), (int, float)):
        issues.append("generation.temperature must be numeric")
    for field in ("timeout_seconds", "max_tokens"):
        if not isinstance(generation.get(field), (int, float)) or generation[field] <= 0:
            issues.append(f"generation.{field} must be positive")

    runtime = config.get("runtime", {})
    if runtime.get("device", "auto") not in {"auto", "cuda", "cpu"}:
        issues.append("runtime.device must be one of: auto, cuda, cpu")
    if not isinstance(runtime.get("batch_size"), int) or runtime["batch_size"] <= 0:
        issues.append("runtime.batch_size must be a positive integer")

    judge_method = config["judge"].get("method")
    if judge_method not in {"nli", "rule_fallback"}:
        issues.append("judge.method must be 'nli' or 'rule_fallback'")
    if mode == "formal" and judge_method != "nli":
        issues.append("formal evaluation requires judge.method=nli")

    retrieval = config["retrieval"]
    if not isinstance(retrieval.get("top_k"), int) or retrieval["top_k"] <= 0:
        issues.append("retrieval.top_k must be a positive integer")
    if not isinstance(retrieval.get("candidate_k"), int) or retrieval["candidate_k"] < retrieval.get("top_k", 0):
        issues.append("retrieval.candidate_k must be >= retrieval.top_k")
    weight_names = {
        "w1_bm25",
        "w2_embedding",
        "w3_evidence_level",
        "w4_term_overlap",
    }
    weights = retrieval.get("weights", {})
    if set(weights) != weight_names:
        issues.append(f"retrieval.weights must contain exactly {sorted(weight_names)}")
    elif any(not isinstance(value, (int, float)) or value < 0 for value in weights.values()):
        issues.append("retrieval weights must be non-negative numbers")
    elif sum(weights.values()) <= 0:
        issues.append("at least one retrieval weight must be positive")

    experiments = config["experiments"]
    rag_configs = experiments.get("rag", {})
    agent_configs = experiments.get("agent", {})
    if set(rag_configs) != set(RAG_EXPERIMENTS):
        issues.append(f"experiments.rag must contain exactly {list(RAG_EXPERIMENTS)}")
    else:
        issues.extend(_validate_rag_experiments(rag_configs))
    if set(agent_configs) != set(AGENT_EXPERIMENTS):
        issues.append(f"experiments.agent must contain exactly {list(AGENT_EXPERIMENTS)}")
    else:
        issues.extend(_validate_agent_experiments(agent_configs))

    root = Path(project_root) if project_root else Path.cwd()
    dataset = config["dataset"]
    for field in (
        "rag_eval_set_path",
        "knowledge_base_path",
        "agent_eval_set_path",
        "agent_sample_manifest_path",
    ):
        value = dataset.get(field)
        if not value:
            issues.append(f"dataset.{field} must be configured")
        elif not _resolve(root, value).is_file():
            issues.append(f"dataset.{field} does not exist: {value}")
    manifest = dataset.get("agent_sample_manifest_path")
    if manifest and _resolve(root, manifest).is_file():
        issues.extend(_validate_agent_manifest(_resolve(root, manifest)))
    if mode == "formal":
        annotation = dataset.get("annotation", {})
        if annotation.get("double_check_ratio") != 0.20:
            issues.append("formal mode requires dataset.annotation.double_check_ratio=0.20")
        # Human labels are created after the formal run from its exact emitted
        # citation pairs. Paths are still required so the post-run audit has a
        # declared destination, but requiring files here would make that run
        # impossible. report.py enforces the completed audit instead.
        for field in (
            "citation_sample_path",
            "annotator_a_path",
            "annotator_b_path",
            "adjudication_path",
        ):
            value = annotation.get(field)
            if not value:
                issues.append(f"formal mode requires configured dataset.annotation.{field}")

    return issues


def _validate_rag_experiments(experiments: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    baseline = experiments["rag_embedding"].get("config", {})
    expected_baseline = {switch: False for switch in _RAG_SWITCHES}
    if baseline != expected_baseline:
        issues.append("rag_embedding must be the pure embedding baseline")

    expected_single_switch = {
        "rag_bm25": "use_bm25",
        "rag_evidence_weight": "use_evidence_weighting",
        "rag_term_norm": "use_term_normalization",
        "rag_citation_review": "use_citation_review",
    }
    for name, changed_switch in expected_single_switch.items():
        candidate = experiments[name].get("config", {})
        if set(candidate) != set(_RAG_SWITCHES):
            issues.append(f"{name}.config must declare all RAG switches")
            continue
        differences = [key for key in _RAG_SWITCHES if candidate[key] != baseline[key]]
        if differences != [changed_switch] or candidate[changed_switch] is not True:
            issues.append(f"{name} must differ from rag_embedding only by {changed_switch}")

    full = experiments["rag_full"].get("config", {})
    if set(full) != set(_RAG_SWITCHES) or not all(full.values()):
        issues.append("rag_full must enable every RAG switch")
    return issues


def _validate_agent_experiments(experiments: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    expected_topologies = {
        "agent_single": "single",
        "agent_fixed_pair": "fixed_pair",
        "agent_dynamic_pair": "dynamic_pair",
    }
    for name, topology in expected_topologies.items():
        experiment = experiments[name]
        if experiment.get("retrieval_profile") != "rag_full":
            issues.append(f"{name} must lock retrieval_profile=rag_full")
        if experiment.get("topology") != topology:
            issues.append(f"{name}.topology must be {topology}")
    pair = experiments["agent_fixed_pair"].get("specialist_pair")
    if not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2:
        issues.append("agent_fixed_pair.specialist_pair must contain two distinct specialties")
    return issues


def get_experiment(config: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    for family in ("rag", "agent"):
        experiment = config["experiments"][family].get(name)
        if experiment is not None:
            return family, copy.deepcopy(experiment)
    raise KeyError(f"unknown experiment: {name}")


def all_experiment_names(config: dict[str, Any]) -> list[str]:
    return [
        *config["experiments"]["rag"].keys(),
        *config["experiments"]["agent"].keys(),
    ]


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _validate_agent_manifest(path: Path) -> list[str]:
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                return [f"agent sample manifest line {line_number} is invalid JSON: {exc}"]
            if not isinstance(record, dict):
                return [f"agent sample manifest line {line_number} must be an object"]
            records.append(record)
    if len(records) != 100:
        issues.append("formal agent sample manifest must contain exactly 100 records")
    sample_ids = [str(record.get("sample_id", "")) for record in records]
    if any(not sample_id for sample_id in sample_ids):
        issues.append("every agent sample manifest record requires sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        issues.append("agent sample manifest sample_id values must be unique")
    for field in ("source", "original_id", "inclusion_rule", "dataset_version"):
        if any(not record.get(field) for record in records):
            issues.append(f"every agent sample manifest record requires {field}")
    return issues
