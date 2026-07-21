"""Configuration-driven MediDiag evaluation runner.

RAG and Agent experiments use separate namespaces. Development runs are
explicitly non-reportable when they use a limit, dry-run, unpinned models, or
the rule fallback judge.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from eval import metrics
from eval.configuration import (
    EXPERIMENTS,
    all_experiment_names,
    get_experiment,
    load_config,
    validate_config,
)
from eval.leakage_check import LEAKAGE_FLAG, run_leakage_check
from medidiag.agents.arbitration import ArbitrationAgent
from medidiag.agents.base import AgentOutput
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.llm_client import LLMClient
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialist import SpecialistAgent
from medidiag.compliance.guard import ComplianceGuard
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever
from medidiag.review.citation import CitationVerifier
from medidiag.review.logic import ClinicalLogicReviewer
from medidiag.schemas import KnowledgeChunk, read_jsonl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_SELECTIONS = (*EXPERIMENTS, "rag_all", "agent_all", "all")


@dataclass
class AgentOutputCache:
    _cache: dict[str, AgentOutput] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def compute_hash(
        self, question: str, evidence_ids: list[str], specialty: str
    ) -> str:
        payload = f"{question}|{'-'.join(sorted(evidence_ids))}|{specialty}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, input_hash: str) -> AgentOutput | None:
        value = self._cache.get(input_hash)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, input_hash: str, output: AgentOutput) -> None:
        self._cache[input_hash] = output

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class SampleResult:
    sample_id: str
    experiment: str
    family: str
    evidence_eligible: bool
    recall_hit: bool | None = None
    citation_results: list[dict[str, Any]] = field(default_factory=list)
    total_claims: int = 0
    pipeline_approved: bool | None = None
    latency_ms: float = 0.0
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    routing_fallback: bool = False
    routing_confidence: float = 0.0
    specialty_pair: list[str] = field(default_factory=list)
    arbitration_verdict: str = ""
    compliance_blocked: bool = False
    agent_abstained: bool = False
    cache_hit: bool = False
    provider_request_ids: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    experiment: str
    family: str
    sample_results: list[SampleResult] = field(default_factory=list)
    cache_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_recall_at_5(self) -> float:
        eligible = [
            result.recall_hit
            for result in self.sample_results
            if result.evidence_eligible and result.recall_hit is not None
        ]
        return metrics.compute_recall_at_k(eligible)

    @property
    def gold_evidence_coverage(self) -> float:
        eligible = sum(result.evidence_eligible for result in self.sample_results)
        return metrics.compute_gold_evidence_coverage(eligible, len(self.sample_results))

    @property
    def citation_precision(self) -> float:
        records = [
            record for result in self.sample_results for record in result.citation_results
        ]
        return metrics.compute_citation_precision(records)

    @property
    def unsupported_claim_rate(self) -> float:
        records = [
            record for result in self.sample_results for record in result.citation_results
        ]
        return metrics.compute_unsupported_claim_rate(records)

    @property
    def pipeline_approval_rate(self) -> float | None:
        values = [
            result.pipeline_approved
            for result in self.sample_results
            if result.pipeline_approved is not None
        ]
        return metrics.compute_workflow_success_rate(values) if values else None

    @property
    def p95_latency_ms(self) -> float:
        return metrics.compute_p95_latency(
            [result.latency_ms for result in self.sample_results]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "family": self.family,
            "sample_count": len(self.sample_results),
            "metrics": {
                "evidence_recall_at_5": round(self.evidence_recall_at_5, 4),
                "gold_evidence_coverage": round(self.gold_evidence_coverage, 4),
                "citation_precision": round(self.citation_precision, 4),
                "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
                # This is a reviewer-pipeline metric, not CLOSED_SUCCESS.
                "pipeline_approval_rate": (
                    round(self.pipeline_approval_rate, 4)
                    if self.pipeline_approval_rate is not None
                    else None
                ),
                "workflow_success_rate": None,
                "p95_latency_ms": round(self.p95_latency_ms, 2),
            },
            "cache_stats": self.cache_stats,
            "samples": [asdict(result) for result in self.sample_results],
        }


def show_config_summary(config: dict[str, Any]) -> None:
    click.echo("MediDiag Eval Configuration")
    click.echo(f"  mode             : {config['evaluation']['mode']}")
    for section in ("generation", "embedding", "rerank", "judge"):
        value = config[section]
        click.echo(f"  {section:<17}: {value['model']}@{value['revision']}")
    click.echo(f"  judge_method     : {config['judge']['method']}")
    click.echo(f"  experiments      : {', '.join(all_experiment_names(config))}")


def select_experiments(config: dict[str, Any], selection: str) -> list[str]:
    if selection == "rag_all":
        return list(config["experiments"]["rag"])
    if selection == "agent_all":
        return list(config["experiments"]["agent"])
    if selection == "all":
        return all_experiment_names(config)
    return [selection]


def run_evaluation(
    config: dict[str, Any],
    experiment_names: list[str],
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[str, dict[str, ExperimentResult]]:
    started_at = datetime.now(UTC)
    mode = config["evaluation"]["mode"]
    if mode == "formal" and (limit is not None or dry_run):
        raise click.UsageError("formal evaluation forbids --limit and --dry-run")

    _run_leakage_gates(config, experiment_names)
    kb_path = _resolve(config["dataset"]["knowledge_base_path"])
    chunks = [KnowledgeChunk(**record) for record in read_jsonl(kb_path)]
    normalizer = TerminologyNormalizer()
    retriever = Retriever(
        chunks,
        weights=config["retrieval"]["weights"],
        evidence_level_scores=config["retrieval"]["evidence_levels"],
        embedding_model=config["embedding"]["model"],
        rerank_model=config["rerank"]["model"],
        normalizer=normalizer,
        embedding_revision=config["embedding"]["revision"],
        rerank_revision=config["rerank"]["revision"],
    )
    retriever.build_index(use_bm25=True, use_embedding=True)

    verifier: CitationVerifier | None = None
    llm: LLMClient | None = None
    if not dry_run:
        verifier = CitationVerifier(
            model_name=config["judge"]["model"],
            model_revision=config["judge"]["revision"],
            method=config["judge"]["method"],
        )
        verifier.initialize()
        generation = config["generation"]
        llm = LLMClient(
            base_url=generation["base_url"],
            model=generation["model"],
            timeout=int(generation["timeout_seconds"]),
            temperature=float(generation["temperature"]),
            max_tokens=int(generation["max_tokens"]),
            seed=int(generation["seed"]),
            require_request_id=(
                config["evaluation"]["mode"] == "formal"
                and generation.get("provenance_mode") == "provider_response_id"
            ),
        )

    cache = AgentOutputCache()
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    dirty_diff_hash = _git_diff_hash()
    run_id = _new_run_id(config)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.snapshot.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    results: dict[str, ExperimentResult] = {}
    for name in experiment_names:
        family, experiment = get_experiment(config, name)
        records = _load_experiment_records(config, family, limit)
        result = _run_experiment(
            name,
            family,
            experiment,
            records,
            config,
            retriever,
            normalizer,
            verifier,
            llm,
            cache,
            dry_run,
        )
        results[name] = result
        (run_dir / f"{name}.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(
            f"{name}: Recall@5={result.evidence_recall_at_5:.4f}, "
            f"GoldCoverage={result.gold_evidence_coverage:.4f}"
        )

    manifest = _build_manifest(
        run_id,
        config,
        experiment_names,
        started_at,
        datetime.now(UTC),
        limit,
        dry_run,
        cache,
        git_commit,
        dirty_diff_hash,
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_id, results


def _run_leakage_gates(config: dict[str, Any], names: list[str]) -> None:
    families = {get_experiment(config, name)[0] for name in names}
    dataset = config["dataset"]
    for family in families:
        eval_path = _resolve(
            dataset["rag_eval_set_path"]
            if family == "rag"
            else dataset["agent_eval_set_path"]
        )
        kb_path = _resolve(dataset["knowledge_base_path"])
        hits = run_leakage_check(eval_path, kb_path, config["leakage_check"])
        if hits:
            detail = "\n".join(f"  - {hit}" for hit in hits[:20])
            raise click.ClickException(f"{LEAKAGE_FLAG}\n{detail}")


def _load_experiment_records(
    config: dict[str, Any], family: str, limit: int | None
) -> list[dict[str, Any]]:
    field = "rag_eval_set_path" if family == "rag" else "agent_eval_set_path"
    records = read_jsonl(_resolve(config["dataset"][field]))
    if family == "agent":
        by_id = {str(record["sample_id"]): record for record in records}
        manifest = read_jsonl(
            _resolve(config["dataset"]["agent_sample_manifest_path"])
        )
        missing = [
            str(item["sample_id"])
            for item in manifest
            if str(item["sample_id"]) not in by_id
        ]
        if missing:
            raise click.ClickException(
                "agent sample manifest references missing samples: "
                + ", ".join(missing[:10])
            )
        records = [by_id[str(item["sample_id"])] for item in manifest]
    return records[:limit] if limit is not None else records


def _run_experiment(
    name: str,
    family: str,
    experiment: dict[str, Any],
    records: list[dict[str, Any]],
    config: dict[str, Any],
    retriever: Retriever,
    normalizer: TerminologyNormalizer,
    verifier: CitationVerifier | None,
    llm: LLMClient | None,
    cache: AgentOutputCache,
    dry_run: bool,
) -> ExperimentResult:
    result = ExperimentResult(experiment=name, family=family)
    rag_config = (
        experiment["config"]
        if family == "rag"
        else config["experiments"]["rag"][experiment["retrieval_profile"]]["config"]
    )
    topology = "single" if family == "rag" else experiment["topology"]
    for record in records:
        result.sample_results.append(
            _run_sample(
                name,
                family,
                topology,
                experiment,
                rag_config,
                record,
                config,
                retriever,
                normalizer,
                verifier,
                llm,
                cache,
                dry_run,
            )
        )
    result.cache_stats = {
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": round(cache.hit_rate, 4),
    }
    return result


def _assert_formal_response_id_provenance(
    config: dict[str, Any], result: SampleResult
) -> None:
    """???? formal ????????????????? ID?"""
    generation = config["generation"]
    if (
        config["evaluation"]["mode"] == "formal"
        and generation.get("provenance_mode", "provider_snapshot")
        == "provider_response_id"
        and not result.provider_request_ids
    ):
        raise click.ClickException(
            "FORMAL_GENERATION_RESPONSE_ID_MISSING: "
            "provider_response_id mode requires response.id for every generated sample"
        )


def _run_sample(
    name: str,
    family: str,
    topology: str,
    experiment: dict[str, Any],
    rag_config: dict[str, bool],
    sample: dict[str, Any],
    config: dict[str, Any],
    retriever: Retriever,
    normalizer: TerminologyNormalizer,
    verifier: CitationVerifier | None,
    llm: LLMClient | None,
    cache: AgentOutputCache,
    dry_run: bool,
) -> SampleResult:
    started = time.perf_counter()
    gold_ids = [str(value) for value in sample.get("gold_evidence_ids", [])]
    result = SampleResult(
        sample_id=str(sample["sample_id"]),
        experiment=name,
        family=family,
        evidence_eligible=bool(gold_ids),
    )
    question = str(sample["question"])
    top_k = int(config["retrieval"]["top_k"])
    candidate_k = int(config["retrieval"]["candidate_k"])

    retrieval_query = question
    if rag_config["use_term_normalization"]:
        normalize_started = time.perf_counter()
        retrieval_query = normalizer.normalize(question).normalized
        result.stage_latency_ms["normalize"] = _elapsed_ms(normalize_started)
    retrieval_started = time.perf_counter()
    search_results = retriever.search(
        retrieval_query,
        top_k=candidate_k if rag_config["use_rerank"] else top_k,
        experiment_config=rag_config,
    )
    result.stage_latency_ms["retrieval"] = _elapsed_ms(retrieval_started)
    if rag_config["use_rerank"]:
        rerank_started = time.perf_counter()
        search_results = retriever.rerank(retrieval_query, search_results, top_k=top_k)
        result.stage_latency_ms["rerank"] = _elapsed_ms(rerank_started)

    evidence_ids = [item.chunk_id for item in search_results]
    evidence = [item.chunk for item in search_results if item.chunk is not None]
    result.recall_hit = bool(set(evidence_ids) & set(gold_ids)) if gold_ids else None
    if dry_run:
        result.latency_ms = _elapsed_ms(started)
        return result

    generation_started = time.perf_counter()
    outputs = _generate_outputs(
        topology,
        experiment,
        question,
        evidence,
        evidence_ids,
        sample.get("options"),
        normalizer,
        config["retrieval"]["evidence_levels"],
        llm,
        cache,
        result,
    )
    result.stage_latency_ms["generation"] = _elapsed_ms(generation_started)
    result.agent_abstained = any(output.abstain for output in outputs)
    result.provider_request_ids = sorted(
        {
            output.provider_request_id
            for output in outputs
            if output.provider_request_id
        }
    )
    _assert_formal_response_id_provenance(config, result)

    review_started = time.perf_counter()
    reviewer = ClinicalLogicReviewer()
    review_approved = all(reviewer.check(output).is_approved for output in outputs)
    result.stage_latency_ms["review"] = _elapsed_ms(review_started)

    claims: list[dict[str, Any]] = []
    for output_index, output in enumerate(outputs):
        for claim_index, claim in enumerate(output.claims):
            claims.append(
                {
                    "claim_id": (
                        f"{result.sample_id}:{output_index:02d}_{claim_index:04d}"
                    ),
                    "text": claim.text,
                    "citation_chunk_ids": claim.citation_chunk_ids,
                }
            )
    judge_started = time.perf_counter()
    citation_results = verifier.verify_batch(claims, evidence) if verifier else []
    result.stage_latency_ms["judge"] = _elapsed_ms(judge_started)
    result.citation_results = [item.to_dict() for item in citation_results]
    result.total_claims = len(claims)

    guard = ComplianceGuard()
    result.compliance_blocked = any(
        guard.check_output(output.to_dict()).blocked for output in outputs
    )
    unsupported_rate = metrics.compute_unsupported_claim_rate(citation_results)
    citation_gate = (
        not rag_config["use_citation_review"]
        or unsupported_rate <= config["workflow"]["review"]["unsupported_claim_threshold"]
    )
    result.pipeline_approved = (
        review_approved
        and citation_gate
        and not result.agent_abstained
        and not result.compliance_blocked
    )
    result.latency_ms = _elapsed_ms(started)
    return result


def _generate_outputs(
    topology: str,
    experiment: dict[str, Any],
    question: str,
    evidence: list[KnowledgeChunk],
    evidence_ids: list[str],
    options: dict[str, str] | None,
    normalizer: TerminologyNormalizer,
    evidence_level_scores: dict[str, float],
    llm: LLMClient | None,
    cache: AgentOutputCache,
    sample_result: SampleResult,
) -> list[AgentOutput]:
    if topology == "single":
        specialties = ["general_diagnosis"]
    elif topology == "fixed_pair":
        specialties = list(experiment["specialist_pair"])
        sample_result.routing_confidence = 1.0
    else:
        routing = SpecialistRouter(
            normalizer=normalizer,
            evidence_level_scores=evidence_level_scores,
        ).route(question, evidence, "")
        specialties = list(routing.specialty_pair)
        sample_result.routing_fallback = routing.is_fallback
        sample_result.routing_confidence = routing.confidence
    sample_result.specialty_pair = specialties

    outputs: list[AgentOutput] = []
    for specialty in specialties:
        input_hash = cache.compute_hash(question, evidence_ids, specialty)
        cached = cache.get(input_hash)
        if cached is not None:
            outputs.append(cached)
            sample_result.cache_hit = True
            continue
        agent = (
            DiagnosisAgent(llm)
            if specialty == "general_diagnosis"
            else SpecialistAgent(specialty, llm)
        )
        output = agent.generate(question, evidence, "", options)
        cache.set(input_hash, output)
        outputs.append(output)

    if len(outputs) == 2 and not any(output.abstain for output in outputs):
        sample_result.arbitration_verdict = ArbitrationAgent().arbitrate(
            outputs[0], outputs[1]
        ).verdict
    elif len(outputs) == 2:
        sample_result.arbitration_verdict = "ESCALATED"
    return outputs


def _build_manifest(
    run_id: str,
    config: dict[str, Any],
    experiments: list[str],
    started_at: datetime,
    finished_at: datetime,
    limit: int | None,
    dry_run: bool,
    cache: AgentOutputCache,
    git_commit: str,
    dirty_diff_hash: str,
) -> dict[str, Any]:
    dataset = config["dataset"]
    formal_candidate = (
        config["evaluation"]["mode"] == "formal"
        and limit is None
        and not dry_run
        and config["judge"]["method"] == "nli"
    )
    return {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "command": " ".join(sys.argv),
        # A completed formal run is only a candidate. It becomes reportable after
        # eval.annotation_audit validates a 20% independent human-review sample.
        "formal_candidate": formal_candidate,
        "report_eligible": False,
        "non_reportable_reasons": _non_reportable_reasons(config, limit, dry_run),
        "pending_report_gate": (
            "CITATION_HUMAN_CALIBRATION_AUDIT_REQUIRED" if formal_candidate else None
        ),
        "experiments": experiments,
        "git_commit": git_commit,
        "dirty_diff_hash": dirty_diff_hash,
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "dataset_hashes": {
            key: _hash_file(_resolve(dataset[key]))
            for key in (
                "rag_eval_set_path",
                "agent_eval_set_path",
                "agent_sample_manifest_path",
                "knowledge_base_path",
            )
        },
        "python_version": sys.version,
        "dependency_lock_hash": _hash_file(_PROJECT_ROOT / "requirements.txt"),
        "models": {
            section: {
                "model": config[section]["model"],
                "revision": config[section]["revision"],
                **(
                    {
                        "provenance_mode": config[section].get(
                            "provenance_mode", "provider_snapshot"
                        ),
                        **(
                            {"snapshot_id": config[section].get("snapshot_id")}
                            if config[section].get("snapshot_id")
                            else {}
                        ),
                        **(
                            {
                                "response_id_source": config[section].get(
                                    "response_id_source"
                                )
                            }
                            if config[section].get("response_id_source")
                            else {}
                        ),
                    }
                    if section == "generation"
                    else {}
                ),
            }
            for section in ("generation", "embedding", "rerank", "judge")
        },
        "generation": {
            "temperature": config["generation"]["temperature"],
            "seed": config["generation"]["seed"],
            "timeout_seconds": config["generation"]["timeout_seconds"],
        },
        "retrieval": config["retrieval"],
        "judge_method": config["judge"]["method"],
        "limit": limit,
        "dry_run": dry_run,
        "cache": {"hits": cache.hits, "misses": cache.misses},
    }


def _non_reportable_reasons(
    config: dict[str, Any], limit: int | None, dry_run: bool
) -> list[str]:
    reasons: list[str] = []
    if config["evaluation"]["mode"] != "formal":
        reasons.append("evaluation_mode_is_development")
    if config["judge"]["method"] != "nli":
        reasons.append("judge_method_is_not_nli")
    if limit is not None:
        reasons.append("sample_limit_used")
    if dry_run:
        reasons.append("dry_run_has_no_generation_or_workflow_result")
    if (
        config["evaluation"]["mode"] == "formal"
        and limit is None
        and not dry_run
        and config["judge"]["method"] == "nli"
    ):
        reasons.append("citation_human_calibration_audit_required")
    return reasons


def _new_run_id(config: dict[str, Any]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return f"{timestamp}_{config_hash}"


def _resolve(configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(args: list[str]) -> str:
    completed = subprocess.run(
        args, cwd=_PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _git_diff_hash() -> str:
    completed = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    digest = hashlib.sha256(completed.stdout)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    for relative in sorted(path for path in untracked.stdout.split(b"\0") if path):
        digest.update(relative)
        path = _PROJECT_ROOT / relative.decode("utf-8")
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


@click.command()
@click.option("--config", "config_path", default="eval/config.yaml", show_default=True)
@click.option(
    "--experiment",
    type=click.Choice(VALID_SELECTIONS, case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option("--output", "output_dir", default="reports/raw/", show_default=True)
@click.option("--show-config", is_flag=True)
@click.option("--validate", is_flag=True)
@click.option("--dry-run", is_flag=True, help="Development retrieval-only run.")
@click.option("--limit", type=click.IntRange(min=1), default=None)
def cli(
    config_path: str,
    experiment: str,
    output_dir: str,
    show_config: bool,
    validate: bool,
    dry_run: bool,
    limit: int | None,
) -> None:
    config = load_config(config_path)
    issues = validate_config(config, _PROJECT_ROOT)
    if issues:
        raise click.UsageError("invalid evaluation config:\n" + "\n".join(f"- {issue}" for issue in issues))
    if show_config:
        show_config_summary(config)
        return
    if validate:
        click.echo("Configuration validation: OK")
        return
    names = select_experiments(config, experiment.lower())
    run_id, _ = run_evaluation(
        config, names, Path(output_dir), limit=limit, dry_run=dry_run
    )
    click.echo(f"run_id: {run_id}")


if __name__ == "__main__":
    cli()
