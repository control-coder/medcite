"""Configuration-driven MediDiag evaluation runner.

RAG and Agent experiments use separate namespaces. Development runs are
explicitly non-reportable when they use a limit, dry-run, unpinned models, or
the rule fallback judge.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, cast

import click

from eval import metrics
from eval.configuration import (
    EXPERIMENTS,
    all_experiment_names,
    get_experiment,
    load_config,
    validate_config,
)
from eval.leakage_check import run_leakage_check
from medidiag.acceleration import runtime_snapshot
from medidiag.agents.arbitration import ArbitrationAgent
from medidiag.agents.base import AgentOutput
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.llm_client import LLMClient
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.runtime import AgentTopologyConfig, RuntimeMedicalAgents
from medidiag.agents.specialist import SpecialistAgent
from medidiag.compliance.guard import ComplianceGuard
from medidiag.llm import LLMProvider
from medidiag.llm.factory import build_llm_provider
from medidiag.rag.leakage import LEAKAGE_FLAG
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.review.citation import CitationVerifier
from medidiag.review.logic import ClinicalLogicReviewer
from medidiag.review.runtime import RuntimeMedicalReview
from medidiag.schemas import KnowledgeChunk, read_jsonl
from medidiag.workflow.assistant_pipeline import AssistantPipeline, StageContext

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_SELECTIONS = (*EXPERIMENTS, "rag_all", "rag_retrieval", "agent_all", "all")


@dataclass
class AgentOutputCache:
    _cache: dict[str, AgentOutput] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    hits: int = 0
    misses: int = 0

    def compute_hash(
        self,
        experiment: str,
        question: str,
        evidence_ids: list[str],
        specialty: str,
        claim_language: str | None = None,
    ) -> str:
        """Key on the experiment as well as the sample.

        The agent arms share one retrieval profile, so a key without the
        experiment name lets the second and third arm answer almost entirely
        from cache. Their ``stage_latency_ms["generation"]`` would then measure
        dictionary lookups, invalidating the topology comparison.
        """
        # claim 语言会改变 prompt 和 judge 输入，不能复用旧语言策略的生成缓存。
        payload = (
            f"{experiment}|{question}|{'-'.join(sorted(evidence_ids))}|"
            f"{specialty}|claim_language={claim_language or 'default'}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, input_hash: str) -> AgentOutput | None:
        with self._lock:
            value = self._cache.get(input_hash)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    def set(self, input_hash: str, output: AgentOutput) -> None:
        with self._lock:
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
    generation_executed: bool = False
    provider_request_ids: list[str] = field(default_factory=list)
    provider_usage: dict[str, int] = field(default_factory=dict)
    stage_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


def _round(value: float | None, digits: int) -> float | None:
    """Round a metric, preserving None for metrics with an empty denominator."""
    return None if value is None else round(value, digits)


def _fmt(value: float | None) -> str:
    """Render a metric for the console, distinguishing undefined from 0.0."""
    return "N/A" if value is None else f"{value:.4f}"


@dataclass
class ExperimentResult:
    experiment: str
    family: str
    sample_results: list[SampleResult] = field(default_factory=list)
    cache_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_recall_at_5(self) -> float | None:
        eligible = [
            result.recall_hit
            for result in self.sample_results
            if result.evidence_eligible and result.recall_hit is not None
        ]
        return metrics.compute_recall_at_k(eligible)

    @property
    def gold_evidence_coverage(self) -> float | None:
        eligible = sum(result.evidence_eligible for result in self.sample_results)
        return metrics.compute_gold_evidence_coverage(eligible, len(self.sample_results))

    @property
    def citation_precision(self) -> float | None:
        records = [
            record for result in self.sample_results for record in result.citation_results
        ]
        return metrics.compute_citation_precision(records)

    @property
    def unsupported_claim_rate(self) -> float | None:
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
                # None means the metric is undefined for this run (empty
                # denominator), which is not the same as a measured 0.0.
                "evidence_recall_at_5": _round(self.evidence_recall_at_5, 4),
                "gold_evidence_coverage": _round(self.gold_evidence_coverage, 4),
                "citation_precision": _round(self.citation_precision, 4),
                "unsupported_claim_rate": _round(self.unsupported_claim_rate, 4),
                # This is a reviewer-pipeline metric, not CLOSED_SUCCESS.
                "pipeline_approval_rate": _round(self.pipeline_approval_rate, 4),
                "workflow_success_rate": None,
                "p95_latency_ms": _round(self.p95_latency_ms, 2),
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
    if selection == "rag_retrieval":
        return [
            name
            for name in config["experiments"]["rag"]
            if not config["experiments"]["rag"][name]["config"].get("use_citation_review")
        ]
    if selection == "agent_all":
        return list(config["experiments"]["agent"])
    if selection == "all":
        return all_experiment_names(config)
    return [selection]


def _experiment_requires_generation(config: dict[str, Any], name: str) -> bool:
    family, experiment = get_experiment(config, name)
    if family == "agent":
        return True
    return bool(experiment["config"].get("use_citation_review"))


def run_evaluation(
    config: dict[str, Any],
    experiment_names: list[str],
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
    resume_run_id: str | None = None,
    sample_workers: int = 1,
) -> tuple[str, dict[str, ExperimentResult]]:
    started_at = datetime.now(UTC)
    mode = config["evaluation"]["mode"]
    if mode == "formal" and (limit is not None or dry_run):
        raise click.UsageError("formal evaluation forbids --limit and --dry-run")
    if sample_workers < 1:
        raise click.UsageError("--sample-workers must be at least 1")

    _run_leakage_gates(config, experiment_names)
    # 评测与 worker 复用同一个版本化 corpus、泄露门禁、normalizer 与索引构建
    # backend；各实验仍通过 Retriever 的 experiment_config 保持单变量控制。
    rag_backend = RuntimeMedicalRAG.from_config(
        config, root=_PROJECT_ROOT, retriever_factory=Retriever
    )
    normalizer = rag_backend.normalizer
    retriever = rag_backend.retriever
    runtime = config.get("runtime", {})
    requested_device = str(runtime.get("device", "auto"))
    batch_size = int(runtime.get("batch_size", config["embedding"].get("batch_size", 32)))

    verifier: CitationVerifier | None = None
    llm: LLMClient | LLMProvider | None = None
    needs_generation = (not dry_run) and any(
        _experiment_requires_generation(config, name) for name in experiment_names
    )
    if needs_generation:
        verifier = CitationVerifier(
            model_name=config["judge"]["model"],
            model_revision=config["judge"]["revision"],
            method=config["judge"]["method"],
            device=requested_device,
            batch_size=batch_size,
            input_language=config["judge"]["input_language"],
        )
        verifier.initialize()
        generation = config["generation"]
        if mode == "formal":
            profile_id = str(
                generation.get("profile")
                or generation.get("provider_profile")
                or generation.get("provider")
            )
            llm = build_llm_provider(profile_id, max_retries=2)
        else:
            # development runner 保留旧客户端，以维持离线 rule-fallback 测试；
            # formal runner 强制走统一 Provider Layer 与 P4/P5 runtime。
            temperature = generation.get("temperature")
            llm = LLMClient(
                base_url=generation["base_url"],
                model=generation["model"],
                timeout=int(generation["timeout_seconds"]),
                temperature=0.0 if temperature is None else float(temperature),
                max_tokens=int(generation["max_tokens"]),
                seed=int(generation["seed"]),
                thinking=str(generation.get("thinking", "disabled")),
                require_request_id=False,
            )

    run_id = resume_run_id or _new_run_id(config)
    run_dir = output_dir / run_id
    config_snapshot = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2)
    resumed = resume_run_id is not None
    if resumed:
        if not run_dir.is_dir():
            raise click.UsageError(f"恢复目录不存在: {run_dir}")
        snapshot_path = run_dir / "config.snapshot.json"
        if not snapshot_path.is_file():
            raise click.UsageError("恢复目录缺少 config.snapshot.json")
        saved_config = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if _config_hash(saved_config) != _config_hash(config):
            raise click.UsageError("恢复配置与原 run 的 config hash 不一致")
        if (run_dir / "manifest.json").exists():
            raise click.UsageError("目标 run 已存在 manifest.json，不允许再次恢复")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.snapshot.json").write_text(config_snapshot, encoding="utf-8")

    cache = AgentOutputCache()
    generation_config = config["generation"]
    pipeline = AssistantPipeline(
        component_version="eval-runner-v1",
        provider_profile=str(generation_config.get("provider_profile", generation_config.get("profile", "evaluation"))),
        stage_timeout_seconds={
            "normalize": float(config.get("retrieval", {}).get("timeout_seconds", 60)),
            "retrieval": float(config.get("retrieval", {}).get("timeout_seconds", 60)),
            "generation": float(generation_config.get("timeout_seconds", 60)),
            "review": float(config.get("judge", {}).get("timeout_seconds", 60)),
        },
    )
    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    dirty_diff_hash = _git_diff_hash()
    retrieval_lock = Lock() if sample_workers > 1 else None
    review_lock = Lock() if sample_workers > 1 else None
    results: dict[str, ExperimentResult] = {}
    reused_experiments: list[str] = []
    checkpoint_samples_reused: dict[str, int] = {}

    for name in experiment_names:
        family, experiment = get_experiment(config, name)
        records = _load_experiment_records(config, family, limit)
        completed_path = run_dir / f"{name}.json"
        if resumed and completed_path.is_file():
            result = _load_completed_experiment(completed_path, name, family, records)
            results[name] = result
            reused_experiments.append(name)
            click.echo(f"{name}: 已复用完整实验 {len(result.sample_results)} 个样本")
            continue

        checkpoint_dir = run_dir / f"{name}.checkpoint"
        restored = _load_checkpoint_samples(checkpoint_dir, name, family, records)
        checkpoint_samples_reused[name] = len(restored)

        def save_checkpoint(
            index: int,
            sample_result: SampleResult,
            target_dir: Path = checkpoint_dir,
        ) -> None:
            _write_sample_checkpoint(target_dir, index, sample_result)

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
            pipeline,
            run_id,
            existing_samples=restored,
            sample_workers=sample_workers,
            checkpoint_callback=save_checkpoint,
            retrieval_lock=retrieval_lock,
            review_lock=review_lock,
        )
        results[name] = result
        _atomic_write_json(completed_path, result.to_dict())
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        click.echo(
            f"{name}: Recall@5={_fmt(result.evidence_recall_at_5)}, "
            f"GoldCoverage={_fmt(result.gold_evidence_coverage)}"
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
        llm=llm,
        retriever=retriever,
        provider_usage=_aggregate_provider_usage(results),
        resume={
            "enabled": resumed,
            "source_run_id": resume_run_id,
            "completed_experiments_reused": reused_experiments,
            "checkpoint_samples_reused": checkpoint_samples_reused,
            "sample_workers": sample_workers,
        },
    )
    _atomic_write_json(run_dir / "manifest.json", manifest)
    return run_id, results


def _config_hash(config: dict[str, Any]) -> str:
    """计算影响评测口径的规范化配置哈希。"""
    return hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """先写同目录临时文件再替换，避免中断留下半个 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _sample_result_from_dict(payload: dict[str, Any]) -> SampleResult:
    """严格反序列化 checkpoint，schema 漂移时拒绝静默复用。"""
    expected = {item.name for item in fields(SampleResult)}
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise click.UsageError(
            f"checkpoint SampleResult schema 不一致: missing={missing}, extra={extra}"
        )
    return SampleResult(**payload)


def _validate_sample_sequence(
    samples: list[SampleResult],
    name: str,
    family: str,
    records: list[dict[str, Any]],
) -> None:
    expected_ids = [str(record["sample_id"]) for record in records]
    actual_ids = [sample.sample_id for sample in samples]
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise click.UsageError(f"{name} 的恢复样本 ID 不完整、重复或顺序不一致")
    if any(sample.experiment != name or sample.family != family for sample in samples):
        raise click.UsageError(f"{name} 的恢复样本 experiment/family 不一致")


def _load_completed_experiment(
    path: Path,
    name: str,
    family: str,
    records: list[dict[str, Any]],
) -> ExperimentResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != name or payload.get("family") != family:
        raise click.UsageError(f"完整实验文件 {path.name} 的标识不一致")
    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, list) or payload.get("sample_count") != len(records):
        raise click.UsageError(f"完整实验文件 {path.name} 的样本数不一致")
    samples = [_sample_result_from_dict(dict(item)) for item in samples_payload]
    _validate_sample_sequence(samples, name, family, records)
    return ExperimentResult(
        experiment=name,
        family=family,
        sample_results=samples,
        cache_stats=dict(payload.get("cache_stats", {})),
    )


def _write_sample_checkpoint(
    checkpoint_dir: Path, index: int, sample_result: SampleResult
) -> None:
    """每个样本独立原子落盘，停止后无需重复已付费 Provider 调用。"""
    _atomic_write_json(
        checkpoint_dir / f"{index:06d}.json",
        {"index": index, "sample": asdict(sample_result)},
    )


def _load_checkpoint_samples(
    checkpoint_dir: Path,
    name: str,
    family: str,
    records: list[dict[str, Any]],
) -> dict[int, SampleResult]:
    if not checkpoint_dir.exists():
        return {}
    restored: dict[int, SampleResult] = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = payload.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(records):
            raise click.UsageError(f"checkpoint 索引越界: {path.name}")
        if index in restored:
            raise click.UsageError(f"checkpoint 索引重复: {index}")
        sample = _sample_result_from_dict(dict(payload.get("sample", {})))
        expected_id = str(records[index]["sample_id"])
        if (
            sample.sample_id != expected_id
            or sample.experiment != name
            or sample.family != family
        ):
            raise click.UsageError(f"checkpoint 样本标识不一致: {path.name}")
        restored[index] = sample
    return restored


def _run_with_lock(
    lock: Any | None, operation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """只串行化共享 Retriever/NLI judge；Provider 网络调用保持有限并发。"""
    if lock is None:
        return operation()
    with lock:
        return operation()


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
    retriever: Any,
    normalizer: TerminologyNormalizer,
    verifier: CitationVerifier | None,
    llm: LLMClient | LLMProvider | None,
    cache: AgentOutputCache,
    dry_run: bool,
    pipeline: AssistantPipeline | None = None,
    pipeline_run_id: str | None = None,
    existing_samples: dict[int, SampleResult] | None = None,
    sample_workers: int = 1,
    checkpoint_callback: Any | None = None,
    retrieval_lock: Any | None = None,
    review_lock: Any | None = None,
) -> ExperimentResult:
    result = ExperimentResult(experiment=name, family=family)
    local_before = (cache.hits, cache.misses)
    provider_before = _legacy_usage_summary(llm)
    retrieval_before = retriever.cache_stats()
    rag_config = (
        experiment["config"]
        if family == "rag"
        else config["experiments"]["rag"][experiment["retrieval_profile"]]["config"]
    )
    topology = "single" if family == "rag" else experiment["topology"]
    restored = existing_samples or {}
    ordered_results: dict[int, SampleResult] = dict(restored)
    pending = [(index, record) for index, record in enumerate(records) if index not in restored]

    def run_one(item: tuple[int, dict[str, Any]]) -> tuple[int, SampleResult]:
        index, record = item
        sample_result = _run_sample(
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
            pipeline,
            pipeline_run_id,
            retrieval_lock=retrieval_lock,
            review_lock=review_lock,
        )
        return index, sample_result

    if sample_workers == 1:
        completed = map(run_one, pending)
        for index, sample_result in completed:
            ordered_results[index] = sample_result
            if checkpoint_callback is not None:
                checkpoint_callback(index, sample_result)
    else:
        # 只保留一个并发窗口，避免单样本失败后队列仍继续发出大量付费请求。
        with ThreadPoolExecutor(
            max_workers=sample_workers, thread_name_prefix="formal-sample"
        ) as executor:
            for offset in range(0, len(pending), sample_workers):
                batch = pending[offset : offset + sample_workers]
                futures = [executor.submit(run_one, item) for item in batch]
                first_error: BaseException | None = None
                for future in as_completed(futures):
                    try:
                        index, sample_result = future.result()
                    except BaseException as exc:
                        first_error = first_error or exc
                        continue
                    ordered_results[index] = sample_result
                    if checkpoint_callback is not None:
                        checkpoint_callback(index, sample_result)
                if first_error is not None:
                    raise first_error
    result.sample_results = [ordered_results[index] for index in range(len(records))]
    local_hits = cache.hits - local_before[0]
    local_misses = cache.misses - local_before[1]
    provider_after = _legacy_usage_summary(llm)
    retrieval_after = retriever.cache_stats()
    result.cache_stats = {
        "local_agent_output": {
            "hits": local_hits,
            "misses": local_misses,
            "hit_rate": round(local_hits / (local_hits + local_misses), 4)
            if local_hits + local_misses
            else 0.0,
        },
        "provider_kv": _counter_delta(provider_before, provider_after),
        "retrieval_scores": {
            component: _counter_delta(
                retrieval_before.get(component, {}), retrieval_after.get(component, {})
            )
            for component in ("embedding", "bm25")
        },
    }
    return result


def _is_unified_provider(llm: object | None) -> bool:
    """统一 Provider 暴露 generate/capabilities；旧评测客户端只暴露 complete。"""
    return llm is not None and callable(getattr(llm, "generate", None)) and callable(
        getattr(llm, "capabilities", None)
    )


def _legacy_usage_summary(llm: object | None) -> dict[str, int]:
    summary = getattr(llm, "usage_summary", None)
    return dict(summary()) if callable(summary) else {}


def _merge_usage(target: dict[str, int], usage: dict[str, Any] | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value


def _assert_formal_response_id_provenance(
    config: dict[str, Any],
    result: SampleResult,
    generation_payload: dict[str, Any] | None = None,
    arbitration_payload: dict[str, Any] | None = None,
) -> None:
    """校验 formal Agent 每一次真实生成调用都保存了 provider 响应 ID。"""
    generation = config["generation"]
    if config["evaluation"]["mode"] != "formal" or generation.get(
        "provenance_mode", "provider_snapshot"
    ) != "provider_response_id":
        return

    if generation_payload is None:
        if not result.provider_request_ids:
            raise click.ClickException(
                "FORMAL_GENERATION_RESPONSE_ID_MISSING: "
                "provider_response_id mode requires response.id for every generated sample"
            )
        return

    arbitration_payload = arbitration_payload or {}
    missing_calls: list[str] = []
    agents = generation_payload.get("agents", [])
    for index, artifact in enumerate(agents, start=1):
        provenance = artifact.get("provider_provenance") or {}
        if not provenance.get("response_id"):
            missing_calls.append(f"specialist_{index}")

    topology = str(generation_payload.get("topology", "single"))
    arbitration_method = str(arbitration_payload.get("arbitration_method", ""))
    if topology != "single" and arbitration_method != "abstention_no_claims":
        arbitration_attempts = arbitration_payload.get("provider_attempt_provenance") or []
        if arbitration_attempts:
            for index, provenance in enumerate(arbitration_attempts, start=1):
                if not provenance.get("response_id"):
                    missing_calls.append(f"arbitration_attempt_{index}")
        else:
            arbitration_provenance = arbitration_payload.get("provider_provenance") or {}
            if not arbitration_provenance.get("response_id"):
                missing_calls.append("arbitration")

    if not agents:
        missing_calls.append("specialist")
    if missing_calls:
        raise click.ClickException(
            "FORMAL_GENERATION_RESPONSE_ID_MISSING: "
            "provider_response_id mode requires response.id for every actual provider "
            f"call; missing={','.join(missing_calls)}"
        )


def _run_sample(
    name: str,
    family: str,
    topology: str,
    experiment: dict[str, Any],
    rag_config: dict[str, bool],
    sample: dict[str, Any],
    config: dict[str, Any],
    retriever: Any,
    normalizer: TerminologyNormalizer,
    verifier: CitationVerifier | None,
    llm: LLMClient | LLMProvider | None,
    cache: AgentOutputCache,
    dry_run: bool,
    pipeline: AssistantPipeline | None = None,
    pipeline_run_id: str | None = None,
    retrieval_lock: Any | None = None,
    review_lock: Any | None = None,
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
    active_pipeline = pipeline or AssistantPipeline(
        component_version="eval-runner-v1",
        provider_profile=str(config["generation"].get(
            "provider_profile", config["generation"].get("profile", "evaluation")
        )),
    )
    stage_context = StageContext(
        run_id=pipeline_run_id or f"eval:{name}:{result.sample_id}",
        case_id=result.sample_id,
        sample_id=result.sample_id,
        experiment=name,
        mode="evaluation" if config["evaluation"]["mode"] == "formal" else "development",
        prompt_version=str(config["generation"].get("prompt_version", "")),
    )
    top_k = int(config["retrieval"]["top_k"])
    candidate_k = int(config["retrieval"]["candidate_k"])

    retrieval_query = question
    if rag_config["use_term_normalization"]:
        normalize_execution = active_pipeline.run_stage(
            stage="normalize",
            input_payload={"question": question},
            operation=lambda: {
                "normalized_query": normalizer.normalize(question).normalized,
            },
            context=stage_context,
        )
        retrieval_query = str(normalize_execution.payload["normalized_query"])
        result.stage_latency_ms["normalize"] = normalize_execution.elapsed_ms
        result.stage_artifacts["normalize"] = normalize_execution.artifact.to_dict()

    search_results: list[Any] = []
    retrieval_payload: dict[str, Any] = {}

    def retrieve_stage() -> dict[str, Any]:
        nonlocal search_results, retrieval_payload
        if _is_unified_provider(llm):
            runtime_rag = RuntimeMedicalRAG(
                chunks=list(retriever.chunks),
                corpus_version=str(config["dataset"]["version"]),
                retrieval_config=dict(config["retrieval"]),
                experiment_config=dict(rag_config),
                normalizer=normalizer,
                retriever=retriever,
                corpus_path=str(_resolve(config["dataset"]["knowledge_base_path"])),
                leakage_gate={
                    "status": "passed",
                    "rule_version": "eval-runner-leakage-v1",
                },
            )
            rerank_started = time.perf_counter() if rag_config["use_rerank"] else None
            retrieval_payload = runtime_rag.retrieve(retrieval_query)
            if rerank_started is not None:
                result.stage_latency_ms["rerank"] = _elapsed_ms(rerank_started)
            return retrieval_payload

        search_results = retriever.search(
            retrieval_query,
            top_k=candidate_k if rag_config["use_rerank"] else top_k,
            experiment_config=rag_config,
        )
        if rag_config["use_rerank"]:
            rerank_started = time.perf_counter()
            search_results = retriever.rerank(
                retrieval_query, search_results, top_k=top_k
            )
            result.stage_latency_ms["rerank"] = _elapsed_ms(rerank_started)
        retrieval_payload = {
            "query": retrieval_query,
            "chunks": [
                {
                    **(asdict(item.chunk) if item.chunk is not None else {}),
                    "chunk_id": item.chunk_id,
                    "score": item.final_score,
                }
                for item in search_results
            ],
        }
        return retrieval_payload

    retrieval_execution = active_pipeline.run_stage(
        stage="retrieval",
        input_payload={
            "normalized_query": retrieval_query,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "experiment_config": rag_config,
        },
        operation=lambda: _run_with_lock(retrieval_lock, retrieve_stage),
        context=stage_context,
    )
    result.stage_latency_ms["retrieval"] = retrieval_execution.elapsed_ms
    result.stage_artifacts["retrieval"] = retrieval_execution.artifact.to_dict()

    if _is_unified_provider(llm):
        evidence_ids = [str(item["chunk_id"]) for item in retrieval_payload["chunks"]]
        evidence = [
            KnowledgeChunk(
                chunk_id=str(item["chunk_id"]),
                source=str(item.get("source", "")),
                source_id=str(item.get("source_id", "")),
                text=str(item.get("text", "")),
                evidence_level=str(item.get("evidence_level", "level_5_other")),
                metadata=dict(item.get("metadata", {})),
            )
            for item in retrieval_payload["chunks"]
        ]
    else:
        evidence_ids = [item.chunk_id for item in search_results]
        evidence = [item.chunk for item in search_results if item.chunk is not None]
    result.recall_hit = bool(set(evidence_ids) & set(gold_ids)) if gold_ids else None
    if dry_run or (family == "rag" and not rag_config["use_citation_review"]):
        result.latency_ms = _elapsed_ms(started)
        return result

    result.generation_executed = True
    # 正式 run 不接受把 provider 故障折算成弃权：AgentProviderError 必须
    # 向上传播，终止运行并阻止 manifest 写出。
    fail_closed = config["evaluation"]["mode"] == "formal"
    # 仅正式 NLI 评测要求 Agent 直接生成英文 claim，避免向英文 judge 输入中英混用文本。
    claim_language = (
        config["judge"]["input_language"]
        if fail_closed and config["judge"]["method"] == "nli"
        else None
    )
    if _is_unified_provider(llm):
        return _run_unified_runtime_sample(
            result=result,
            started=started,
            question=question,
            sample=sample,
            topology=topology,
            experiment=experiment,
            config=config,
            retrieval_payload=retrieval_payload,
            verifier=verifier,
            provider=cast(LLMProvider, llm),
            active_pipeline=active_pipeline,
            stage_context=stage_context,
            generation_input_hash=retrieval_execution.artifact.output_hash,
            review_lock=review_lock,
        )

    outputs: list[AgentOutput] = []

    def generation_stage() -> dict[str, Any]:
        nonlocal outputs
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
            fail_closed,
            claim_language=claim_language,
        )
        agents = [output.to_dict() for output in outputs]
        return {
            "agents": agents,
            "claims": [claim for agent in agents for claim in agent.get("claims", [])],
        }

    generation_execution = active_pipeline.run_stage(
        stage="generation",
        input_payload={
            "question": question,
            "evidence_ids": evidence_ids,
            "topology": topology,
            "experiment": name,
        },
        operation=generation_stage,
        context=stage_context,
    )
    result.stage_latency_ms["generation"] = generation_execution.elapsed_ms
    result.stage_artifacts["generation"] = generation_execution.artifact.to_dict()
    result.agent_abstained = any(output.abstain for output in outputs)
    result.provider_request_ids = sorted(
        {
            output.provider_request_id
            for output in outputs
            if output.provider_request_id
        }
    )

    citation_results: list[Any] = []
    review_approved = False
    unsupported_rate: float | None = None

    def review_stage() -> dict[str, Any]:
        nonlocal citation_results, review_approved, unsupported_rate
        reviewer = ClinicalLogicReviewer()
        review_approved = all(reviewer.check(output).is_approved for output in outputs)

        claims: list[dict[str, Any]] = []
        for output_index, output in enumerate(outputs):
            for claim_index, claim in enumerate(output.claims):
                claims.append(
                    {
                        "claim_id": f"{result.sample_id}:{output_index:02d}_{claim_index:04d}",
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
            or unsupported_rate is None
            or unsupported_rate
            <= config["workflow"]["review"]["unsupported_claim_threshold"]
        )
        approved = (
            review_approved
            and citation_gate
            and not result.agent_abstained
            and not result.compliance_blocked
        )
        return {
            "verdict": "APPROVED" if approved else "REVISION_REQUIRED",
            "citation_verdicts": result.citation_results,
            "compliance_status": (
                "BLOCKED" if result.compliance_blocked else "PASSED"
            ),
        }

    review_execution = active_pipeline.run_stage(
        stage="review",
        input_payload={
            "generation_output_hash": generation_execution.artifact.output_hash,
            "use_citation_review": rag_config["use_citation_review"],
        },
        operation=lambda: _run_with_lock(review_lock, review_stage),
        context=stage_context,
    )
    result.stage_latency_ms["review"] = review_execution.elapsed_ms
    result.stage_artifacts["review"] = review_execution.artifact.to_dict()
    result.pipeline_approved = review_execution.payload["verdict"] == "APPROVED"
    result.latency_ms = _elapsed_ms(started)
    return result


def _run_unified_runtime_sample(
    *,
    result: SampleResult,
    started: float,
    question: str,
    sample: dict[str, Any],
    topology: str,
    experiment: dict[str, Any],
    config: dict[str, Any],
    retrieval_payload: dict[str, Any],
    verifier: CitationVerifier | None,
    provider: LLMProvider,
    active_pipeline: AssistantPipeline,
    stage_context: StageContext,
    generation_input_hash: str,
    review_lock: Any | None = None,
) -> SampleResult:
    """使用 P4/P5 统一 runtime 完成 formal generation、仲裁与固定 NLI 审核。"""
    if verifier is None:
        raise click.ClickException("FORMAL_JUDGE_MISSING: 统一 runtime 缺少固定 NLI judge")
    fixed_pair = tuple(experiment.get("specialist_pair", ("cardiology", "respiratory")))
    if len(fixed_pair) != 2:
        fixed_pair = ("cardiology", "respiratory")
    runtime_agents = RuntimeMedicalAgents(
        provider,
        config=AgentTopologyConfig(
            topology=cast(Any, topology),
            fixed_pair=(str(fixed_pair[0]), str(fixed_pair[1])),
            specialist_prompt_version=str(config["generation"].get("prompt_version", "")),
            arbitration_prompt_version=str(
                config["generation"].get("arbitration_prompt_version", "arbitrator-agent-v1")
            ),
            timeout_s=float(config["generation"].get("timeout_seconds", 60)),
            max_tokens=int(config["generation"].get("max_tokens", 4096)),
            reasoning_mode=cast(Any, config["generation"].get("thinking", "provider_default")),
            allow_arbitration_fallback=config["evaluation"]["mode"] != "formal",
        ),
        normalizer=None,
        evidence_level_scores=config["retrieval"]["evidence_levels"],
    )
    generation: dict[str, Any] = {}
    arbitration: dict[str, Any] = {}

    def generation_stage() -> dict[str, Any]:
        nonlocal generation
        generation = runtime_agents.generate(
            question,
            retrieval_payload,
            {
                "question": question,
                "options": sample.get("options"),
                "evaluation_experiment": result.experiment,
            },
        )
        return generation

    generation_execution = active_pipeline.run_stage(
        stage="generation",
        input_payload={
            "retrieval_output_hash": generation_input_hash,
            "topology": topology,
            "experiment": result.experiment,
        },
        operation=generation_stage,
        context=stage_context,
    )
    result.stage_latency_ms["generation"] = generation_execution.elapsed_ms
    result.stage_artifacts["generation"] = generation_execution.artifact.to_dict()

    def arbitration_stage() -> dict[str, Any]:
        nonlocal arbitration
        arbitration = runtime_agents.arbitrate(generation, retrieval_payload)
        if (
            config["evaluation"]["mode"] == "formal"
            and topology != "single"
            and arbitration.get("fallback_used")
        ):
            raise click.ClickException(
                "FORMAL_ARBITRATION_FALLBACK: 正式评测禁止把仲裁 Provider 故障折算为规则回退"
            )
        return arbitration

    arbitration_execution = active_pipeline.run_stage(
        stage="arbitration",
        input_payload={
            "generation_output_hash": generation_execution.artifact.output_hash,
            "evidence_bundle_id": retrieval_payload.get("evidence_bundle_id"),
            "topology": topology,
        },
        operation=arbitration_stage,
        context=stage_context,
    )
    result.stage_latency_ms["arbitration"] = arbitration_execution.elapsed_ms
    result.stage_artifacts["arbitration"] = arbitration_execution.artifact.to_dict()
    routing = generation.get("routing", {})
    result.specialty_pair = [str(value) for value in routing.get("specialty_pair", [])]
    result.routing_confidence = float(routing.get("confidence", 0.0) or 0.0)
    result.routing_fallback = bool(routing.get("is_fallback", False))
    result.arbitration_verdict = str(arbitration.get("verdict", ""))

    provider_ids: set[str] = set()
    usage: dict[str, int] = {}
    for artifact in generation.get("agents", []):
        provenance = artifact.get("provider_provenance", {})
        response_id = provenance.get("response_id")
        if response_id:
            provider_ids.add(str(response_id))
        _merge_usage(usage, provenance.get("usage"))
    arbitration_attempts = arbitration.get("provider_attempt_provenance") or []
    if arbitration_attempts:
        for provenance in arbitration_attempts:
            response_id = provenance.get("response_id")
            if response_id:
                provider_ids.add(str(response_id))
            _merge_usage(usage, provenance.get("usage"))
    else:
        arbitration_provenance = arbitration.get("provider_provenance") or {}
        arbitration_response_id = arbitration_provenance.get("response_id")
        if arbitration_response_id:
            provider_ids.add(str(arbitration_response_id))
        _merge_usage(usage, arbitration_provenance.get("usage"))
    result.provider_request_ids = sorted(provider_ids)
    result.provider_usage = usage
    _assert_formal_response_id_provenance(config, result, generation, arbitration)

    review_payload: dict[str, Any] = {}

    def review_stage() -> dict[str, Any]:
        nonlocal review_payload
        review_started = time.perf_counter()
        review_payload = RuntimeMedicalReview(verifier).review(
            generation, arbitration, retrieval_payload
        )
        result.stage_latency_ms["judge"] = _elapsed_ms(review_started)
        return review_payload

    review_execution = active_pipeline.run_stage(
        stage="review",
        input_payload={
            "generation_output_hash": generation_execution.artifact.output_hash,
            "arbitration_output_hash": arbitration_execution.artifact.output_hash,
            "evidence_bundle_id": retrieval_payload.get("evidence_bundle_id"),
        },
        operation=lambda: _run_with_lock(review_lock, review_stage),
        context=stage_context,
    )
    result.stage_latency_ms["review"] = review_execution.elapsed_ms
    result.stage_artifacts["review"] = review_execution.artifact.to_dict()
    result.citation_results = []
    for item in review_payload.get("citation_verdicts", []):
        value = dict(item)
        value.setdefault("evidence_chunk_id", value.get("chunk_id", ""))
        result.citation_results.append(value)
    result.total_claims = len(review_payload.get("canonical_claims", []))
    result.compliance_blocked = review_payload.get("compliance_status") == "BLOCKED"
    result.agent_abstained = any(
        bool((artifact.get("output") or {}).get("abstain"))
        for artifact in generation.get("agents", [])
    )
    result.pipeline_approved = (
        review_payload.get("verdict") == "APPROVED" and not result.agent_abstained
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
    llm: LLMClient | LLMProvider | None,
    cache: AgentOutputCache,
    sample_result: SampleResult,
    fail_closed: bool,
    claim_language: str | None = None,
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
        ).route(question, evidence)
        specialties = list(routing.specialty_pair)
        sample_result.routing_fallback = routing.is_fallback
        sample_result.routing_confidence = routing.confidence
    sample_result.specialty_pair = specialties

    outputs: list[AgentOutput] = []
    for specialty in specialties:
        input_hash = cache.compute_hash(
            sample_result.experiment,
            question,
            evidence_ids,
            specialty,
            claim_language=claim_language,
        )
        cached = cache.get(input_hash)
        if cached is not None:
            outputs.append(cached)
            sample_result.cache_hit = True
            continue
        agent = (
            DiagnosisAgent(cast(Any, llm), fail_closed=fail_closed)
            if specialty == "general_diagnosis"
            else SpecialistAgent(specialty, cast(Any, llm), fail_closed=fail_closed)
        )
        output = agent.generate(
            question, evidence, "", options, claim_language=claim_language
        )
        cache.set(input_hash, output)
        for key, value in output.provider_usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and key != "prompt_cache_hit_rate":
                sample_result.provider_usage[key] = sample_result.provider_usage.get(key, 0) + value
        hit = sample_result.provider_usage.get("prompt_cache_hit_tokens", 0)
        miss = sample_result.provider_usage.get("prompt_cache_miss_tokens", 0)
        if hit + miss > 0:
            sample_result.provider_usage["prompt_cache_hit_rate"] = round(
                hit / (hit + miss) * 1000000
            )
        outputs.append(output)

    if len(outputs) == 2 and not any(output.abstain for output in outputs):
        sample_result.arbitration_verdict = ArbitrationAgent().arbitrate(
            outputs[0], outputs[1]
        ).verdict
    elif len(outputs) == 2:
        sample_result.arbitration_verdict = "ESCALATED"
    return outputs


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    keys = set(before) | set(after)
    result = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
        if key != "prompt_cache_hit_rate"
    }
    hit = result.get("prompt_cache_hit_tokens", 0)
    miss = result.get("prompt_cache_miss_tokens", 0)
    if hit + miss > 0:
        result["prompt_cache_hit_rate"] = round(hit / (hit + miss) * 1000000)
    return result


def _aggregate_provider_usage(
    results: dict[str, ExperimentResult],
) -> dict[str, int]:
    total: dict[str, int] = {}
    for experiment in results.values():
        for sample in experiment.sample_results:
            _merge_usage(total, sample.provider_usage)
    return total


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
    llm: LLMClient | LLMProvider | None = None,
    retriever: Any | None = None,
    provider_usage: dict[str, int] | None = None,
    resume: dict[str, Any] | None = None,
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
        # requirements.txt 全部为 `>=` 范围，对它求哈希只能证明"声明未变"，
        # 不能证明两次运行装的是同一批依赖，因此不再称为 lock hash。
        "dependency_spec_hash": _hash_file(_PROJECT_ROOT / "requirements.txt"),
        "resolved_package_versions": _resolved_package_versions(),
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
            # 统一 Provider 契约当前不发送 seed；配置 seed 只用于数据抽样与本地确定性步骤。
            "seed_applied": False,
            "seed_not_applied_reason": "provider_profile_does_not_send_seed",
            "thinking": config["generation"].get("thinking", "disabled"),
            "timeout_seconds": config["generation"]["timeout_seconds"],
            "cache_strategy": config["generation"].get("cache_strategy"),
            "provider_usage": provider_usage or _legacy_usage_summary(llm),
        },
        "retrieval": {
            **config["retrieval"],
            "score_cache": retriever.cache_stats() if retriever is not None else {},
        },
        "runtime": runtime_snapshot(
            str(config.get("runtime", {}).get("device", "auto")),
            retriever.actual_device if retriever is not None else "cpu",
            int(config.get("runtime", {}).get("batch_size", config["embedding"].get("batch_size", 32))),
        ),
        "judge_method": config["judge"]["method"],
        "limit": limit,
        "dry_run": dry_run,
        "resume": resume or {"enabled": False, "sample_workers": 1},
        "cache": {
            "local_agent_output": {"hits": cache.hits, "misses": cache.misses},
            "provider_kv": _legacy_usage_summary(llm),
        },
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


# 影响评测数值的发行版。记录实际解析到的版本，而不是范围声明。
_PROVENANCE_DISTRIBUTIONS = (
    "faiss-cpu",
    "httpx",
    "numpy",
    "rank-bm25",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
)


def _resolved_package_versions() -> dict[str, str]:
    """Record the versions actually installed at run time.

    This is the dependency provenance a range specification cannot provide.
    Distributions absent from the environment are recorded as ``not_installed``
    rather than omitted, so a missing accelerator stack is visible in the
    manifest instead of silently indistinguishable from an unrecorded field.
    """
    from importlib.metadata import PackageNotFoundError, version

    resolved: dict[str, str] = {}
    for name in _PROVENANCE_DISTRIBUTIONS:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = "not_installed"
    return resolved


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
@click.option("--resume-run-id", default=None, help="从已有未完成 run 恢复。")
@click.option("--sample-workers", type=click.IntRange(min=1, max=8), default=1, show_default=True, help="样本级有限并发数；正式评测建议从 2 开始。")
def cli(
    config_path: str,
    experiment: str,
    output_dir: str,
    show_config: bool,
    validate: bool,
    dry_run: bool,
    limit: int | None,
    resume_run_id: str | None,
    sample_workers: int,
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
        config, names, Path(output_dir), limit=limit, dry_run=dry_run,
        resume_run_id=resume_run_id, sample_workers=sample_workers,
    )
    click.echo(f"run_id: {run_id}")


if __name__ == "__main__":
    cli()
