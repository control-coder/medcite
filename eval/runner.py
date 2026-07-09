"""评测 runner（完整实现）。

API 调用优化:
1. AgentOutputCache: 基于 input_hash 缓存，避免重复 LLM 调用
2. 批量 embedding: 检索索引构建一次，所有组复用
3. 检索结果复用: embedding_scores 跨消融组复用（只权重不同）
4. --dry-run: 只计算检索指标，不调 LLM
5. --limit: 限制样本数（调试用）

用法:
    python -m eval.runner --help
    python -m eval.runner --config eval/config.yaml --show-config
    python -m eval.runner --config eval/config.yaml --dry-run --group all
    python -m eval.runner --config eval/config.yaml --group F --limit 10
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import click
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from medidiag.agents.arbitration import ArbitrationAgent
from medidiag.agents.base import AgentOutput
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.llm_client import LLMClient
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialist import SpecialistAgent
from medidiag.agents.specialty_data import BASELINE_PAIR
from medidiag.compliance.guard import ComplianceGuard
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import ABLATION_CONFIGS, Retriever
from medidiag.review.citation import CitationVerifier
from medidiag.review.logic import ClinicalLogicReviewer
from medidiag.schemas import KnowledgeChunk, read_jsonl

import eval.metrics as metrics

VALID_GROUPS = ("A", "B", "C", "D", "E", "F", "all")


# ===== 配置加载与校验（保留阶段 0 接口）=====


def load_config(config_path: str | Path) -> dict[str, Any]:
    """加载评测配置 YAML。"""
    path = Path(config_path)
    if not path.exists():
        raise click.FileError(str(path), hint="eval config not found")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise click.UsageError(f"eval config must be a mapping, got {type(cfg)}")
    return cfg


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """校验配置完整性，返回缺失字段列表。"""
    issues: list[str] = []
    for key in ("generation", "embedding", "rerank", "judge", "dataset", "retrieval", "ablation", "workflow", "metrics", "reproduction"):
        if key not in cfg:
            issues.append(f"missing top-level key: {key}")
    if "generation" in cfg and not cfg["generation"].get("model"):
        issues.append("generation.model must be locked")
    if "embedding" in cfg and not cfg["embedding"].get("model"):
        issues.append("embedding.model must be locked")
    if "rerank" in cfg and not cfg["rerank"].get("model"):
        issues.append("rerank.model must be locked")
    if "judge" in cfg and not cfg["judge"].get("model"):
        issues.append("judge.model must be locked")
    if "ablation" in cfg and "groups" in cfg["ablation"]:
        for g in ("A", "B", "C", "D", "E", "F"):
            if g not in cfg["ablation"]["groups"]:
                issues.append(f"missing ablation group: {g}")
    return issues


def show_config_summary(cfg: dict[str, Any]) -> None:
    """打印配置摘要。"""
    click.echo("=" * 70)
    click.echo("MediDiag Eval Configuration Summary")
    click.echo("=" * 70)
    click.echo(f"  generation_model : {cfg['generation']['model']}")
    click.echo(f"  embedding_model  : {cfg['embedding']['model']}")
    click.echo(f"  rerank_model     : {cfg['rerank']['model']}")
    click.echo(f"  judge_model      : {cfg['judge']['model']}")
    click.echo(f"  temperature      : {cfg['generation']['temperature']}")
    click.echo(f"  seed             : {cfg['generation']['seed']}")
    click.echo(f"  dataset_version  : {cfg['dataset']['version']}")
    w = cfg["retrieval"]["weights"]
    click.echo(f"  retrieval weights: w1={w['w1_bm25']} w2={w['w2_embedding']} w3={w['w3_evidence_level']} w4={w['w4_term_overlap']}")
    click.echo(f"  ablation groups  : {', '.join(cfg['ablation']['groups'].keys())}")
    click.echo(f"  metrics count    : {len(cfg['metrics'])}")
    click.echo("=" * 70)


# ===== Agent 输出缓存（API 调用优化核心）=====


@dataclass
class AgentOutputCache:
    """Agent 输出缓存，基于 input_hash 避免重复 LLM 调用。

    缓存策略:
        - key: input_hash（question + evidence_ids + specialty + routing_note 的 sha256）
        - value: AgentOutput
        - 过期: 不过期（temperature=0，相同输入产生相同输出）
    """

    _cache: dict[str, AgentOutput] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def compute_hash(
        self,
        question: str,
        evidence_ids: list[str],
        specialty: str,
        routing_note: str = "",
    ) -> str:
        """计算 Agent 输入的哈希。"""
        data = f"{question}|{'-'.join(sorted(evidence_ids))}|{specialty}|{routing_note}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def get(self, input_hash: str) -> AgentOutput | None:
        result = self._cache.get(input_hash)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def set(self, input_hash: str, output: AgentOutput) -> None:
        self._cache[input_hash] = output

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ===== 评测结果数据结构 =====


@dataclass
class SampleResult:
    """单个样本的评测结果。"""

    sample_id: str
    group: str
    recall_hit: bool = False
    citation_verdicts: list[str] = field(default_factory=list)
    total_claims: int = 0
    unsupported_claims: int = 0
    workflow_success: bool = False
    latency_ms: float = 0.0
    routing_fallback: bool = False
    routing_confidence: float = 0.0
    specialty_pair: list[str] = field(default_factory=list)
    arbitration_verdict: str = ""
    compliance_blocked: bool = False
    agent_abstained: bool = False
    cache_hit: bool = False


@dataclass
class GroupResult:
    """单个消融组的结果。"""

    group: str
    sample_results: list[SampleResult] = field(default_factory=list)
    cache_stats: dict = field(default_factory=dict)

    @property
    def recall_at_5(self) -> float:
        hits = [r for r in self.sample_results if r.recall_hit]
        return len(hits) / len(self.sample_results) if self.sample_results else 0.0

    @property
    def citation_precision(self) -> float:
        all_v = [v for r in self.sample_results for v in r.citation_verdicts]
        return metrics.compute_citation_precision(all_v)

    @property
    def unsupported_claim_rate(self) -> float:
        all_v = [v for r in self.sample_results for v in r.citation_verdicts]
        return metrics.compute_unsupported_claim_rate(all_v)

    @property
    def workflow_success_rate(self) -> float:
        return metrics.compute_workflow_success_rate(
            [r.workflow_success for r in self.sample_results]
        )

    @property
    def p95_latency(self) -> float:
        return metrics.compute_p95_latency(
            [r.latency_ms for r in self.sample_results]
        )

    @property
    def routing_coverage(self) -> float:
        non_fb = [r for r in self.sample_results if not r.routing_fallback]
        return len(non_fb) / len(self.sample_results) if self.sample_results else 0.0

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "sample_count": len(self.sample_results),
            "metrics": {
                "recall_at_5": round(self.recall_at_5, 4),
                "citation_precision": round(self.citation_precision, 4),
                "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
                "workflow_success_rate": round(self.workflow_success_rate, 4),
                "p95_latency_ms": round(self.p95_latency, 2),
                "routing_coverage": round(self.routing_coverage, 4),
            },
            "cache_stats": self.cache_stats,
            "samples": [asdict(r) for r in self.sample_results],
        }


# ===== 评测流程 =====


def run_evaluation(
    cfg: dict[str, Any],
    groups: list[str],
    output_dir: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, GroupResult]:
    """运行评测。

    API 调用优化:
        1. 检索索引构建一次，所有组复用
        2. AgentOutputCache 避免重复 LLM 调用
        3. embedding_scores 跨组复用
    """
    # 1. 加载数据
    eval_set = read_jsonl(cfg["dataset"]["eval_set_path"])
    kb_chunks = read_jsonl(cfg["dataset"]["knowledge_base_path"])

    if limit:
        eval_set = eval_set[:limit]

    click.echo(f"评测集: {len(eval_set)} 样本")
    click.echo(f"知识库: {len(kb_chunks)} chunks")

    # 2. 构建检索索引（一次，所有组复用）
    chunks = [KnowledgeChunk(**c) for c in kb_chunks]
    normalizer = TerminologyNormalizer()
    click.echo("构建检索索引（embedding + BM25）...")
    retriever = Retriever(chunks, normalizer=normalizer)
    retriever.build_index(use_bm25=True, use_embedding=True)

    # 3. 初始化组件
    router = SpecialistRouter(normalizer=normalizer)
    verifier = CitationVerifier(use_nli=False)
    reviewer = ClinicalLogicReviewer()
    guard = ComplianceGuard()
    arb = ArbitrationAgent()
    llm = LLMClient() if not dry_run else None
    cache = AgentOutputCache()

    # 4. 对每个消融组运行评测
    all_results: dict[str, GroupResult] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        click.echo(f"\n--- 消融组 {group} ---")
        group_result = _run_group(
            group, eval_set, retriever, normalizer, router,
            verifier, reviewer, guard, arb, llm, cache, dry_run,
        )
        all_results[group] = group_result

        # 输出原始结果
        output_path = output_dir / f"group_{group}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(group_result.to_dict(), f, ensure_ascii=False, indent=2)
        click.echo(f"  样本数: {len(group_result.sample_results)}")
        click.echo(f"  Recall@5: {group_result.recall_at_5:.4f}")
        if not dry_run:
            click.echo(f"  Citation Precision: {group_result.citation_precision:.4f}")
            click.echo(f"  Workflow Success: {group_result.workflow_success_rate:.4f}")
        click.echo(f"  缓存命中率: {cache.hit_rate:.2%} (hits={cache.hits}, misses={cache.misses})")
        click.echo(f"  输出: {output_path}")

    return all_results


def _run_group(
    group: str,
    eval_set: list[dict],
    retriever: Retriever,
    normalizer: TerminologyNormalizer,
    router: SpecialistRouter,
    verifier: CitationVerifier,
    reviewer: ClinicalLogicReviewer,
    guard: ComplianceGuard,
    arb: ArbitrationAgent,
    llm: LLMClient | None,
    cache: AgentOutputCache,
    dry_run: bool,
) -> GroupResult:
    """运行单个消融组。"""
    result = GroupResult(group=group)

    for i, sample in enumerate(eval_set):
        if (i + 1) % 50 == 0:
            click.echo(f"  进度: {i + 1}/{len(eval_set)}")
        sr = _run_sample(
            group, sample, retriever, normalizer, router,
            verifier, reviewer, guard, arb, llm, cache, dry_run,
        )
        result.sample_results.append(sr)

    result.cache_stats = {
        "hits": cache.hits,
        "misses": cache.misses,
        "hit_rate": round(cache.hit_rate, 4),
    }
    return result


def _run_sample(
    group: str,
    sample: dict,
    retriever: Retriever,
    normalizer: TerminologyNormalizer,
    router: SpecialistRouter,
    verifier: CitationVerifier,
    reviewer: ClinicalLogicReviewer,
    guard: ComplianceGuard,
    arb: ArbitrationAgent,
    llm: LLMClient | None,
    cache: AgentOutputCache,
    dry_run: bool,
) -> SampleResult:
    """运行单个样本评测。"""
    start = time.time()
    sample_id = sample["sample_id"]
    question = sample["question"]
    gold_ids = sample.get("gold_evidence_ids", [])
    options = sample.get("options")

    sr = SampleResult(sample_id=sample_id, group=group)

    # 1. 术语归一化（D/F 组）
    use_norm = ABLATION_CONFIGS.get(group, {}).get("use_term_normalization", False)
    search_query = normalizer.normalize(question).normalized if use_norm else question

    # 2. 检索（复用 embedding_scores）
    search_results = retriever.search(search_query, top_k=5, ablation_group=group)
    evidence_chunks = [r.chunk for r in search_results if r.chunk]
    evidence_ids = [r.chunk_id for r in search_results]

    # 3. Recall@5
    if gold_ids:
        sr.recall_hit = len(set(evidence_ids) & set(gold_ids)) > 0

    # 4. dry-run: 只计算检索指标
    if dry_run:
        sr.workflow_success = sr.recall_hit
        sr.latency_ms = (time.time() - start) * 1000
        return sr

    # 5. 诊断生成
    if group == "A":
        # 单 Agent baseline
        sr.specialty_pair = ["general_diagnosis"]
        sr = _run_single_agent(
            sr, question, evidence_chunks, evidence_ids, options,
            llm, cache, reviewer, verifier, guard,
        )
    else:
        # 双专科
        if group == "B":
            sr.specialty_pair = list(BASELINE_PAIR)
            sr.routing_fallback = False
            sr.routing_confidence = 1.0
        else:
            routing = router.route(question, evidence_chunks, "")
            sr.specialty_pair = list(routing.specialty_pair)
            sr.routing_fallback = routing.is_fallback
            sr.routing_confidence = routing.confidence

        sr = _run_dual_specialist(
            sr, question, evidence_chunks, evidence_ids, options,
            llm, cache, arb, verifier, guard,
        )

    sr.latency_ms = (time.time() - start) * 1000
    return sr


def _run_single_agent(
    sr: SampleResult,
    question: str,
    evidence: list,
    evidence_ids: list[str],
    options: dict | None,
    llm: LLMClient | None,
    cache: AgentOutputCache,
    reviewer: ClinicalLogicReviewer,
    verifier: CitationVerifier,
    guard: ComplianceGuard,
) -> SampleResult:
    """单 Agent 流程（A 组 baseline）。"""
    agent = DiagnosisAgent(llm)
    ih = cache.compute_hash(question, evidence_ids, "general_diagnosis", "")
    cached = cache.get(ih)
    if cached:
        output = cached
        sr.cache_hit = True
    else:
        output = agent.generate(question, evidence, "", options)
        cache.set(ih, output)

    sr.agent_abstained = output.abstain
    review = reviewer.check(output)
    sr.workflow_success = review.is_approved and not output.abstain

    # 引用校验
    claims = [{"text": c.text, "citation_chunk_ids": c.citation_chunk_ids} for c in output.claims]
    cit_results = verifier.verify_batch(claims, evidence)
    sr.citation_verdicts = [cr.verdict.value for cr in cit_results]
    sr.total_claims = len(claims)
    sr.unsupported_claims = sum(1 for v in sr.citation_verdicts if v == "UNSUPPORTED")

    # 合规
    sr.compliance_blocked = guard.check_output(output.to_dict()).blocked
    return sr


def _run_dual_specialist(
    sr: SampleResult,
    question: str,
    evidence: list,
    evidence_ids: list[str],
    options: dict | None,
    llm: LLMClient | None,
    cache: AgentOutputCache,
    arb: ArbitrationAgent,
    verifier: CitationVerifier,
    guard: ComplianceGuard,
) -> SampleResult:
    """双专科流程（B/C 组）。"""
    outputs = []
    for specialty in sr.specialty_pair:
        agent = SpecialistAgent(specialty, llm)
        ih = cache.compute_hash(question, evidence_ids, specialty, "")
        cached = cache.get(ih)
        if cached:
            outputs.append(cached)
            sr.cache_hit = True
        else:
            output = agent.generate(question, evidence, "", options)
            cache.set(ih, output)
            outputs.append(output)

    sr.agent_abstained = any(o.abstain for o in outputs)

    # 仲裁
    if len(outputs) == 2 and not outputs[0].abstain and not outputs[1].abstain:
        arb_result = arb.arbitrate(outputs[0], outputs[1])
        sr.arbitration_verdict = arb_result.verdict
        sr.workflow_success = arb_result.verdict == "APPROVED"
    else:
        sr.arbitration_verdict = "ESCALATED"
        sr.workflow_success = False

    # 引用校验（合并两方 claims）
    all_claims = []
    for output in outputs:
        for claim in output.claims:
            all_claims.append({"text": claim.text, "citation_chunk_ids": claim.citation_chunk_ids})
    cit_results = verifier.verify_batch(all_claims, evidence)
    sr.citation_verdicts = [cr.verdict.value for cr in cit_results]
    sr.total_claims = len(all_claims)
    sr.unsupported_claims = sum(1 for v in sr.citation_verdicts if v == "UNSUPPORTED")

    # 合规
    for output in outputs:
        if guard.check_output(output.to_dict()).blocked:
            sr.compliance_blocked = True
            break
    return sr


# ===== CLI =====


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=False), default="eval/config.yaml", show_default=True)
@click.option("--group", type=click.Choice(VALID_GROUPS, case_sensitive=False), default="all", show_default=True)
@click.option("--output", "output_dir", type=click.Path(exists=False), default="reports/raw/", show_default=True)
@click.option("--show-config", is_flag=True, help="仅打印配置摘要并退出。")
@click.option("--validate", is_flag=True, help="仅校验配置完整性并退出。")
@click.option("--dry-run", is_flag=True, help="只计算检索指标，不调 LLM。")
@click.option("--limit", type=int, default=None, help="限制样本数（调试用）。")
def cli(
    config_path: str,
    group: str,
    output_dir: str,
    show_config: bool,
    validate: bool,
    dry_run: bool,
    limit: int | None,
) -> None:
    """MediDiag 评测 runner。

    API 调用优化: input_hash 缓存 + 批量 embedding + 检索结果跨组复用。
    """
    cfg = load_config(config_path)

    if show_config:
        show_config_summary(cfg)
        return

    issues = validate_config(cfg)
    if issues:
        click.echo("Configuration validation FAILED:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(2)

    if validate:
        click.echo("Configuration validation: OK")
        return

    # 确定消融组别
    groups = list("ABCDEF") if group.lower() == "all" else [group.upper()]

    click.echo("=" * 70)
    click.echo("MediDiag Eval Runner")
    click.echo("=" * 70)
    click.echo(f"  config  : {config_path}")
    click.echo(f"  groups  : {groups}")
    click.echo(f"  output  : {output_dir}")
    click.echo(f"  dry_run : {dry_run}")
    click.echo(f"  limit   : {limit}")
    click.echo("")

    results = run_evaluation(
        cfg, groups, Path(output_dir), limit=limit, dry_run=dry_run,
    )

    # 打印汇总
    click.echo("\n" + "=" * 70)
    click.echo("评测汇总")
    click.echo("=" * 70)
    click.echo(f"{'组':>4s}  {'Recall@5':>10s}  {'Citation':>10s}  {'Unsupp':>10s}  {'Success':>10s}  {'P95(ms)':>10s}")
    for g, gr in results.items():
        click.echo(
            f"{g:>4s}  {gr.recall_at_5:>10.4f}  {gr.citation_precision:>10.4f}  "
            f"{gr.unsupported_claim_rate:>10.4f}  {gr.workflow_success_rate:>10.4f}  {gr.p95_latency:>10.1f}"
        )
    click.echo("=" * 70)


if __name__ == "__main__":
    cli()
