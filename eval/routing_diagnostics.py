"""输出 SpecialistRouter 的分支分布与分项得分诊断，不调用 generation 或 judge。

这是路由**代码行为**的测量工具，不产生任何评测指标。它复现
``eval/runner.py`` 中 ``agent_dynamic_pair`` 的检索链路（固定
``rag_full`` profile，term normalization -> search(candidate_k) -> rerank(top_k)），
然后对每个样本调用 ``route(question, evidence)``，统计：

- 三条路由规则各自命中多少样本，最终有多少落到兜底组合；
- top1 总分与三个分项各自达到理论上限的比例（定位「哪个分项在饿死」）；
- 歧义比值 ``top2.total / top1.total`` 的分布。

2026-07-27（DD-023）起分项只有三个：``plan_hint`` 分项已从打分公式中删除，因此
名义上限 7.0 与可达上限相同，不再区分 ``theoretical_max_total`` 与
``reachable_max_total``。

2026-07-28（DD-025）起规则只有三条：原规则 4 按歧义比值兜底，方向颠倒，已删除。
输出中 ``branch_counts`` 不再含 ``rule_4_low_confidence``，``rule_4_with_both_
scores_eligible`` 字段一并移除；历史 ``artifacts/reports/routing_diagnostics_*_fixed.json``
仍带这两项。歧义比值仍在 ``confidence_distribution`` 中报告。

用法::

    python -m eval.routing_diagnostics --config eval/config.formal.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from eval.configuration import load_config, validate_config
from medidiag.acceleration import runtime_snapshot
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialty_data import (
    MATCH_SATURATION_COUNT,
    ROUTING_WEIGHTS,
    THRESHOLDS,
)
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever
from medidiag.schemas import KnowledgeChunk, read_jsonl

ROOT = Path(__file__).resolve().parent.parent

COMPONENTS = ("keyword", "normalized_term", "evidence")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _percentiles(values: list[float]) -> dict[str, float]:
    """无依赖的分位数（最近秩），空输入返回全 0。"""
    if not values:
        return {key: 0.0 for key in ("min", "p10", "p50", "p90", "max")}
    ordered = sorted(values)
    last = len(ordered) - 1

    def at(fraction: float) -> float:
        return ordered[min(last, max(0, round(fraction * last)))]

    return {
        "min": round(ordered[0], 4),
        "p10": round(at(0.10), 4),
        "p50": round(at(0.50), 4),
        "p90": round(at(0.90), 4),
        "max": round(ordered[-1], 4),
    }


def _classify(top1_total: float, top2_total: float) -> str:
    """复现 ``SpecialistRouter.route`` 的规则顺序，返回命中的规则名。

    2026-07-28（DD-025）起只有三条规则：原规则 4 已删除，因此本函数不再需要
    ``confidence``。歧义度比值仍单独统计分布，只是不再决定分支。
    """
    th = THRESHOLDS
    if top1_total < th["MIN_PRIMARY_SCORE"]:
        return "rule_1_primary_below_threshold"
    if (
        top2_total < th["MIN_SECONDARY_SCORE"]
        or (top1_total - top2_total) >= th["SCORE_GAP"]
    ):
        return "rule_2_top1_plus_skeptic"
    return "rule_3_dynamic_top2"


def run_routing_diagnostics(
    config: dict[str, Any], limit: int | None = None, holdout: bool = False
) -> dict[str, Any]:
    """对 agent manifest 的全部样本测量路由分支分布。

    ``holdout=True`` 时改为测量 agent eval set 中**不在** manifest 里的样本。
    关键词表是人工编写的，编写者看得到 manifest 的词频统计；holdout 提供一个
    未被查看过的对照集，用于判断分布变化是词表泛化还是对 manifest 的过拟合。
    """
    chunks = [
        KnowledgeChunk(**record)
        for record in read_jsonl(_resolve(config["dataset"]["knowledge_base_path"]))
    ]
    normalizer = TerminologyNormalizer()
    runtime = config.get("runtime", {})
    batch_size = int(
        runtime.get("batch_size", config["embedding"].get("batch_size", 32))
    )
    retriever = Retriever(
        chunks,
        weights=config["retrieval"]["weights"],
        evidence_level_scores=config["retrieval"]["evidence_levels"],
        embedding_model=config["embedding"]["model"],
        rerank_model=config["rerank"]["model"],
        normalizer=normalizer,
        embedding_revision=config["embedding"]["revision"],
        rerank_revision=config["rerank"]["revision"],
        device=str(runtime.get("device", "auto")),
        embedding_batch_size=batch_size,
        rerank_batch_size=batch_size,
    )
    retriever.build_index(use_bm25=True, use_embedding=True)

    # agent 族固定使用 rag_full 的检索配置（DD-004）。
    rag_config = config["experiments"]["rag"]["rag_full"]["config"]
    top_k = int(config["retrieval"]["top_k"])
    candidate_k = int(config["retrieval"]["candidate_k"])

    records = read_jsonl(_resolve(config["dataset"]["agent_eval_set_path"]))
    by_id = {str(record["sample_id"]): record for record in records}
    manifest = read_jsonl(_resolve(config["dataset"]["agent_sample_manifest_path"]))
    manifest_ids = [str(item["sample_id"]) for item in manifest]
    if holdout:
        selected = set(manifest_ids)
        ordered = [
            record
            for record in records
            if str(record["sample_id"]) not in selected
        ]
    else:
        ordered = [
            by_id[sample_id] for sample_id in manifest_ids if sample_id in by_id
        ]
    if limit is not None:
        ordered = ordered[:limit]

    router = SpecialistRouter(
        normalizer=normalizer,
        evidence_level_scores=config["retrieval"]["evidence_levels"],
    )

    branch_counts: dict[str, int] = {
        "rule_1_primary_below_threshold": 0,
        "rule_2_top1_plus_skeptic": 0,
        "rule_3_dynamic_top2": 0,
    }
    pair_counts: dict[str, int] = {}
    top1_totals: list[float] = []
    confidences: list[float] = []
    component_values: dict[str, list[float]] = {key: [] for key in COMPONENTS}
    fallback_count = 0
    samples: list[dict[str, Any]] = []

    for record in ordered:
        question = str(record["question"])
        retrieval_query = question
        if rag_config["use_term_normalization"]:
            retrieval_query = normalizer.normalize(question).normalized
        search_results = retriever.search(
            retrieval_query,
            top_k=candidate_k if rag_config["use_rerank"] else top_k,
            experiment_config=rag_config,
        )
        if rag_config["use_rerank"]:
            search_results = retriever.rerank(
                retrieval_query, search_results, top_k=top_k
            )
        evidence = [item.chunk for item in search_results if item.chunk is not None]

        result = router.route(question, evidence)
        ranked = sorted(
            result.scores.values(), key=lambda s: s.total, reverse=True
        )
        top1 = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None
        top2_total = top2.total if top2 else 0.0
        branch = _classify(top1.total, top2_total)

        branch_counts[branch] += 1
        fallback_count += int(result.is_fallback)
        pair_key = " + ".join(result.specialty_pair)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        top1_totals.append(top1.total)
        confidences.append(result.confidence)
        component_values["keyword"].append(top1.keyword_score)
        component_values["normalized_term"].append(top1.normalized_term_score)
        component_values["evidence"].append(top1.evidence_score)

        samples.append(
            {
                "sample_id": str(record["sample_id"]),
                "question_length": len(question),
                "branch": branch,
                "specialty_pair": list(result.specialty_pair),
                "is_fallback": result.is_fallback,
                "confidence": round(result.confidence, 4),
                "top1": top1.to_dict(),
                "top2": top2.to_dict() if top2 else None,
            }
        )

    total = len(ordered)
    theoretical_max = sum(ROUTING_WEIGHTS.values())
    return {
        "sample_count": total,
        "sample_set": "agent_manifest_holdout" if holdout else "agent_manifest_v1",
        "config_mode": config["evaluation"]["mode"],
        "retrieval_profile": "rag_full",
        "thresholds": dict(THRESHOLDS),
        "routing_weights": dict(ROUTING_WEIGHTS),
        "theoretical_max_total": theoretical_max,
        # DD-023 起三个分项的分母都与词表长度无关，因此名义上限就是可达上限。
        "reachable_max_total": theoretical_max,
        "match_saturation_count": MATCH_SATURATION_COUNT,
        "branch_counts": branch_counts,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "specialty_pair_counts": dict(
            sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "top1_total_distribution": _percentiles(top1_totals),
        "confidence_distribution": _percentiles(confidences),
        "top1_component_distribution": {
            key: _percentiles(values) for key, values in component_values.items()
        },
        "top1_component_mean_share_of_weighted_max": {
            key: round(
                sum(values) / len(values), 4
            )
            if values
            else 0.0
            for key, values in component_values.items()
        },
        "runtime": runtime_snapshot(
            str(runtime.get("device", "auto")), retriever.actual_device, batch_size
        ),
        "samples": samples,
    }


@click.command()
@click.option("--config", default="eval/config.formal.yaml", show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option(
    "--holdout",
    is_flag=True,
    help="改为测量 agent eval set 中不在 manifest 里的样本（词表泛化对照）。",
)
@click.option(
    "--output", default="artifacts/reports/routing_diagnostics.json", show_default=True
)
def cli(config: str, limit: int | None, holdout: bool, output: str) -> None:
    loaded = load_config(config)
    issues = validate_config(loaded, ROOT)
    if issues:
        raise click.UsageError("invalid evaluation config:\n" + "\n".join(issues))
    report = run_routing_diagnostics(loaded, limit, holdout)
    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    click.echo(f"samples={report['sample_count']} set={report['sample_set']}")
    for branch, count in report["branch_counts"].items():
        click.echo(f"  {branch:<34}: {count}")
    click.echo(
        f"  fallback pair total               : {report['fallback_count']} "
        f"({report['fallback_rate']:.2%})"
    )
    click.echo(f"top1 total    : {report['top1_total_distribution']}")
    # 只是诊断量，不再是门禁（DD-025）。
    click.echo(f"ambiguity     : {report['confidence_distribution']}")
    for key, dist in report["top1_component_distribution"].items():
        click.echo(f"  {key:<16}: {dist}")
    click.echo(f"output={output_path}")


if __name__ == "__main__":
    cli()
