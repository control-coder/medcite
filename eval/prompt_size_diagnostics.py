"""测量 agent 族的 prompt 体积，不调用 generation、judge 或 DeepSeek。

这是**工程体积测量**，不产生任何评测指标。它复现 `eval/runner.py` 中三组 agent
实验的检索链路（固定 `rag_full` profile，term normalization ->
`search(candidate_k=10)` -> `rerank(top_k=5)`），然后调用真实的
`BaseAgent.build_prompt` 拼出每次 provider 调用**会**发送的 prompt，只统计字符数。

为什么需要这个工具：`--dry-run` 在 `_run_sample` 中于 generation 之前就返回
（`eval/runner.py`），因此 dry-run 本身测不到 prompt 体积。2026-07-26 的首次测量
是临时脚本，本工具把它固化为可复现命令，使 `doc/evaluation_protocol.md` 的
体积门禁能被逐条重跑。

用法::

    python -m eval.prompt_size_diagnostics --config eval/config.formal.yaml
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from eval.configuration import load_config, validate_config
from medidiag.acceleration import runtime_snapshot
from medidiag.agents.base import AGENT_SYSTEM_PROMPT
from medidiag.agents.diagnosis import DiagnosisAgent
from medidiag.agents.router import SpecialistRouter
from medidiag.agents.specialist import SpecialistAgent
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever
from medidiag.schemas import KnowledgeChunk, read_jsonl

ROOT = Path(__file__).resolve().parent.parent

# 英文 BPE 的经验区间；prompt 内容 99%+ 为 ASCII 英文，固定指令部分含少量 CJK。
CHARS_PER_TOKEN = (3.0, 3.5, 4.0, 4.5)

# 常见上下文窗口分档（token）。
CONTEXT_WINDOWS = (32_768, 65_536, 131_072, 262_144)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _percentiles(values: list[int]) -> dict[str, int]:
    """无依赖的分位数（最近秩），空输入返回全 0。"""
    if not values:
        return {key: 0 for key in ("min", "p50", "p90", "p99", "max")}
    ordered = sorted(values)
    last = len(ordered) - 1

    def at(fraction: float) -> int:
        return ordered[min(last, max(0, round(fraction * last)))]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _overflow_counts(prompt_chars: list[int]) -> dict[str, dict[str, int]]:
    """在每个字符/token 假设下，统计超过各上下文窗口的**调用**次数。"""
    result: dict[str, dict[str, int]] = {}
    for ratio in CHARS_PER_TOKEN:
        key = f"chars_per_token_{ratio}"
        result[key] = {
            f"over_{window // 1024}k_tokens": sum(
                1 for chars in prompt_chars if chars / ratio > window
            )
            for window in CONTEXT_WINDOWS
        }
    return result


def run_prompt_size_diagnostics(
    config: dict[str, Any], limit: int | None = None
) -> dict[str, Any]:
    """测量全部会调用 generation 的实验的 prompt 字符量。

    覆盖两族：三组 agent 实验（MedQA 100 样本 manifest，`rag_full` profile）与
    需要 citation review 的 rag 实验（PubMedQA 300 样本，各自的 profile，
    ``topology="single"``，见 `eval/runner.py:413`）。
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
    evidence_levels = config["retrieval"]["evidence_levels"]
    agent_experiments = config["experiments"]["agent"]

    records = read_jsonl(_resolve(config["dataset"]["agent_eval_set_path"]))
    by_id = {str(record["sample_id"]): record for record in records}
    manifest = read_jsonl(_resolve(config["dataset"]["agent_sample_manifest_path"]))
    ordered = [
        by_id[str(item["sample_id"])]
        for item in manifest
        if str(item["sample_id"]) in by_id
    ]
    if limit is not None:
        ordered = ordered[:limit]

    system_chars = len(AGENT_SYSTEM_PROMPT)
    router = SpecialistRouter(
        normalizer=normalizer, evidence_level_scores=evidence_levels
    )
    # 每个实验 -> 每次 provider 调用的 prompt 字符数（含 system prompt）。
    per_experiment: dict[str, list[int]] = {
        name: [] for name in agent_experiments
    }
    evidence_chars: list[int] = []

    for record in ordered:
        question = str(record["question"])
        options = record.get("options")
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
        evidence_chars.append(sum(len(chunk.text) for chunk in evidence))

        for name, experiment in agent_experiments.items():
            topology = experiment["topology"]
            if topology == "single":
                specialties = ["general_diagnosis"]
            elif topology == "fixed_pair":
                specialties = list(experiment["specialist_pair"])
            else:
                specialties = list(router.route(question, evidence).specialty_pair)
            for specialty in specialties:
                agent = (
                    DiagnosisAgent(None)
                    if specialty == "general_diagnosis"
                    else SpecialistAgent(specialty, None)
                )
                prompt = agent.build_prompt(question, evidence, "", options)
                per_experiment[name].append(len(prompt) + system_chars)

    # rag 族中需要 citation review 的实验也会调用 generation，且各自使用自己的
    # 检索 profile 与 PubMedQA 评测集。
    rag_records = read_jsonl(_resolve(config["dataset"]["rag_eval_set_path"]))
    if limit is not None:
        rag_records = rag_records[:limit]
    rag_generation = {
        name: experiment
        for name, experiment in config["experiments"]["rag"].items()
        if experiment["config"].get("use_citation_review")
    }
    for name, experiment in rag_generation.items():
        profile = experiment["config"]
        per_experiment[name] = []
        agent = DiagnosisAgent(None)
        for record in rag_records:
            question = str(record["question"])
            retrieval_query = question
            if profile["use_term_normalization"]:
                retrieval_query = normalizer.normalize(question).normalized
            search_results = retriever.search(
                retrieval_query,
                top_k=candidate_k if profile["use_rerank"] else top_k,
                experiment_config=profile,
            )
            if profile["use_rerank"]:
                search_results = retriever.rerank(
                    retrieval_query, search_results, top_k=top_k
                )
            evidence = [
                item.chunk for item in search_results if item.chunk is not None
            ]
            prompt = agent.build_prompt(
                question, evidence, "", record.get("options")
            )
            per_experiment[name].append(len(prompt) + system_chars)

    all_chars = [
        chars
        for name in agent_experiments
        for chars in per_experiment[name]
    ]
    every_chars = [chars for values in per_experiment.values() for chars in values]
    return {
        "sample_count": len(ordered),
        "sample_set": "agent_manifest_v1",
        "config_mode": config["evaluation"]["mode"],
        "retrieval_profile": "rag_full",
        "knowledge_base_chunk_count": len(chunks),
        "knowledge_base_total_chars": sum(len(chunk.text) for chunk in chunks),
        "system_prompt_chars": system_chars,
        "max_tokens_per_call": config["generation"]["max_tokens"],
        "evidence_chars_per_sample": _percentiles(evidence_chars),
        "experiments": {
            name: {
                "provider_calls": len(values),
                "prompt_chars": _percentiles(values),
                "prompt_chars_total": sum(values),
                "context_window_overflow_calls": _overflow_counts(values),
            }
            for name, values in per_experiment.items()
        },
        "agent_total": {
            "provider_calls": len(all_chars),
            "prompt_chars": _percentiles(all_chars),
            "prompt_chars_total": sum(all_chars),
            "context_window_overflow_calls": _overflow_counts(all_chars),
        },
        # `--experiment all` 会触发的全部 generation 调用（agent 三组 + rag 两组）。
        "all_generation_total": {
            "provider_calls": len(every_chars),
            "prompt_chars": _percentiles(every_chars),
            "prompt_chars_total": sum(every_chars),
            "context_window_overflow_calls": _overflow_counts(every_chars),
        },
        "runtime": runtime_snapshot(
            str(runtime.get("device", "auto")), retriever.actual_device, batch_size
        ),
    }


@click.command()
@click.option("--config", default="eval/config.formal.yaml", show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option(
    "--output", default="reports/prompt_size_diagnostics.json", show_default=True
)
def cli(config: str, limit: int | None, output: str) -> None:
    loaded = load_config(config)
    issues = validate_config(loaded, ROOT)
    if issues:
        raise click.UsageError("invalid evaluation config:\n" + "\n".join(issues))
    report = run_prompt_size_diagnostics(loaded, limit)
    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    click.echo(f"samples={report['sample_count']} set={report['sample_set']}")
    for name, stats in report["experiments"].items():
        dist = stats["prompt_chars"]
        click.echo(
            f"  {name:<20} calls={stats['provider_calls']:<4} "
            f"p50={dist['p50']} p90={dist['p90']} max={dist['max']}"
        )
    for key in ("agent_total", "all_generation_total"):
        total = report[key]
        dist = total["prompt_chars"]
        click.echo(
            f"  {key:<20} calls={total['provider_calls']:<4} "
            f"p50={dist['p50']} p90={dist['p90']} max={dist['max']}"
        )
        click.echo(
            "    overflow @4.0 chars/token: "
            f"{total['context_window_overflow_calls']['chars_per_token_4.0']}"
        )
    click.echo(f"output={output_path}")


if __name__ == "__main__":
    cli()
