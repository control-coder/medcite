"""输出可审计的 RAG top-k 诊断结果，不调用 generation、judge 或 DeepSeek。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from eval.configuration import get_experiment, load_config, validate_config
from medidiag.acceleration import runtime_snapshot
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever
from medidiag.schemas import KnowledgeChunk, read_jsonl

ROOT = Path(__file__).resolve().parent.parent


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_diagnostics(
    config: dict[str, Any], experiment_name: str, limit: int | None = None
) -> dict[str, Any]:
    family, experiment = get_experiment(config, experiment_name)
    if family != "rag":
        raise click.UsageError("diagnostics requires a RAG experiment")
    chunks = [
        KnowledgeChunk(**record)
        for record in read_jsonl(_resolve(config["dataset"]["knowledge_base_path"]))
    ]
    normalizer = TerminologyNormalizer()
    runtime = config.get("runtime", {})
    batch_size = int(runtime.get("batch_size", config["embedding"].get("batch_size", 32)))
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
    records = read_jsonl(_resolve(config["dataset"]["rag_eval_set_path"]))
    if limit is not None:
        records = records[:limit]
    rag_config = experiment["config"]
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    empty_chunks = sum(not chunk.text.strip() for chunk in chunks)
    eligible = 0
    gold_in_kb = 0
    hits = 0
    samples: list[dict[str, Any]] = []
    for record in records:
        gold_ids = [str(value) for value in record.get("gold_evidence_ids", [])]
        gold_present = bool(gold_ids) and bool(set(gold_ids) & chunk_ids)
        if gold_ids:
            eligible += 1
            gold_in_kb += int(gold_present)
        query = str(record["question"])
        retrieval_query = query
        if rag_config["use_term_normalization"]:
            retrieval_query = normalizer.normalize(query).normalized
        results = retriever.search(
            retrieval_query,
            top_k=int(config["retrieval"]["top_k"]),
            experiment_config=rag_config,
        )
        top = [
            {
                "chunk_id": item.chunk_id,
                "final_score": round(item.final_score, 6),
                "embedding_score": round(item.embedding_score, 6),
                "bm25_score": round(item.bm25_score, 6),
                "evidence_level_score": round(item.evidence_level_score, 6),
                "term_overlap": round(item.term_overlap, 6),
                "source": item.chunk.source if item.chunk else "",
                "source_id": item.chunk.source_id if item.chunk else "",
                "text_length": len(item.chunk.text) if item.chunk else 0,
            }
            for item in results
        ]
        hit = bool(set(item["chunk_id"] for item in top) & set(gold_ids)) if gold_ids else None
        hits += int(bool(hit)) if gold_ids else 0
        samples.append(
            {
                "sample_id": str(record["sample_id"]),
                "question": query,
                "gold_evidence_ids": gold_ids,
                "gold_present_in_kb": gold_present,
                "hit_at_5": hit,
                "top_5": top,
            }
        )
    return {
        "experiment": experiment_name,
        "sample_count": len(records),
        "eligible_sample_count": eligible,
        "gold_evidence_in_kb_rate": gold_in_kb / eligible if eligible else 0.0,
        "recall_at_5": hits / eligible if eligible else 0.0,
        "knowledge_base_chunk_count": len(chunks),
        "empty_chunk_count": empty_chunks,
        "runtime": runtime_snapshot(
            str(runtime.get("device", "auto")), retriever.actual_device, batch_size
        ),
        "retrieval_cache": retriever.cache_stats(),
        "samples": samples,
    }


@click.command()
@click.option("--config", default="eval/config.formal.yaml", show_default=True)
@click.option("--experiment", default="rag_embedding", show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
@click.option("--output", default="reports/retrieval_diagnostics.json", show_default=True)
def cli(config: str, experiment: str, limit: int | None, output: str) -> None:
    loaded = load_config(config)
    issues = validate_config(loaded, ROOT)
    if issues:
        raise click.UsageError("invalid evaluation config:\n" + "\n".join(issues))
    report = run_diagnostics(loaded, experiment, limit)
    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(
        f"{experiment}: Recall@5={report['recall_at_5']:.4f}, "
        f"GoldInKB={report['gold_evidence_in_kb_rate']:.4f}, "
        f"output={output_path}"
    )


if __name__ == "__main__":
    cli()
