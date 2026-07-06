"""改造 PubMedQA 数据集为评测集 + 知识库 chunks。

输入: eval/datasets/raw/pubmedqa/data/ori_pqal.json (1000 labeled samples)
输出:
  - eval/datasets/eval_set_pubmedqa.jsonl  (评测样本)
  - eval/datasets/chunks_pubmedqa.jsonl    (知识库 chunks，来自 CONTEXTS)

数据泄露防护:
  - sample_id = "pubmedqa_{pubmed_id}"（带前缀）
  - chunk.source_id = "{pubmed_id}"（文献来源 ID，非 sample_id）
  - leakage_check 检查 sample_id 是否出现在 chunk 字段 → 不会命中（前缀不同）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 让脚本可直接运行（无需 PYTHONPATH）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import click

from medidiag.schemas import (
    EvalSample,
    KnowledgeChunk,
    chunk_to_dict,
    sample_to_dict,
    write_jsonl,
)


# CONTEXT 标签 -> 证据等级
EVIDENCE_LEVEL_MAP = {
    "BACKGROUND": "level_2_review",
    "OBJECTIVE": "level_2_review",
    "METHODS": "level_3_primary_study",
    "RESULTS": "level_3_primary_study",
    "CONCLUSIONS": "level_3_primary_study",
}

# gold evidence 优先取 RESULTS / CONCLUSIONS 段落
GOLD_LABELS = {"RESULTS", "CONCLUSIONS"}


@click.command()
@click.option(
    "--input", "input_path",
    type=click.Path(exists=False),
    default="eval/datasets/raw/pubmedqa/data/ori_pqal.json",
    show_default=True,
    help="PubMedQA ori_pqal.json 路径。",
)
@click.option(
    "--eval-output", "eval_output",
    default="eval/datasets/eval_set_pubmedqa.jsonl",
    show_default=True,
)
@click.option(
    "--chunk-output", "chunk_output",
    default="eval/datasets/chunks_pubmedqa.jsonl",
    show_default=True,
)
@click.option(
    "--limit", type=int, default=None,
    help="限制样本数（调试用）。默认全部。",
)
def cli(
    input_path: str,
    eval_output: str,
    chunk_output: str,
    limit: int | None,
) -> None:
    """改造 PubMedQA 数据集。"""

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    items = list(data.items())
    if limit:
        items = items[:limit]

    samples: list[EvalSample] = []
    chunks: list[KnowledgeChunk] = []
    chunk_seq = 0

    for pubmed_id, item in items:
        sample_id = f"pubmedqa_{pubmed_id}"
        question = item["QUESTION"]
        gold_answer = item["final_decision"]
        contexts = item.get("CONTEXTS", [])
        labels = item.get("LABELS", [])

        # 为每个 context 段落生成 chunk
        gold_evidence_ids: list[str] = []
        for ctx_text, label in zip(contexts, labels):
            chunk_id = f"kb_pubmedqa_{chunk_seq:05d}"
            chunk_seq += 1
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                source="PubMedQA_context",
                source_id=str(pubmed_id),  # 文献来源 ID，非 sample_id
                text=ctx_text,
                evidence_level=EVIDENCE_LEVEL_MAP.get(label, "level_5_other"),
                metadata={
                    "context_label": label,
                    "pubmed_id": str(pubmed_id),
                },
            )
            chunks.append(chunk)
            if label in GOLD_LABELS:
                gold_evidence_ids.append(chunk_id)

        sample = EvalSample(
            sample_id=sample_id,
            source="PubMedQA",
            question=question,
            gold_answer=gold_answer,
            gold_evidence_ids=gold_evidence_ids,
            label_source="dataset",
            labeler="dataset",
            review_status="single_checked",
            options=None,
            metadata={
                "meshes": item.get("MESHES", []),
                "year": item.get("YEAR", ""),
                "long_answer": item.get("LONG_ANSWER", ""),
            },
        )
        samples.append(sample)

    n_samples = write_jsonl(
        [sample_to_dict(s) for s in samples], eval_output
    )
    n_chunks = write_jsonl(
        [chunk_to_dict(c) for c in chunks], chunk_output
    )

    has_evidence = sum(1 for s in samples if s.gold_evidence_ids)
    click.echo(f"PubMedQA 改造完成:")
    click.echo(f"  样本数          : {n_samples}")
    click.echo(f"  知识库 chunks   : {n_chunks}")
    click.echo(f"  有 gold_evidence: {has_evidence}/{n_samples} ({has_evidence / n_samples * 100:.1f}%)")
    click.echo(f"  eval  -> {eval_output}")
    click.echo(f"  chunks-> {chunk_output}")


if __name__ == "__main__":
    cli()
