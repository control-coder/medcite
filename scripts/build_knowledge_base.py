"""构建知识库：合并 PubMedQA chunks + MedQA textbook 段落，合并评测集。

输入:
  - eval/datasets/chunks_pubmedqa.jsonl    (PubMedQA CONTEXTS chunks)
  - eval/datasets/eval_set_pubmedqa.jsonl  (PubMedQA 评测样本)
  - eval/datasets/eval_set_medqa.jsonl     (MedQA 评测样本)
  - eval/datasets/raw/medqa_data/data_clean/textbooks/en/*.txt (MedQA 教材)

输出:
  - eval/datasets/knowledge_chunks.jsonl   (合并后的知识库)
  - eval/datasets/eval_set.jsonl           (合并后的评测集)

数据泄露防护:
  - textbook chunk 的 source_id = 教材名（如 "Anatomy_Gray"），非 sample_id
  - 构建时过滤包含评测问题原文（len > 50）的 chunk
  - 合并后由 leakage_check 最终校验
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import click

from medidiag.schemas import (
    KnowledgeChunk,
    chunk_to_dict,
    read_jsonl,
    write_jsonl,
)


def split_textbook_paragraphs(content: str, min_chars: int = 50) -> list[str]:
    """按空行分割教材段落，过滤过短段落。"""
    paragraphs = []
    for block in content.split("\n\n"):
        text = block.strip()
        if len(text) >= min_chars:
            paragraphs.append(text)
    return paragraphs


@click.command()
@click.option(
    "--textbook-dir",
    default="eval/datasets/raw/medqa_data/data_clean/textbooks/en",
    show_default=True,
)
@click.option(
    "--pubmedqa-chunks", default="eval/datasets/chunks_pubmedqa.jsonl",
    show_default=True,
)
@click.option(
    "--pubmedqa-eval", default="eval/datasets/eval_set_pubmedqa.jsonl",
    show_default=True,
)
@click.option(
    "--medqa-eval", default="eval/datasets/eval_set_medqa.jsonl",
    show_default=True,
)
@click.option(
    "--kb-output", default="eval/datasets/knowledge_chunks.jsonl",
    show_default=True,
)
@click.option(
    "--eval-output", default="eval/datasets/eval_set.jsonl",
    show_default=True,
)
@click.option(
    "--chunks-per-book", type=int, default=50,
    show_default=True,
    help="每本教材取前 N 段（控制知识库规模）。",
)
def cli(
    textbook_dir: str,
    pubmedqa_chunks: str,
    pubmedqa_eval: str,
    medqa_eval: str,
    kb_output: str,
    eval_output: str,
    chunks_per_book: int,
) -> None:
    """构建知识库与合并评测集。"""

    # 1. 加载 PubMedQA chunks
    pmc_records = read_jsonl(pubmedqa_chunks)
    chunks: list[KnowledgeChunk] = [KnowledgeChunk(**rec) for rec in pmc_records]
    click.echo(f"PubMedQA chunks: {len(chunks)}")

    # 2. 处理 MedQA textbooks
    textbook_path = Path(textbook_dir)
    if not textbook_path.exists():
        click.echo(f"WARNING: textbook dir not found: {textbook_dir}")
    else:
        chunk_seq = 0
        txt_files = sorted(textbook_path.glob("*.txt"))
        for txt_file in txt_files:
            book_name = txt_file.stem
            content = txt_file.read_text(encoding="utf-8", errors="ignore")
            paragraphs = split_textbook_paragraphs(content)
            paragraphs = paragraphs[:chunks_per_book]

            for para in paragraphs:
                chunk_id = f"kb_textbook_{chunk_seq:05d}"
                chunk_seq += 1
                chunk = KnowledgeChunk(
                    chunk_id=chunk_id,
                    source="MedQA_textbook",
                    source_id=book_name,  # 教材名，非 sample_id
                    text=para,
                    evidence_level="level_2_review",  # 教材 = 综述级别
                    metadata={"textbook_name": book_name},
                )
                chunks.append(chunk)

        click.echo(f"MedQA textbook chunks: {chunk_seq} (from {len(txt_files)} books, {chunks_per_book}/book)")

    # 3. 合并评测集
    eval_samples: list[dict] = []
    for eval_file in [pubmedqa_eval, medqa_eval]:
        eval_path = Path(eval_file)
        if eval_path.exists():
            eval_samples.extend(read_jsonl(eval_file))

    # 3.5 数据泄露防护：提取评测问题原文，过滤包含问题的 chunk
    #     避免 RAG 检索时直接匹配到评测问题（len > 50 的问题才检查）
    eval_questions = {
        s["question"].strip()
        for s in eval_samples
        if len(s.get("question", "")) > 50
    }
    before_filter = len(chunks)
    chunks = [
        c for c in chunks
        if not any(q in c.text for q in eval_questions)
    ]
    filtered_count = before_filter - len(chunks)
    if filtered_count > 0:
        click.echo(f"泄露过滤: 移除 {filtered_count} 个包含评测问题原文的 chunk")

    # 4. 写入
    n_chunks = write_jsonl(
        [chunk_to_dict(c) for c in chunks], kb_output
    )
    n_samples = write_jsonl(eval_samples, eval_output)

    # 5. 统计
    public = sum(1 for s in eval_samples if s.get("source") in ("PubMedQA", "MedQA"))
    has_evidence = sum(1 for s in eval_samples if s.get("gold_evidence_ids"))

    # chunk 来源分布
    source_dist: dict[str, int] = {}
    for c in chunks:
        source_dist[c.source] = source_dist.get(c.source, 0) + 1

    click.echo("")
    click.echo("========== 知识库构建完成 ==========")
    click.echo(f"知识库 chunks 总数  : {n_chunks}")
    for src, cnt in sorted(source_dist.items()):
        click.echo(f"  {src:25s}: {cnt}")
    click.echo(f"评测集样本总数      : {n_samples}")
    click.echo(f"公开题占比          : {public}/{n_samples} ({public / n_samples * 100:.1f}%)")
    click.echo(f"有 gold_evidence    : {has_evidence}/{n_samples} ({has_evidence / n_samples * 100:.1f}%)")
    click.echo(f"  kb_output  -> {kb_output}")
    click.echo(f"  eval_output-> {eval_output}")


if __name__ == "__main__":
    cli()
