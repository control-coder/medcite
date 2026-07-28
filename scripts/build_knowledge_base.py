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

切分长度上限:
  - 空行切分只对排版规范的教材有效。18 本教材里 `Surgery_Schwartz.txt`（11.4MB）
    只有 250 个换行、几乎没有空行段落边界，因此只能切出 126 个巨型块，最大
    722,301 字符。见 `MAX_CHUNK_CHARS` 与 `split_textbook_paragraphs`。
  - `--chunks-per-book` 的截断在切分之后，因此它与上限耦合：改动任一方都要重新
    核对教材语料总字符数，否则「切分」会静默变成「削减语料」。见 `CHUNKS_PER_BOOK`。
"""

from __future__ import annotations

import re
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

MAX_CHUNK_CHARS = 800
"""单个 chunk 的硬字符上限。

800 的依据是既有语料本身：修复前 PubMedQA + 正常教材 chunk 的 p90 恰为 800 字符，
因此这个上限对 18 本教材中排版规范的那 17 本几乎不改变切分结果，只截断异常块。它同时
落在 embedding 模型 `all-MiniLM-L6-v2` 的 256 token 输入窗口的同一量级
（约 200 token），使 chunk 能被向量完整表示——修复前 722,301 字符的 chunk 实际
只有开头约 1000 字符参与检索。
"""

CHUNKS_PER_BOOK = 650
"""每本教材取前 N 段。

该值与 `MAX_CHUNK_CHARS` 耦合：截断发生在切分之后，因此上限从「一整章」降到 800
字符时，同样的 N 保留的正文量会同比例下降。修复前 N=50 配巨型块 = 每本约 22 万字符，
修复后 N=50 只剩约 1.6 万字符（仅前言与第一章开头），教材语料整体从 3,955,185 降到
296,002 字符——这是内容丢失，不是切分收益。650 使教材语料回到 4,092,853 字符
（原体量的 1.03 倍），因此「切分」不再附带削减知识库。
"""

# 句末标点后接空白：英文教材用 .!?，CJK 语料用 。！？。
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def _hard_split(text: str, max_chars: int) -> list[str]:
    """按定长切分：句子本身超过上限时的最后手段。"""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def split_long_block(text: str, max_chars: int) -> list[str]:
    """把超过 ``max_chars`` 的块切成若干不超过上限的片段。

    优先在句边界切分并贪心合并相邻句子，使片段尽量接近上限而不越界；单个句子
    本身超过上限时对该句做定长硬切。
    """
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_BOUNDARY.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            # 该句自身越界：先冲刷缓冲区，再对它硬切。
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_hard_split(sentence, max_chars))
            continue
        candidate = f"{buffer} {sentence}" if buffer else sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return [piece.strip() for piece in pieces if piece.strip()]


def split_textbook_paragraphs(
    content: str,
    min_chars: int = 50,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """按空行分割教材段落，对超长块继续切分，过滤过短段落。

    Args:
        content: 教材全文。
        min_chars: 片段最小字符数，短于此值的片段被丢弃（原有行为）。
        max_chars: 片段最大字符数。超过上限的块按句边界切分，仍越界的残片硬切。

    Returns:
        全部片段，长度均在 ``[min_chars, max_chars]`` 内。
    """
    paragraphs: list[str] = []
    for block in content.split("\n\n"):
        text = block.strip()
        if len(text) <= max_chars:
            if len(text) >= min_chars:
                paragraphs.append(text)
            continue
        for piece in split_long_block(text, max_chars):
            if len(piece) >= min_chars:
                paragraphs.append(piece)
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
    "--chunks-per-book", type=int, default=CHUNKS_PER_BOOK,
    show_default=True,
    help="每本教材取前 N 段（控制知识库规模）。",
)
@click.option(
    "--max-chunk-chars", type=int, default=MAX_CHUNK_CHARS,
    show_default=True,
    help="单个 chunk 的硬字符上限；超长块按句边界切分。",
)
def cli(
    textbook_dir: str,
    pubmedqa_chunks: str,
    pubmedqa_eval: str,
    medqa_eval: str,
    kb_output: str,
    eval_output: str,
    chunks_per_book: int,
    max_chunk_chars: int,
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
            paragraphs = split_textbook_paragraphs(
                content, max_chars=max_chunk_chars
            )
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
