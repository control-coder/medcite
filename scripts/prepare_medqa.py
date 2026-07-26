"""改造 MedQA US 数据集为评测集。

输入: eval/datasets/raw/medqa_data/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl
输出: eval/datasets/eval_set_medqa.jsonl

注意: MedQA 没有 explanation 字段，无法直接派生 gold_evidence_ids。
      本脚本将 gold_evidence_ids 置空，label_source 标为 "dataset_no_evidence"。
      后续可通过 textbook 检索派生或人工标注补充。

数据泄露防护:
  - sample_id = "medqa_us_{seq:05d}"（序号，非原始标识）
  - MedQA 的 textbooks 作为独立知识库来源，不引用 sample_id
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import click

from medidiag.schemas import EvalSample, sample_to_dict, write_jsonl


@click.command()
@click.option(
    "--input", "input_path",
    type=click.Path(exists=False),
    default="eval/datasets/raw/medqa_data/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl",
    show_default=True,
)
@click.option(
    "--output", "output_path",
    default="eval/datasets/eval_set_medqa.jsonl",
    show_default=True,
)
@click.option(
    "--limit", type=int, default=None,
    help="限制样本数（调试用）。",
)
def cli(input_path: str, output_path: str, limit: int | None) -> None:
    """改造 MedQA US 数据集。"""

    samples: list[EvalSample] = []
    with open(input_path, encoding="utf-8") as f:
        for seq, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit and seq >= limit:
                break

            item = json.loads(line)
            sample_id = f"medqa_us_{seq:05d}"

            sample = EvalSample(
                sample_id=sample_id,
                source="MedQA",
                question=item["question"],
                gold_answer=item["answer"],
                gold_evidence_ids=[],  # MedQA 无 explanation
                label_source="dataset_no_evidence",
                labeler="dataset",
                review_status="single_checked",
                options=item.get("options"),
                metadata={
                    "meta_info": item.get("meta_info", ""),
                    "answer_idx": item.get("answer_idx", ""),
                },
            )
            samples.append(sample)

    n = write_jsonl(
        [sample_to_dict(s) for s in samples], output_path
    )

    click.echo("MedQA 改造完成:")
    click.echo(f"  样本数          : {n}")
    click.echo("  gold_evidence   : 空（MedQA 无 explanation，label_source=dataset_no_evidence）")
    click.echo(f"  output -> {output_path}")


if __name__ == "__main__":
    cli()
