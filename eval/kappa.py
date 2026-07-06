"""Cohen's Kappa 标注一致性计算。

PLAN.md 要求:
- 抽样 20% 做双人复核
- Kappa < 0.6: 剔除或重新标注
- 0.6 <= Kappa < 0.8: 进入分歧讨论
- Kappa >= 0.8: 视为稳定标注
- 分歧讨论必须记录最终裁决人、裁决理由和被修改字段

输入: 两份标注 JSONL，每行 {"sample_id": ..., "label": ...}
输出: Kappa 值 + 一致性判定 + 混淆矩阵

用法:
  python -m eval.kappa --annotator-a annotations_a.jsonl --annotator-b annotations_b.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click


def load_annotations(path: str | Path) -> dict[str, str]:
    """加载标注文件，返回 {sample_id: label}。"""
    p = Path(path)
    if not p.exists():
        raise click.FileError(str(p), hint="annotation file not found")
    ann: dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ann[rec["sample_id"]] = rec["label"]
    return ann


def cohen_kappa(
    ann_a: dict[str, str], ann_b: dict[str, str]
) -> tuple[float, float, float, list[str], dict[str, int], int]:
    """计算 Cohen's Kappa。

    公式: kappa = (po - pe) / (1 - pe)
      po = 观察一致率 = 一致样本数 / 总样本数
      pe = 期望一致率 = Σ(类别 i 在 A 的比例 × 类别 i 在 B 的比例)

    返回 (kappa, po, pe, labels, confusion_matrix, n_common)
    """
    common_ids = set(ann_a.keys()) & set(ann_b.keys())
    if not common_ids:
        raise ValueError("no common sample_ids between annotators")

    common_ids = sorted(common_ids)
    labels = sorted(set(ann_a.values()) | set(ann_b.values()))
    n = len(common_ids)

    # 观察一致率 po
    agree = sum(1 for sid in common_ids if ann_a[sid] == ann_b[sid])
    po = agree / n

    # 期望一致率 pe
    counter_a = Counter(ann_a[sid] for sid in common_ids)
    counter_b = Counter(ann_b[sid] for sid in common_ids)
    pe = sum((counter_a[l] / n) * (counter_b[l] / n) for l in labels)

    # Kappa
    if pe >= 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    # 混淆矩阵
    confusion: dict[str, int] = {}
    for l1 in labels:
        for l2 in labels:
            count = sum(
                1 for sid in common_ids
                if ann_a[sid] == l1 and ann_b[sid] == l2
            )
            if count > 0:
                confusion[f"{l1}|{l2}"] = count

    return kappa, po, pe, labels, confusion, n


def kappa_verdict(kappa: float) -> str:
    """根据 PLAN.md 阈值判定一致性等级。"""
    if kappa < 0.6:
        return "BELOW_THRESHOLD: 低于 0.6，样本需剔除或重新标注"
    elif kappa < 0.8:
        return "DISCUSS: 0.6-0.8，进入分歧讨论（记录裁决人、理由、修改字段）"
    else:
        return "STABLE: 高于 0.8，视为稳定标注"


@click.command()
@click.option(
    "--annotator-a", required=True,
    type=click.Path(exists=False),
    help="标注者 A 的 JSONL 文件（每行 {sample_id, label}）。",
)
@click.option(
    "--annotator-b", required=True,
    type=click.Path(exists=False),
    help="标注者 B 的 JSONL 文件（每行 {sample_id, label}）。",
)
def cli(annotator_a: str, annotator_b: str) -> None:
    """计算 Cohen's Kappa 标注一致性。"""

    ann_a = load_annotations(annotator_a)
    ann_b = load_annotations(annotator_b)

    kappa, po, pe, labels, confusion, n = cohen_kappa(ann_a, ann_b)

    click.echo("========== Cohen's Kappa ==========")
    click.echo(f"共同样本数    : {n}")
    click.echo(f"标注类别      : {labels}")
    click.echo(f"观察一致率 po : {po:.4f}")
    click.echo(f"期望一致率 pe : {pe:.4f}")
    click.echo(f"Cohen's Kappa : {kappa:.4f}")
    click.echo(f"判定          : {kappa_verdict(kappa)}")
    click.echo("")
    click.echo("混淆矩阵 (annotator_a | annotator_b):")
    for key, cnt in sorted(confusion.items()):
        click.echo(f"  {key:40s}: {cnt}")


if __name__ == "__main__":
    cli()
