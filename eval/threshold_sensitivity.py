"""测量 `MIN_PRIMARY_SCORE` 取不同值时的路由分支分布，为阈值决策提供证据。

这是**反事实重算工具**，不产生任何评测指标，也不调用 provider、不加载模型、
不访问网络：它只读 `eval/routing_diagnostics.py` 已写出的产物（每个样本的 top1 /
top2 分项得分是实测的检索 + 打分结果），然后用不同阈值重新归类同一批分数。

为什么可以离线重算：删改阈值不改变任何分数（DD-025 已验证过这一点——删除规则 4
后规则 1 的计数一字未变）。因此「阈值取 X 会怎样」完全可以由已有产物算出，不需要
重跑检索。这也使对照口径天然只有阈值一个变量。

**这个工具不能回答的问题**：被放行的样本走对了哪个专科。那需要人工标注，本项目
没有。它只能回答「有多少样本被放行」以及「这些样本的信号有多强」。

用法::

    python -m eval.threshold_sensitivity
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from medidiag.agents.specialty_data import (
    MATCH_SATURATION_COUNT,
    ROUTING_WEIGHTS,
    THRESHOLDS,
)

ROOT = Path(__file__).resolve().parent.parent

# 与 eval/routing_diagnostics.py 的 `_classify` 同一组规则，只是 MIN_PRIMARY_SCORE
# 变成参数。其余两个阈值取生产值，本工具不扫它们。
CANDIDATE_THRESHOLDS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0)

# 每命中一个关键词对总分的贡献：权重 3.0 / 饱和数 4.0。
KEYWORD_HIT_VALUE = ROUTING_WEIGHTS["keyword"] / MATCH_SATURATION_COUNT


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


class ThresholdSensitivityError(Exception):
    """输入产物不是规则 4 删除后的路由诊断产物。"""


def classify(top1_total: float, top2_total: float, min_primary: float) -> str:
    """复现 `SpecialistRouter.route` 的三条规则，`MIN_PRIMARY_SCORE` 为参数。

    与 `eval/routing_diagnostics.py::_classify` 必须保持一致；差异会使本工具的
    反事实与实测产物不可比。另外两个阈值取生产值。
    """
    if top1_total < min_primary:
        return "rule_1_primary_below_threshold"
    if (
        top2_total < THRESHOLDS["MIN_SECONDARY_SCORE"]
        or (top1_total - top2_total) >= THRESHOLDS["SCORE_GAP"]
    ):
        return "rule_2_top1_plus_skeptic"
    return "rule_3_dynamic_top2"


def _keyword_hits(keyword_score: float) -> int:
    """由归一化后的关键词分反推命中词条数（分数 = 命中数 / 饱和数，已封顶）。"""
    return int(round(float(keyword_score) * MATCH_SATURATION_COUNT))


def _signal_profile(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """描述一批样本的信号强度，用于判断放行它们是否有依据。

    只看 top1 的分项构成，不判断专科对不对——后者需要人工标注。
    """
    if not samples:
        return {
            "count": 0,
            "keyword_hits_ge_1": 0,
            "keyword_hits_ge_2": 0,
            "normalized_term_hit": 0,
            "evidence_only_no_query_signal": 0,
        }
    return {
        "count": len(samples),
        "keyword_hits_ge_1": sum(
            1 for s in samples if _keyword_hits(s["top1"]["keyword_score"]) >= 1
        ),
        "keyword_hits_ge_2": sum(
            1 for s in samples if _keyword_hits(s["top1"]["keyword_score"]) >= 2
        ),
        "normalized_term_hit": sum(
            1 for s in samples if s["top1"]["normalized_term_score"] > 0
        ),
        # 问题正文完全没有专科信号，分数只来自检索回来的 chunk。
        "evidence_only_no_query_signal": sum(
            1
            for s in samples
            if s["top1"]["keyword_score"] == 0
            and s["top1"]["normalized_term_score"] == 0
        ),
        "top1_specialty_counts": dict(
            Counter(s["top1"]["specialty"] for s in samples).most_common(6)
        ),
    }


def run_sensitivity(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """对一份路由诊断产物逐档重算分支分布。"""
    samples = diagnostics.get("samples")
    if not samples:
        raise ThresholdSensitivityError(
            "routing diagnostics artifact has no 'samples'; regenerate with "
            "`python -m eval.routing_diagnostics`"
        )
    if "rule_4_low_confidence" in diagnostics.get("branch_counts", {}):
        # 规则 4 删除前的产物按四条规则归类，反事实不可比（DD-025）。
        raise ThresholdSensitivityError(
            "artifact predates the rule-4 removal (DD-025); use a "
            "*_rule4_removed.json artifact instead"
        )

    total = len(samples)
    pairs = [
        (s["top1"]["total"], s["top2"]["total"] if s["top2"] else 0.0)
        for s in samples
    ]

    by_threshold: dict[str, Any] = {}
    for threshold in CANDIDATE_THRESHOLDS:
        counts = Counter(
            classify(t1, t2, threshold) for t1, t2 in pairs
        )
        fallback = counts["rule_1_primary_below_threshold"]
        by_threshold[f"min_primary_{threshold}"] = {
            "min_primary_score": threshold,
            "is_current_production_value": threshold
            == THRESHOLDS["MIN_PRIMARY_SCORE"],
            "rule_1_primary_below_threshold": fallback,
            "rule_2_top1_plus_skeptic": counts["rule_2_top1_plus_skeptic"],
            "rule_3_dynamic_top2": counts["rule_3_dynamic_top2"],
            "fallback_count": fallback,
            "fallback_rate": round(fallback / total, 4) if total else 0.0,
        }

    # 关键词命中数的分布，以及「恰好 N 次命中」的样本有多少能过当前阈值。
    # 这是本工具最有解释力的一段：它把抽象的 2.0 翻译成「几个关键词」。
    hit_counts = Counter(_keyword_hits(s["top1"]["keyword_score"]) for s in samples)
    current = THRESHOLDS["MIN_PRIMARY_SCORE"]
    passes_by_hits = {}
    for hits in sorted(hit_counts):
        group = [
            s for s in samples if _keyword_hits(s["top1"]["keyword_score"]) == hits
        ]
        passes_by_hits[str(hits)] = {
            "sample_count": len(group),
            "passes_current_threshold": sum(
                1 for s in group if s["top1"]["total"] >= current
            ),
        }

    # 固定分段边界，使不同代产物可逐段对照；「现在放行」一段跟随生产阈值而不是
    # 写死 2.0，否则改阈值后这个标签会静默变成谎话。
    bands = {
        "[0.0,1.0)": [s for s in samples if s["top1"]["total"] < 1.0],
        "[1.0,1.5)": [s for s in samples if 1.0 <= s["top1"]["total"] < 1.5],
        "[1.5,2.0)": [s for s in samples if 1.5 <= s["top1"]["total"] < 2.0],
        "[2.0,inf)": [s for s in samples if s["top1"]["total"] >= 2.0],
        f">={current}_released_today": [
            s for s in samples if s["top1"]["total"] >= current
        ],
    }

    return {
        "sample_count": total,
        "sample_set": diagnostics.get("sample_set", ""),
        "source_thresholds": dict(THRESHOLDS),
        "keyword_hit_value": round(KEYWORD_HIT_VALUE, 4),
        "match_saturation_count": MATCH_SATURATION_COUNT,
        "by_threshold": by_threshold,
        "top1_keyword_hit_counts": {
            str(k): v for k, v in sorted(hit_counts.items())
        },
        "passes_current_threshold_by_keyword_hits": passes_by_hits,
        "signal_profile_by_score_band": {
            label: _signal_profile(group) for label, group in bands.items()
        },
    }


@click.command()
@click.option(
    "--manifest-diagnostics",
    default="artifacts/reports/routing_diagnostics_manifest_rule4_removed.json",
    show_default=True,
)
@click.option(
    "--holdout-diagnostics",
    default="artifacts/reports/routing_diagnostics_holdout_rule4_removed.json",
    show_default=True,
)
@click.option("--output", default="artifacts/reports/threshold_sensitivity.json", show_default=True)
def cli(
    manifest_diagnostics: str, holdout_diagnostics: str, output: str
) -> None:
    report: dict[str, Any] = {
        "note": (
            "Counterfactual re-classification of already-measured routing scores. "
            "Changing a threshold changes no score, so this needs no re-retrieval. "
            "It answers how many samples get released, NOT whether they are routed "
            "to the right specialty -- that requires human labels this project "
            "does not have."
        ),
        "sets": {},
    }
    for label, path_value in (
        ("manifest", manifest_diagnostics),
        ("holdout", holdout_diagnostics),
    ):
        path = _resolve(path_value)
        if not path.exists():
            raise click.UsageError(
                f"{path} not found; run `python -m eval.routing_diagnostics` first"
            )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        try:
            report["sets"][label] = run_sensitivity(loaded)
        except ThresholdSensitivityError as exc:
            raise click.UsageError(f"{path}: {exc}") from exc
        report["sets"][label]["source_artifact"] = path_value

    output_path = _resolve(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for label, data in report["sets"].items():
        click.echo(f"=== {label} (n={data['sample_count']}) ===")
        click.echo(f"  {'MIN_PRIMARY':>11} {'fallback':>9} {'rate':>6} {'rule2':>6} {'rule3':>6}")
        for entry in data["by_threshold"].values():
            marker = " <- current" if entry["is_current_production_value"] else ""
            click.echo(
                f"  {entry['min_primary_score']:>11} "
                f"{entry['fallback_count']:>9} "
                f"{entry['fallback_rate']:>6.0%} "
                f"{entry['rule_2_top1_plus_skeptic']:>6} "
                f"{entry['rule_3_dynamic_top2']:>6}{marker}"
            )
        click.echo(
            f"  one keyword hit is worth {data['keyword_hit_value']} total points"
        )
        for hits, stats in data["passes_current_threshold_by_keyword_hits"].items():
            click.echo(
                f"    {hits} keyword hit(s): {stats['sample_count']:>3} samples, "
                f"{stats['passes_current_threshold']:>3} pass current threshold"
            )
    click.echo(f"output={output_path}")


if __name__ == "__main__":
    cli()
