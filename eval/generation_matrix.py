"""生成层与端到端矩阵：由一份端到端评测报告算出各条件的逐字绑定通过率、误答、拒答、作答且选对、回答 F1。

只读报告 JSON，不联网、不调用模型。统计口径与 ``eval.llm_pipeline_eval`` 完全一致（直接调用它的
``generation_metrics`` 和 ``answer_f1``）；这里额外给出 95% 区间：比例用 Wilson 区间，F1 用按查询重抽样的自助法区间。

逐字绑定检查只核对“摘录是否等于本轮给出的某段完整原文、引用是否存在且不重复”。它是确定性的忠实度检查，
不判断摘录是否真正回答了问题，更不是语义层面的忠实度。

用法::

    python -m eval.generation_matrix --report artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json --out ...
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.llm_pipeline_eval import answer_f1, generation_metrics
from eval.retrieval_benchmark import proportion

SCHEMA_VERSION = "generation-matrix-v1"
RETRY_KEY = "feedback_retry"


def _outcomes(rows: Sequence[dict[str, Any]], condition: str, trial: int, retry_key: str | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        first = row["conditions"][condition]["trials"][trial]
        final = first[retry_key] if retry_key and retry_key in first else first
        out.append({"gold": set(row["gold"]), "first": first, "final": final})
    return out


def _f1_from(outcomes: Sequence[dict[str, Any]]) -> float | None:
    correct = sum(1 for o in outcomes if o["gold"] and o["final"]["decision"] == "answered"
                  and set(o["final"]["selected"]) & o["gold"])
    answered = sum(1 for o in outcomes if o["final"]["decision"] == "answered")
    answerable = sum(1 for o in outcomes if o["gold"])
    if not answered or not answerable or not correct:
        return 0.0 if answerable else None
    precision, recall = correct / answered, correct / answerable
    return 2 * precision * recall / (precision + recall)


def bootstrap_f1_ci(outcomes: Sequence[dict[str, Any]], resamples: int = 2000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(outcomes)
    values = sorted(v for v in (_f1_from([outcomes[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
                    if v is not None)
    return [round(values[int(0.025 * len(values))], 4), round(values[int(0.975 * len(values)) - 1], 4)]


def condition_row(report: dict[str, Any], condition: str, trial: int = 0) -> dict[str, Any]:
    rows = report["per_query"]
    outcomes = _outcomes(rows, condition, trial, RETRY_KEY)
    first_pass = generation_metrics(rows, condition, trial)
    final = generation_metrics(rows, condition, trial, with_retry=RETRY_KEY)
    called = sum(1 for o in outcomes if o["first"].get("model_called"))
    first_blocked = sum(1 for o in outcomes if o["first"]["decision"] in {"invalid", "error"})
    final_blocked = sum(1 for o in outcomes if o["final"]["decision"] in {"invalid", "error"})
    details = sorted({o["first"].get("detail", "") for o in outcomes if o["first"]["decision"] in {"invalid", "error"}})
    f1 = answer_f1(final)
    return {
        "model_calls": called,
        # L2：逐字绑定通过率 = 模型被调用的请求里，输出一次通过校验的比例；重试一次后再算一遍
        "binding_pass_first": proportion(called - first_blocked, called),
        "binding_pass_after_retry": proportion(called - final_blocked, called),
        "blocked_first": first_blocked, "blocked_after_retry": final_blocked,
        "blocked_reasons": details,
        "false_answer": final["false_answer_rate"],
        "over_abstention": final["over_abstention_rate"],
        # L3
        "useful_answer": final["useful_answer_rate"],
        "useful_answer_first_pass": first_pass["useful_answer_rate"],
        "precision": f1["precision"], "recall": f1["recall"], "f1": f1["f1"],
        "f1_ci95": bootstrap_f1_ci(outcomes),
    }


def build_matrix(report: dict[str, Any]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for condition in report["config"]["conditions"]:
        trials = len(report["per_query"][0]["conditions"][condition]["trials"])
        matrix[condition] = {"trial_0": condition_row(report, condition, 0)}
        if trials > 1:  # 只有 BM25 采样多次：单独列出其余采样，说明采样波动
            matrix[condition].update({f"trial_{t}": condition_row(report, condition, t) for t in range(1, trials)})
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    matrix = build_matrix(report)
    payload = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
               "source_report": args.report.name, "external_model_calls": 0,
               "model": report["config"]["model"], "retry": "被拦截时带原因重试一次", "matrix": matrix}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for condition, trials in matrix.items():
        for trial, row in trials.items():
            print(f"{condition}/{trial}: 绑定一次 {row['binding_pass_first']['k']}/{row['binding_pass_first']['n']} "
                  f"重试后 {row['binding_pass_after_retry']['k']}/{row['binding_pass_after_retry']['n']} "
                  f"误答 {row['false_answer']['k']}/{row['false_answer']['n']} "
                  f"拒答 {row['over_abstention']['k']}/{row['over_abstention']['n']} "
                  f"选对 {row['useful_answer']['k']}/{row['useful_answer']['n']} F1 {row['f1']} {row['f1_ci95']}")
    print("报告：", args.out)


if __name__ == "__main__":
    main()
