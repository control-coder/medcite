"""端到端失败归因：把每条没达到理想结果的查询归到六类原因之一，按条件统计。

理想结果：可回答的问题，作答并选中正确片段；不可回答的问题，弃答。其余都算失败，**全部**打标签，不挑样本。

六类标签互斥，按流水线先后取第一个出问题的环节（规则由程序判定，见 ``classify``）：

========== ============================================================================
检索缺失   可回答，但交给生成阶段的 3 个片段里缺少至少一个必需的正确片段，且模型没有拿噪声作答
           （弃答或被拦截）。模型这一步的行为没有问题，问题在检索。
检索噪声   可回答，模型只引用了正确片段以外的片段作答（含正确片段缺失时拿相近片段硬答）。
充分性误判 不可回答的问题，模型却作答（即误答）。
生成违规   输出被校验拦截且重试后仍被拦截，经人工复核确认模型输出确实违反约束。
校验误判   输出被校验拦截且重试后仍被拦截，但人工复核确认模型输出忠实于原文，是校验规则过严。
弃答误判   可回答、所需正确片段都在 3 个片段里，模型却弃答。
========== ============================================================================

"生成违规"与"校验误判"无法由程序区分，要读模型输出，所以放在 ``REVIEWED`` 里，每条写明理由与复核人。
本次复核只有一位复核人（项目作者），没有第二人独立标注，也没有计算标注者一致性。

用法::

    python -m eval.failure_attribution --report artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json \\
        --out artifacts/reports/llm/failure-attribution-v3-20261011.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.llm_pipeline_eval import answer_f1, generation_metrics

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "failure-attribution-v1"
LABELS = ("检索缺失", "检索噪声", "充分性误判", "生成违规", "校验误判", "弃答误判")
BLOCKED = {"invalid", "error"}

# 人工复核：(条件, 样本 ID) -> (标签, 理由)。只收程序无法判定的“被拦截”类失败。
REVIEWED: dict[tuple[str, str], tuple[str, str]] = {
    ("bm25", "pv2_D_11"): (
        "校验误判",
        "第二条引用 kb_oral_04 的摘录是该片段后半句的原文，内容忠实、引用正确；被拒是因为规则要求摘录等于整段原文。"
        "带原因重试一次后模型仍截取半句。",
    ),
}


def final_outcome(first: dict[str, Any], retry_key: str | None) -> dict[str, Any]:
    """首次被契约拦截的请求，以重试那一次的结果为准（与评测的统计口径一致）。"""
    return first[retry_key] if retry_key and retry_key in first else first


def classify(
    gold: Sequence[str], retrieved: Sequence[str], outcome: dict[str, Any], review: tuple[str, str] | None
) -> tuple[str, str] | None:
    """返回 (标签, 说明)；理想结果返回 None。"""
    decision, selected = outcome["decision"], set(outcome.get("selected", []))
    gold_set, pool = set(gold), set(retrieved)
    if not gold_set:
        if decision == "abstained":
            return None
        if decision == "answered":
            return "充分性误判", "不可回答的问题被作答"
        if review:
            return review
        raise ValueError("不可回答的问题被拦截，需要人工复核后写入 REVIEWED")
    if decision == "answered" and selected & gold_set:
        return None
    missing = sorted(gold_set - pool)
    if decision == "answered":
        return "检索噪声", f"只引用了非正确片段 {sorted(selected)}；缺少的正确片段 {missing or '无'}"
    if decision == "abstained":
        if missing:
            return "检索缺失", f"3 个片段里缺少 {missing}；模型弃答"
        return "弃答误判", "必需的正确片段都在 3 个片段里，模型仍弃答"
    if missing:
        return "检索缺失", f"3 个片段里缺少 {missing}；输出另被拦截"
    if review:
        return review
    raise ValueError("输出被拦截且正确片段齐全，需要人工复核后写入 REVIEWED")


def attribute(report: dict[str, Any], retry_key: str | None = "feedback_retry") -> dict[str, Any]:
    rows = report["per_query"]
    conditions = report["config"]["conditions"]
    failures: list[dict[str, Any]] = []
    for condition in conditions:
        for row in rows:
            cell = row["conditions"][condition]
            for trial, first in enumerate(cell["trials"]):
                label = classify(row["gold"], cell["retrieved"], final_outcome(first, retry_key),
                                 REVIEWED.get((condition, row["sample_id"])))
                if label:
                    failures.append({
                        "condition": condition, "trial": trial, "sample_id": row["sample_id"],
                        "bucket": row["bucket"], "label": label[0], "note": label[1],
                        "gold": row["gold"], "retrieved": cell["retrieved"],
                        "decision": final_outcome(first, retry_key)["decision"],
                    })

    # 条件 × 类别：每个条件取第 0 次采样（BM25 另有 2 次重复采样，单列）
    table = {c: {label: 0 for label in LABELS} for c in conditions}
    for f in failures:
        if f["trial"] == 0:
            table[f["condition"]][f["label"]] += 1
    repeats = defaultdict(lambda: {label: 0 for label in LABELS})
    for f in failures:
        if f["trial"] > 0:
            repeats[f"{f['condition']}/trial_{f['trial']}"][f["label"]] += 1

    first_trial = [f for f in failures if f["trial"] == 0]
    # 检索缺失里，有多少是“多片段问题只缺一部分”（片段里至少有一个正确片段）
    partial_missing = {c: sum(1 for f in first_trial if f["condition"] == c and f["label"] == "检索缺失"
                              and set(f["gold"]) & set(f["retrieved"])) for c in conditions}
    by_query: dict[str, set[str]] = defaultdict(set)
    for f in first_trial:
        by_query[f["sample_id"]].add(f["label"])
    distinct_by_label = Counter(label for labels in by_query.values() for label in labels)
    answerable = sum(1 for r in rows if r["gold"])
    return {
        "retry_key": retry_key,
        "labels": list(LABELS),
        "answerable_n": answerable, "unanswerable_n": len(rows) - answerable,
        "failures_by_condition_and_label": table,
        "failure_totals_by_condition": {c: sum(v.values()) for c, v in table.items()},
        "retrieval_missing_partial_by_condition": partial_missing,
        "repeated_trials": dict(repeats),
        "distinct_failing_queries": len(by_query),
        "distinct_queries_by_label": {label: distinct_by_label.get(label, 0) for label in LABELS},
        "queries_failing_in_all_conditions": sorted(
            sid for sid in by_query if {f["condition"] for f in first_trial if f["sample_id"] == sid} == set(conditions)),
        "reviewed": [{"condition": c, "sample_id": s, "label": v[0], "reason": v[1]} for (c, s), v in REVIEWED.items()],
        "failures": failures,
    }


def headline(report: dict[str, Any], retry_key: str | None = "feedback_retry") -> dict[str, Any]:
    """用与评测相同的函数重算各条件的头条指标，供文档引用时核对。"""
    out = {}
    for condition in report["config"]["conditions"]:
        metrics = generation_metrics(report["per_query"], condition, 0, with_retry=retry_key or False)
        f1 = answer_f1(metrics)
        out[condition] = {
            "useful": f"{metrics['useful_answer_rate']['k']}/{metrics['useful_answer_rate']['n']}",
            "false_answer": f"{metrics['false_answer_rate']['k']}/{metrics['false_answer_rate']['n']}",
            "over_abstention": f"{metrics['over_abstention_rate']['k']}/{metrics['over_abstention_rate']['n']}",
            "f1": f1["f1"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = attribute(report)
    payload = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
               "source_report": args.report.name, "external_model_calls": 0,
               "annotators": "1 人（项目作者），未做第二人独立标注", "headline": headline(report), **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("头条指标:", json.dumps(payload["headline"], ensure_ascii=False))
    for condition, counts in result["failures_by_condition_and_label"].items():
        print(f"{condition:12s}", {k: v for k, v in counts.items() if v}, "合计", result["failure_totals_by_condition"][condition])
    print("重复采样:", dict(result["repeated_trials"]))
    print("去重失败查询数:", result["distinct_failing_queries"], "各类:", result["distinct_queries_by_label"])
    print("报告：", args.out)


if __name__ == "__main__":
    main()
