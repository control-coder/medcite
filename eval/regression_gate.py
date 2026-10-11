"""回归门槛：用一份端到端评测报告对照阈值文件，任何一条退化就返回非零退出码。

阈值取自当前入库报告的观测值，含义是“不得比现在更差”，不是质量证明：
误答 0/28 的 95% 区间上限仍有约 12%，门槛只能发现退化，不能证明安全。

零网络：只读报告 JSON 和阈值 JSON。报告本身是否可信，由 ``tests/test_llm_replay_*.py`` 用仓库内的
模型调用记录回放后逐项比对来保证（BM25 三个条件；向量条件依赖本地嵌入模型，CI 里不回放）。

用法::

    python -m eval.regression_gate
    python -m eval.regression_gate --report artifacts/reports/llm/某份新报告.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = ROOT / "examples/rag_regression/thresholds.json"
# 两条统计路径：不重试的一次结果，和“被拦截后带原因重试一次”的最终结果
PATHS = {"generation": "一次结果", "generation_with_feedback_retry": "带原因重试后"}


def check_report(report: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """返回违反的规则（空列表表示通过）。"""
    rules = thresholds["rules"]
    violations: list[str] = []
    for path, path_name in PATHS.items():
        for condition, trials in report.get(path, {}).items():
            for trial, metrics in trials.items():
                where = f"{condition}/{trial}/{path_name}"
                false_answers = metrics["false_answer_rate"]["k"]
                if false_answers > rules["max_false_answers"]:
                    violations.append(f"{where}：误答 {false_answers} 条，上限 {rules['max_false_answers']}")
                blocked = metrics["contract_violations_or_errors"]
                limit = (rules["max_first_pass_violations_per_trial"] if path == "generation"
                         else rules["max_unrecovered_violations_per_trial"])
                if blocked > limit:
                    violations.append(f"{where}：被拦截 {blocked} 条，上限 {limit}")
                if path == "generation_with_feedback_retry":
                    floor = rules["min_useful_answers"].get(condition)
                    useful = metrics["useful_answer_rate"]["k"]
                    if floor is None:
                        violations.append(f"{where}：阈值文件没有这个条件")
                    elif useful < floor:
                        violations.append(f"{where}：有用回答 {useful} 条，下限 {floor}")
    missing = set(rules["min_useful_answers"]) - set(report.get("generation_with_feedback_retry", {}))
    if missing:
        violations.append(f"报告缺少条件：{sorted(missing)}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    parser.add_argument("--report", type=Path, default=None, help="默认用阈值文件里登记的入库报告")
    args = parser.parse_args()
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    report_path = args.report or ROOT / thresholds["report"]
    violations = check_report(json.loads(report_path.read_text(encoding="utf-8")), thresholds)
    for line in violations:
        print("退化：", line)
    print(f"回归门槛{'未通过' if violations else '通过'}：{report_path.name}，共 {len(violations)} 条退化")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
