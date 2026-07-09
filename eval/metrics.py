"""评测指标计算。

每个指标包含: 公式 / ground truth 来源 / 计算函数。
所有指标可被 eval/runner.py 和 eval/report.py 调用。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_recall_at_k(hit_results: Sequence[bool]) -> float:
    """Evidence Recall@k。

    公式: 至少命中 1 条 gold_evidence 的样本数 / 总样本数
    ground truth: eval_set.gold_evidence_ids
    """
    if not hit_results:
        return 0.0
    return sum(1 for h in hit_results if h) / len(hit_results)


def compute_citation_precision(citation_verdicts: Sequence[str]) -> float:
    """Citation Precision。

    公式: SUPPORTED citation 数 / 系统输出 citation 总数
    ground truth: judge_model (deberta-v3-base-mnli) + 人工抽样
    """
    if not citation_verdicts:
        return 0.0
    supported = sum(1 for v in citation_verdicts if v == "SUPPORTED")
    return supported / len(citation_verdicts)


def compute_unsupported_claim_rate(
    citation_verdicts: Sequence[str],
) -> float:
    """Unsupported Claim Rate。

    公式: UNSUPPORTED claims / total claims
    ground truth: judge_model
    """
    if not citation_verdicts:
        return 0.0
    unsupported = sum(1 for v in citation_verdicts if v == "UNSUPPORTED")
    return unsupported / len(citation_verdicts)


def compute_workflow_success_rate(
    success_results: Sequence[bool],
) -> float:
    """Workflow Success Rate。

    公式: CLOSED_SUCCESS case 数 / 总 case 数
    ground truth: case_event_log 终态统计
    """
    if not success_results:
        return 0.0
    return sum(1 for s in success_results if s) / len(success_results)


def compute_terminology_normalization_gain(
    recall_with_norm: float,
    recall_without_norm: float,
) -> float:
    """Terminology Normalization Gain。

    公式: Recall@5(with normalization) - Recall@5(without normalization)
    ground truth: ablation group A vs D
    """
    return recall_with_norm - recall_without_norm


def compute_p95_latency(latencies_ms: Sequence[float]) -> float:
    """P95 latency。

    公式: 端到端 P95 延迟
    ground truth: 分阶段 latency 埋点
    """
    if not latencies_ms:
        return 0.0
    return float(np.percentile(list(latencies_ms), 95))


def compute_judge_agreement(
    judge_results: Sequence[str],
    human_results: Sequence[str],
) -> float:
    """Judge Agreement。

    公式: judge 判定与人工抽样复核一致样本数 / 抽样复核样本数
    ground truth: 人工双标注 + Cohen's Kappa
    """
    if not judge_results or len(judge_results) != len(human_results):
        return 0.0
    agree = sum(
        1 for j, h in zip(judge_results, human_results) if j == h
    )
    return agree / len(judge_results)


def compute_routing_coverage(
    routed_count: int,
    total_count: int,
) -> float:
    """路由覆盖率（动态路由非兜底比例）。

    公式: 非兜底路由 case 数 / 总 case 数
    """
    if total_count == 0:
        return 0.0
    return routed_count / total_count


def compute_specialty_relevance(
    relevant_count: int,
    total_count: int,
) -> float:
    """专科相关率。

    公式: 路由到相关专科的 case 数 / 总 case 数
    """
    if total_count == 0:
        return 0.0
    return relevant_count / total_count


def compute_multi_agent_gain(
    multi_agent_metric: float,
    single_agent_metric: float,
) -> float:
    """多 Agent 相对单 Agent 增益。

    公式: multi_agent_metric - single_agent_metric
    """
    return multi_agent_metric - single_agent_metric


def compute_fixed_pair_noise_rate(
    irrelevant_count: int,
    total_count: int,
) -> float:
    """固定配对噪声率（无关专科占比）。

    公式: 固定配对中无关专科的 case 数 / 总 case 数
    """
    if total_count == 0:
        return 0.0
    return irrelevant_count / total_count
