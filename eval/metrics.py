"""评测指标计算。

每个指标包含: 公式 / ground truth 来源 / 计算函数。
所有指标可被 eval/runner.py 和 eval/report.py 调用。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def compute_recall_at_k(hit_results: Sequence[bool]) -> float:
    """Evidence Recall@k。

    公式: 至少命中 1 条 gold_evidence 的 eligible 样本数 / eligible 样本数
    ground truth: eval_set.gold_evidence_ids

    调用方必须只传入 ``gold_evidence_ids`` 非空的样本。
    """
    if not hit_results:
        return 0.0
    return sum(1 for h in hit_results if h) / len(hit_results)


def compute_gold_evidence_coverage(eligible_count: int, total_count: int) -> float:
    """Gold Evidence Coverage = eligible samples / all samples."""
    if total_count == 0:
        return 0.0
    return eligible_count / total_count


def compute_citation_precision(citation_results: Sequence[Any]) -> float:
    """Citation Precision。

    公式: SUPPORTED claim-citation pairs / emitted claim-citation pairs
    ground truth: judge_model (deberta-v3-base-mnli) + 人工抽样
    """
    emitted_pairs = [
        result for result in citation_results if _citation_chunk_id(result)
    ]
    if not emitted_pairs:
        return 0.0
    supported = sum(1 for result in emitted_pairs if _verdict(result) == "SUPPORTED")
    return supported / len(emitted_pairs)


def compute_unsupported_claim_rate(citation_results: Sequence[Any]) -> float:
    """Unsupported Claim Rate。

    公式: UNSUPPORTED claims / total claims
    ground truth: judge_model
    """
    if not citation_results:
        return 0.0
    best_by_claim: dict[str, int] = {}
    rank = {"UNSUPPORTED": 0, "PARTIAL": 1, "SUPPORTED": 2}
    for index, result in enumerate(citation_results):
        claim_id = _claim_id(result) or f"legacy_claim_{index}"
        best_by_claim[claim_id] = max(
            best_by_claim.get(claim_id, -1), rank.get(_verdict(result), 0)
        )
    unsupported = sum(1 for best_rank in best_by_claim.values() if best_rank == 0)
    return unsupported / len(best_by_claim)


def _value(result: Any, field: str, default: str = "") -> str:
    if isinstance(result, dict):
        value = result.get(field, default)
    else:
        value = getattr(result, field, default)
    return value.value if hasattr(value, "value") else str(value)


def _verdict(result: Any) -> str:
    return _value(result, "verdict")


def _claim_id(result: Any) -> str:
    return _value(result, "claim_id")


def _citation_chunk_id(result: Any) -> str:
    return _value(result, "evidence_chunk_id")


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
    ground truth: rag_embedding vs rag_term_norm
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
        1 for j, h in zip(judge_results, human_results, strict=True) if j == h
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
