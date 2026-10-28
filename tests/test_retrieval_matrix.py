"""检索层对照矩阵的统计与装配测试；向量检索用确定性假编码器，不下载模型。"""

from __future__ import annotations

import pytest

from eval import retrieval_benchmark as rb
from eval import retrieval_matrix as rm
from tests.test_retrieval_benchmark import CHUNKS, fake_encoder

QUERIES = [
    {"sample_id": "t1", "split": "test", "bucket": "direct", "question": "咳嗽时怎样遮住口鼻？", "gold_evidence_ids": ["kb_a_01"]},
    {"sample_id": "t2", "split": "test", "bucket": "direct", "question": "孩子能留在车辆里吗", "gold_evidence_ids": ["kb_b_01"]},
    {"sample_id": "t3", "split": "test", "bucket": "near_miss", "question": "咳嗽药一天吃几次", "gold_evidence_ids": []},
    {"sample_id": "d1", "split": "dev", "bucket": "direct", "question": "糖尿病会导致什么", "gold_evidence_ids": ["kb_c_01"]},
]


def _pool_row(sample_id: str, gold: list[str], retrieved: list[str]) -> dict:
    return {"sample_id": sample_id, "gold": gold,
            "conditions": {name: {"retrieved": retrieved} for name in
                           ("bm25", "dense", "bm25_rewrite", "bm25_agent", "dense_agent")}}


def test_query_metrics_use_rank_of_each_gold() -> None:
    first = rm.query_metrics([1], 1)
    assert first["hit@1"] == 1.0 and first["mrr@3"] == 1.0 and first["ndcg@10"] == pytest.approx(1.0)
    late = rm.query_metrics([5], 1)
    assert late["hit@3"] == 0.0 and late["mrr@3"] == 0.0 and late["mrr@10"] == pytest.approx(0.2)
    two = rm.query_metrics([1, 7], 2)  # 两个正确片段只找回一个
    assert two["recall@3"] == 0.5 and two["hit@3"] == 1.0


def test_sign_test_is_exact_and_two_sided() -> None:
    assert rm.sign_test_p(0, 0) == 1.0
    assert rm.sign_test_p(9, 1) == pytest.approx(0.0215, abs=1e-4)  # 与评测报告里改写/检索 Agent 的检验同一算法
    assert rm.sign_test_p(5, 5) == 1.0


def test_compare_reports_wins_losses_and_ties() -> None:
    result = rm.compare([1, 1, 0, 0], [1, 0, 0, 1])
    assert (result["a_better"], result["b_better"], result["tied"]) == (1, 1, 2)
    assert result["diff"] == 0.0 and result["sign_test_p"] == 1.0
    assert result["diff_ci95"][0] <= 0.0 <= result["diff_ci95"][1]


def test_build_matrix_aligns_systems_and_checks_consistency() -> None:
    dataset = rb.Dataset(chunks=CHUNKS, queries=QUERIES)
    rows = [_pool_row("t1", ["kb_a_01"], ["kb_a_01", "kb_b_01", "kb_c_01"]),
            _pool_row("t2", ["kb_b_01"], ["kb_a_01", "kb_b_01", "kb_c_01"]),
            _pool_row("t3", [], ["kb_a_01"])]
    result = rm.build_matrix(dataset, {"per_query": rows}, fake_encoder, alpha=0.2)

    assert result["answerable_n"] == 2  # 无答案的查询和 dev 查询都不进矩阵
    assert set(result["matrix"]) == {"bm25_bigram", "dense", "hybrid", "bm25_rewrite", "bm25_agent", "dense_agent"}
    assert set(result["matrix"]["bm25_agent"]) >= {"hit@3", "recall@3", "mrr@3"}
    assert "ndcg@10" not in result["matrix"]["bm25_agent"]  # 只有前 3 条的系统不报 nDCG@10
    assert result["matrix"]["bm25_agent"]["mrr@3"]["mean"] == pytest.approx(0.75)  # t1 第 1 名、t2 第 2 名
    assert {c["metric"] for c in result["paired_comparisons"] if c["a"] == "hybrid"} == {"hit@3", "mrr@10", "ndcg@10"}
    assert result["consistency_with_llm_report"]["queries"] == 2


def test_build_matrix_rejects_mismatched_query_sets() -> None:
    dataset = rb.Dataset(chunks=CHUNKS, queries=QUERIES)
    rows = [_pool_row("t1", ["kb_a_01"], ["kb_a_01"])]  # 缺 t2
    with pytest.raises(ValueError, match="不一致"):
        rm.build_matrix(dataset, {"per_query": rows}, fake_encoder, alpha=0.2)
