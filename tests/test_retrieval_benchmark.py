"""检索/弃答基准的统计与流程测试；稠密检索用确定性假编码器，不下载模型。"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from eval import retrieval_benchmark as rb

CHUNKS = [
    {"chunk_id": "kb_a_01", "source_id": "a", "text": "咳嗽时应用纸巾遮住口鼻并经常洗手"},
    {"chunk_id": "kb_b_01", "source_id": "b", "text": "高温天气不要把孩子留在停放的车辆中"},
    {"chunk_id": "kb_c_01", "source_id": "c", "text": "糖尿病是一种慢性病会导致血糖升高"},
    {"chunk_id": "kb_d_01", "source_id": "d", "text": "结核病是由结核分枝杆菌引起的传染病"},
]
QUERIES = [
    {"sample_id": "q1", "split": "dev", "bucket": "direct", "question": "咳嗽时怎样遮住口鼻？", "gold_evidence_ids": ["kb_a_01"]},
    {"sample_id": "q2", "split": "dev", "bucket": "direct", "question": "孩子能留在车辆里吗", "gold_evidence_ids": ["kb_b_01"]},
    {"sample_id": "q3", "split": "dev", "bucket": "out_of_scope", "question": "量子计算芯片", "gold_evidence_ids": []},
    {"sample_id": "q4", "split": "test", "bucket": "direct", "question": "糖尿病会导致什么", "gold_evidence_ids": ["kb_c_01"]},
    {"sample_id": "q5", "split": "test", "bucket": "near_miss", "question": "咳嗽药一天吃几次", "gold_evidence_ids": []},
]


def fake_encoder(texts: list[str]) -> np.ndarray:
    """字符双字哈希到 64 维：相同双字越多余弦越高，完全确定。"""
    out = np.zeros((len(texts), 64))
    for row, text in enumerate(texts):
        for i in range(len(text) - 1):
            out[row, zlib.crc32(text[i:i + 2].encode()) % 64] += 1
    return out


def write_dataset(directory: Path, chunks=CHUNKS, queries=QUERIES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    (directory / "queries.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries), encoding="utf-8")
    return directory


def test_wilson_interval_matches_known_value():
    low, high = rb.wilson_interval(5, 6)
    assert (low, high) == pytest.approx((0.4365, 0.9699), abs=1e-3)
    assert rb.wilson_interval(0, 0) is None
    low, high = rb.wilson_interval(0, 10)
    assert low == 0.0 and high < 0.31


def test_auroc_handles_ties_and_empty():
    assert rb.auroc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert rb.auroc([1.0], [1.0]) == 0.5
    assert rb.auroc([], [1.0]) is None


def test_random_hit_expectation():
    assert rb.random_hit_expectation(8, 1, 3) == pytest.approx(3 / 8)
    assert rb.random_hit_expectation(200, 1, 3) == pytest.approx(0.015)
    assert rb.random_hit_expectation(4, 2, 3) == 1.0
    assert rb.random_hit_expectation(10, 0, 3) == 0.0


def test_rank_chunks_stable_ties_and_positive_only():
    ids = ["b", "a", "c"]
    scores = np.array([1.0, 1.0, 0.0])
    assert rb.rank_chunks(scores, ids, positive_only=True) == [("a", 1.0), ("b", 1.0)]
    assert [cid for cid, _ in rb.rank_chunks(scores, ids, positive_only=False)] == ["a", "b", "c"]


def test_load_dataset_rejects_inconsistent_records(tmp_path):
    bad = [dict(QUERIES[0], gold_evidence_ids=["kb_missing"])]
    with pytest.raises(ValueError, match="不存在"):
        rb.load_dataset(write_dataset(tmp_path / "missing", queries=bad))
    wrong_bucket = [dict(QUERIES[2], bucket="direct")]
    with pytest.raises(ValueError, match="不一致"):
        rb.load_dataset(write_dataset(tmp_path / "bucket", queries=wrong_bucket))
    no_split = [{k: v for k, v in QUERIES[0].items() if k != "split"}]
    with pytest.raises(ValueError, match="split"):
        rb.load_dataset(write_dataset(tmp_path / "split", queries=no_split))


def test_bigram_bm25_beats_whitespace_on_chinese(tmp_path):
    dataset = rb.load_dataset(write_dataset(tmp_path))
    ws = rb.evaluate_system("bm25_ws", lambda: rb.Bm25Scorer("bm25_ws", "whitespace"), dataset, ["dev", "test"])
    bi = rb.evaluate_system("bm25_bigram", lambda: rb.Bm25Scorer("bm25_bigram", "cjk_bigram"), dataset, ["dev", "test"])
    assert bi["dev"]["retrieval"]["hit@3"]["rate"] == 1.0
    assert ws["dev"]["retrieval"]["hit@3"]["rate"] == 0.0
    # 无任何字面重合的范围外问题：现网规则（阈值 0）正确弃答
    assert bi["dev"]["abstention_at_production_threshold_0"]["false_answer_rate"]["k"] == 0


def test_abstention_counts_and_threshold_selection():
    records = [
        {"gold": ["x"], "first_gold_rank": 1, "top1_score": 5.0},
        {"gold": ["x"], "first_gold_rank": 2, "top1_score": 4.0},
        {"gold": ["x"], "first_gold_rank": None, "top1_score": 1.0},
        {"gold": [], "first_gold_rank": None, "top1_score": 2.0},
        {"gold": [], "first_gold_rank": None, "top1_score": 0.0},
    ]
    at_zero = rb.abstention_at(records, 0.0)
    assert at_zero["false_answer_rate"]["k"] == 1
    assert at_zero["wrong_evidence_rate"]["k"] == 1
    assert at_zero["over_abstention_rate"]["k"] == 0
    threshold = rb.select_threshold(records)
    assert 2.0 <= threshold < 4.0
    at_selected = rb.abstention_at(records, threshold)
    assert at_selected["false_answer_rate"]["k"] == 0
    assert at_selected["useful_answer_rate"]["k"] == 2


def test_coverage_scorer_penalises_partial_overlap():
    scorer = rb.CoverageScorer("coverage_bigram", "cjk_bigram")
    scorer.fit(CHUNKS)
    full = scorer.score("咳嗽时应用纸巾遮住口鼻")
    partial = scorer.score("咳嗽药一天吃几次")
    assert full.max() == pytest.approx(1.0)
    assert 0 < partial.max() < 0.5


def test_dense_and_hybrid_with_fake_encoder(tmp_path):
    dataset = rb.load_dataset(write_dataset(tmp_path))
    cache = rb.EmbeddingCache(fake_encoder)
    alpha, rows = rb.tune_alpha(cache, dataset, [0.0, 0.5, 1.0])
    assert alpha in {0.0, 0.5, 1.0} and len(rows) == 3
    registry = rb.build_registry(cache, alpha)
    result = rb.evaluate_system("hybrid", registry["hybrid"], dataset, ["dev"])
    assert result["dev"]["retrieval"]["hit@3"]["rate"] == 1.0
    assert result["latency"]["query_p95_ms"] >= 0


def test_scaling_is_deterministic_and_keeps_gold(tmp_path):
    dataset = rb.load_dataset(write_dataset(tmp_path))
    make = lambda: rb.Bm25Scorer("bm25_bigram", "cjk_bigram")  # noqa: E731
    first = rb.scaling_curve(make, dataset, [1, 2, 4], range(2))
    second = rb.scaling_curve(make, dataset, [1, 2, 4], range(2))
    assert first == second
    assert first[0]["docs"] == 1 and first[-1]["docs"] == 4
    # 全部文档都在时与整库评测一致；BM25 在 1-2 篇的退化语料里 IDF 为负，故不对极小规模作断言
    assert first[-1]["hit@3_mean"] == 1.0
    assert first[-1]["answerable_queries"] == 3


def test_latency_profile_really_encodes_queries(tmp_path):
    dataset = rb.load_dataset(write_dataset(tmp_path))
    calls: list[int] = []

    def counting_encoder(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return fake_encoder(texts)

    cache = rb.EmbeddingCache(counting_encoder)
    scorer = rb.DenseScorer("dense", cache)
    scorer.fit(dataset.chunks)
    calls.clear()
    rb.latency_profile(scorer, dataset.chunks, dataset.queries)
    # 预热 1 次 + 每条查询 1 次；文档向量已缓存，不应再编码
    assert sum(calls) == len(dataset.queries) + 1
