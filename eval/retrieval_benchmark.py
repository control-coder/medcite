"""中文公开语料检索与弃答基准。

回答三个问题，全部带置信区间并区分 dev/test：

1. 不同检索器（BM25 空格/双字、IDF 覆盖率、稠密向量、混合）的 Hit@k、Recall@k、Precision@k、
   nDCG@k、MAP@10、MRR@10；以及作答决定的精确率、召回率与 F1。
2. 以 top1 分数作为置信度时，弃答阈值如何在“误答率”和“过度弃答率”之间取舍；
   阈值与混合权重只在 dev 上选择，test 只报告不调参。
3. 语料规模从小到大变化时 Hit@3 如何衰减（每条查询保留自己的金标准文档，
   其余文档随机抽样作干扰项）。

不调用任何在线模型。稠密检索读取本地缓存的嵌入模型（默认 bge-small-zh-v1.5）。

用法::

    python -m eval.retrieval_benchmark --dataset-dir examples/public_health_v2 \\
        --systems bm25_ws bm25_bigram coverage_bigram --splits dev test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from medidiag.rag.retrieval import tokenize_text

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "retrieval-benchmark-v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
# 模型卡建议的检索查询前缀；文档侧不加前缀。
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
TOP_N = 10
ANSWERABLE_BUCKETS = ("direct", "paraphrase", "multi")
UNANSWERABLE_BUCKETS = ("near_miss", "out_of_scope", "uncovered")


# ----------------------------------------------------------------- 统计工具


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """二项比例的 Wilson 区间；total=0 时无意义。"""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def proportion(successes: int, total: int) -> dict[str, Any]:
    interval = wilson_interval(successes, total)
    return {
        "k": successes,
        "n": total,
        "rate": successes / total if total else None,
        "ci95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
    }


def bootstrap_mean_ci(values: Sequence[float], resamples: int = 2000, seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)]


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Mann-Whitney 形式的 AUROC：随机正例分数高于随机负例的概率，平局记 0.5。"""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def random_hit_expectation(corpus_size: int, gold_count: int, k: int) -> float:
    """随机排序下 Hit@k 的期望：1 - C(N-g, k) / C(N, k)。"""
    if gold_count <= 0 or corpus_size <= 0:
        return 0.0
    k = min(k, corpus_size)
    if corpus_size - gold_count < k:
        return 1.0
    return 1.0 - math.comb(corpus_size - gold_count, k) / math.comb(corpus_size, k)


# ----------------------------------------------------------------- 数据集


@dataclass(frozen=True)
class Dataset:
    chunks: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    fingerprints: dict[str, str] = field(default_factory=dict)

    @property
    def doc_count(self) -> int:
        return len({c["source_id"] for c in self.chunks})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dataset(directory: Path) -> Dataset:
    chunks = _read_jsonl(directory / "chunks.jsonl")
    queries = _read_jsonl(directory / "queries.jsonl")
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("chunk_id 不唯一")
    known = set(ids)
    for query in queries:
        missing = set(query["gold_evidence_ids"]) - known
        if missing:
            raise ValueError(f"{query['sample_id']} 引用了不存在的 chunk: {sorted(missing)}")
        if query.get("bucket") not in ANSWERABLE_BUCKETS + UNANSWERABLE_BUCKETS:
            raise ValueError(f"{query['sample_id']} 缺少合法 bucket")
        if query.get("split") not in {"dev", "test"}:
            raise ValueError(f"{query['sample_id']} 缺少合法 split")
        answerable = bool(query["gold_evidence_ids"])
        if answerable != (query["bucket"] in ANSWERABLE_BUCKETS):
            raise ValueError(f"{query['sample_id']} 的 bucket 与金标准是否为空不一致")
    fingerprints = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("chunks.jsonl", "queries.jsonl")
    }
    return Dataset(chunks=chunks, queries=queries, fingerprints=fingerprints)


# ----------------------------------------------------------------- 检索系统


class Scorer(Protocol):
    """对一份语料建立索引，并为查询给出与 chunks 对齐的分数向量。"""

    name: str
    positive_only: bool

    def fit(self, chunks: Sequence[dict[str, Any]]) -> None: ...

    def score(self, query: str) -> np.ndarray: ...


class Bm25Scorer:
    positive_only = True

    def __init__(self, name: str, tokenizer: str) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self._bm25: BM25Okapi | None = None
        self._size = 0

    def fit(self, chunks: Sequence[dict[str, Any]]) -> None:
        self._size = len(chunks)
        self._bm25 = BM25Okapi([tokenize_text(c["text"], self.tokenizer) for c in chunks])

    def score(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError("fit() 尚未调用")
        tokens = tokenize_text(query, self.tokenizer)
        if not tokens:
            return np.zeros(self._size, dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)


class CoverageScorer:
    """查询词的 IDF 加权覆盖率，范围 [0, 1]；未登录词计入分母，天然惩罚“只沾边”。"""

    positive_only = True

    def __init__(self, name: str, tokenizer: str) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self._chunk_tokens: list[set[str]] = []
        self._idf: dict[str, float] = {}
        self._unseen_idf = 0.0

    def fit(self, chunks: Sequence[dict[str, Any]]) -> None:
        self._chunk_tokens = [set(tokenize_text(c["text"], self.tokenizer)) for c in chunks]
        size = len(chunks)
        document_frequency: Counter[str] = Counter()
        for tokens in self._chunk_tokens:
            document_frequency.update(tokens)
        self._idf = {t: math.log(1 + (size - df + 0.5) / (df + 0.5)) for t, df in document_frequency.items()}
        self._unseen_idf = math.log(1 + (size + 0.5) / 0.5)

    def idf(self, token: str) -> float:
        return self._idf.get(token, self._unseen_idf)

    def score(self, query: str) -> np.ndarray:
        query_tokens = set(tokenize_text(query, self.tokenizer))
        total = sum(self.idf(t) for t in query_tokens)
        if total <= 0:
            return np.zeros(len(self._chunk_tokens), dtype=np.float64)
        return np.asarray(
            [sum(self.idf(t) for t in query_tokens & tokens) / total for tokens in self._chunk_tokens],
            dtype=np.float64,
        )


class EmbeddingCache:
    """文本 -> 归一化向量。encoder 可注入，测试无需下载模型。"""

    def __init__(self, encoder: Callable[[list[str]], np.ndarray]) -> None:
        self._encoder = encoder
        self._store: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if not use_cache or t not in self._store))
        if missing:
            vectors = np.asarray(self._encoder(missing), dtype=np.float64)
            norms = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            for text, vector in zip(missing, vectors / norms, strict=True):
                self._store[text] = vector
        return np.stack([self._store[t] for t in texts])


class DenseScorer:
    positive_only = False

    def __init__(self, name: str, cache: EmbeddingCache, query_prefix: str = "") -> None:
        self.name = name
        self.cache = cache
        self.query_prefix = query_prefix
        self._matrix: np.ndarray | None = None
        self.use_query_cache = True

    def fit(self, chunks: Sequence[dict[str, Any]]) -> None:
        self._matrix = self.cache.encode([c["text"] for c in chunks])

    def score(self, query: str) -> np.ndarray:
        if self._matrix is None:
            raise RuntimeError("fit() 尚未调用")
        vector = self.cache.encode([self.query_prefix + query], use_cache=self.use_query_cache)[0]
        return np.asarray(self._matrix @ vector, dtype=np.float64)


class HybridScorer:
    """alpha * 词法覆盖率 + (1 - alpha) * 稠密余弦；两路分数都已在固定量纲内，可直接作置信度。"""

    positive_only = False

    def __init__(self, name: str, lexical: CoverageScorer, dense: DenseScorer, alpha: float) -> None:
        self.name = name
        self.lexical = lexical
        self.dense = dense
        self.alpha = alpha

    def fit(self, chunks: Sequence[dict[str, Any]]) -> None:
        self.lexical.fit(chunks)
        self.dense.fit(chunks)

    def score(self, query: str) -> np.ndarray:
        return self.alpha * self.lexical.score(query) + (1 - self.alpha) * self.dense.score(query)


# ----------------------------------------------------------------- 评估


def rank_chunks(scores: np.ndarray, chunk_ids: Sequence[str], positive_only: bool, top_n: int = TOP_N) -> list[tuple[str, float]]:
    order = sorted(range(len(chunk_ids)), key=lambda i: (-float(scores[i]), chunk_ids[i]))
    ranked = [(chunk_ids[i], float(scores[i])) for i in order if not positive_only or scores[i] > 0]
    return ranked[:top_n]


def run_queries(scorer: Scorer, chunks: Sequence[dict[str, Any]], queries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    scorer.fit(chunks)
    chunk_ids = [c["chunk_id"] for c in chunks]
    records = []
    for query in queries:
        ranked = rank_chunks(scorer.score(query["question"]), chunk_ids, scorer.positive_only)
        gold = set(query["gold_evidence_ids"])
        first_gold = next((r for r, (cid, _) in enumerate(ranked, 1) if cid in gold), None)
        records.append({
            "sample_id": query["sample_id"], "split": query["split"], "bucket": query["bucket"],
            "gold": sorted(gold), "retrieved": [cid for cid, _ in ranked[:5]],
            # 每个金标准片段在结果里的名次（从 1 起，只含前 TOP_N 名内的），供 Recall/Precision/nDCG/MAP 使用
            "gold_ranks": sorted(r for r, (cid, _) in enumerate(ranked, 1) if cid in gold),
            "top1_score": ranked[0][1] if ranked else 0.0, "first_gold_rank": first_gold,
            "gold_in_top3": len(gold & {cid for cid, _ in ranked[:3]}),
        })
    return records


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    return {"mean": round(statistics.mean(values), 4) if values else None, "ci95": bootstrap_mean_ci(values)}


def _gold_ranks_within(record: dict[str, Any], k: int) -> list[int]:
    return [rank for rank in record["gold_ranks"] if rank <= k]


def _recall_at(record: dict[str, Any], k: int) -> float:
    """前 k 条里找回的金标准片段数 / 金标准片段总数。"""
    return len(_gold_ranks_within(record, k)) / len(record["gold"])


def _precision_at(record: dict[str, Any], k: int) -> float:
    """前 k 条里属于金标准的比例。多数问题只有 1 个金标准，所以上限是 1/k，这是该指标本身的性质。"""
    return len(_gold_ranks_within(record, k)) / k


def _ndcg_at(record: dict[str, Any], k: int) -> float:
    """二元相关度的 nDCG：命中越靠前得分越高，与理想排序（金标准全部排在最前）相比。"""
    dcg = sum(1 / math.log2(rank + 1) for rank in _gold_ranks_within(record, k))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(record["gold"]), k)))
    return dcg / ideal if ideal else 0.0


def _average_precision(record: dict[str, Any]) -> float:
    """AP@TOP_N：每个金标准片段出现处的 Precision 之平均，分母为金标准总数（没找回的记 0）。"""
    total = sum((i + 1) / rank for i, rank in enumerate(record["gold_ranks"]))
    return total / len(record["gold"])


def retrieval_metrics(records: Sequence[dict[str, Any]], corpus_size: int) -> dict[str, Any]:
    answerable = [r for r in records if r["gold"]]
    out: dict[str, Any] = {"answerable_n": len(answerable)}
    for k in (1, 3, 5):
        hits = sum(1 for r in answerable if r["first_gold_rank"] is not None and r["first_gold_rank"] <= k)
        out[f"hit@{k}"] = proportion(hits, len(answerable))
    recalls = [r["gold_in_top3"] / len(r["gold"]) for r in answerable]
    out["recall@3"] = {"mean": round(statistics.mean(recalls), 4) if recalls else None, "ci95": bootstrap_mean_ci(recalls)}
    reciprocal = [1 / r["first_gold_rank"] if r["first_gold_rank"] else 0.0 for r in answerable]
    out["mrr@10"] = {"mean": round(statistics.mean(reciprocal), 4) if reciprocal else None, "ci95": bootstrap_mean_ci(reciprocal)}
    # 常用的排序指标：金标准片段可能不止一个，所以 Recall、Precision、nDCG、MAP 与 Hit 不同
    for k in (1, 5, 10):
        out[f"recall@{k}"] = _mean_ci([_recall_at(r, k) for r in answerable])
    for k in (1, 3, 5):
        out[f"precision@{k}"] = _mean_ci([_precision_at(r, k) for r in answerable])
    for k in (3, 10):
        out[f"ndcg@{k}"] = _mean_ci([_ndcg_at(r, k) for r in answerable])
    out["map@10"] = _mean_ci([_average_precision(r) for r in answerable])
    baseline = [random_hit_expectation(corpus_size, len(r["gold"]), 3) for r in answerable]
    out["random_hit@3"] = round(statistics.mean(baseline), 4) if baseline else None
    return out


def abstention_at(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """top1 分数严格大于阈值才作答；阈值 0 等价于现网“有正分命中即作答”。"""
    answerable = [r for r in records if r["gold"]]
    unanswerable = [r for r in records if not r["gold"]]
    answered = [r for r in answerable if r["top1_score"] > threshold]
    answered_hit = [r for r in answered if r["first_gold_rank"] is not None and r["first_gold_rank"] <= 3]
    answered_total = len(answered) + sum(r["top1_score"] > threshold for r in unanswerable)
    precision = len(answered_hit) / answered_total if answered_total else None
    recall = len(answered_hit) / len(answerable) if answerable else None
    return {
        "threshold": round(threshold, 6),
        # 不可回答的问题仍被作答
        "false_answer_rate": proportion(sum(r["top1_score"] > threshold for r in unanswerable), len(unanswerable)),
        # 可回答的问题被拒答
        "over_abstention_rate": proportion(len(answerable) - len(answered), len(answerable)),
        # 可回答且已作答的问题中，前 3 条证据不含金标准（作答但给错证据）
        "wrong_evidence_rate": proportion(len(answered) - len(answered_hit), len(answerable)),
        # 可回答、作答且前 3 条命中
        "useful_answer_rate": proportion(len(answered_hit), len(answerable)),
        # 把“作答且前 3 条含金标准”当作正确回答：精确率 = 正确回答数 / 全部回答数（含不该答的），
        # 召回率 = 正确回答数 / 可回答问题数（与 useful_answer_rate 相同），F1 为两者的调和平均
        "answer_precision": proportion(len(answered_hit), answered_total),
        "answer_recall": proportion(len(answered_hit), len(answerable)),
        "answer_f1": round(2 * precision * recall / (precision + recall), 4)
        if precision and recall else (0.0 if answered_total else None),
    }


def select_threshold(dev_records: Sequence[dict[str, Any]]) -> float:
    """在 dev 上最大化 Youden 型效用：有用作答率 - 误答率；平局取更低阈值（覆盖更大）。"""
    scores = sorted({0.0, *(r["top1_score"] for r in dev_records)})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)]
    best_threshold, best_utility = 0.0, -math.inf
    for threshold in candidates:
        result = abstention_at(dev_records, threshold)
        utility = (result["useful_answer_rate"]["rate"] or 0.0) - (result["false_answer_rate"]["rate"] or 0.0)
        if utility > best_utility + 1e-12:
            best_threshold, best_utility = threshold, utility
    return best_threshold


def risk_coverage_curve(records: Sequence[dict[str, Any]], points: int = 12) -> list[dict[str, Any]]:
    scores = sorted({r["top1_score"] for r in records})
    if not scores:
        return []
    picks = sorted({0.0, *(scores[min(len(scores) - 1, int(i * len(scores) / points))] for i in range(points))})
    curve = []
    for threshold in picks:
        result = abstention_at(records, threshold)
        curve.append({
            "threshold": result["threshold"],
            "answer_coverage": round(1 - (result["over_abstention_rate"]["rate"] or 0.0), 4),
            "false_answer_rate": round(result["false_answer_rate"]["rate"] or 0.0, 4),
            "useful_answer_rate": round(result["useful_answer_rate"]["rate"] or 0.0, 4),
        })
    return curve


def split_summary(records: Sequence[dict[str, Any]], corpus_size: int, threshold: float) -> dict[str, Any]:
    by_bucket = {
        bucket: retrieval_metrics([r for r in records if r["bucket"] == bucket], corpus_size)["hit@3"]
        for bucket in ANSWERABLE_BUCKETS
    }
    positives = [r["top1_score"] for r in records if r["gold"] and r["first_gold_rank"] and r["first_gold_rank"] <= 3]
    negatives = [r["top1_score"] for r in records if not r["gold"]]
    value = auroc(positives, negatives)
    return {
        "query_count": len(records),
        "retrieval": retrieval_metrics(records, corpus_size),
        "hit@3_by_bucket": by_bucket,
        "abstention_at_production_threshold_0": abstention_at(records, 0.0),
        "abstention_at_dev_threshold": abstention_at(records, threshold),
        # 按不可回答的类型拆开：越像“同主题但没答案”，越难靠分数挡住
        "false_answer_by_bucket_at_dev_threshold": {
            bucket: proportion(
                sum(r["top1_score"] > threshold for r in records if r["bucket"] == bucket),
                sum(1 for r in records if r["bucket"] == bucket))
            for bucket in UNANSWERABLE_BUCKETS
        },
        "top1_score_auroc": round(value, 4) if value is not None else None,
        "risk_coverage": risk_coverage_curve(records),
    }


def latency_profile(scorer: Scorer, chunks: Sequence[dict[str, Any]], queries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """单查询打分耗时（含查询向量化，不含模型加载；稠密路径的 index_build_ms 只是向量查表，文档向量化另计）。"""
    started = time.perf_counter()
    scorer.fit(chunks)
    build_ms = (time.perf_counter() - started) * 1000
    # 稠密路径必须每次真实编码查询，否则测到的是字典查找。
    for part in (scorer, getattr(scorer, "dense", None)):
        if isinstance(part, DenseScorer):
            part.use_query_cache = False
    scorer.score(queries[0]["question"])  # 预热，不计时
    timings = []
    for query in queries:
        started = time.perf_counter()
        scorer.score(query["question"])
        timings.append((time.perf_counter() - started) * 1000)
    ordered = sorted(timings)
    return {
        "index_build_ms": round(build_ms, 2),
        "query_p50_ms": round(statistics.median(ordered), 3),
        "query_p95_ms": round(ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 3),
    }


def scaling_curve(
    make_scorer: Callable[[], Scorer],
    dataset: Dataset,
    sizes: Sequence[int],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """每条可回答查询保留自己的金标准文档，其余文档随机抽样补足到 size 篇后重新建索引。"""
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in dataset.chunks:
        by_doc[chunk["source_id"]].append(chunk)
    chunk_doc = {c["chunk_id"]: c["source_id"] for c in dataset.chunks}
    answerable = [q for q in dataset.queries if q["gold_evidence_ids"]]
    rows = []
    for size in sizes:
        per_seed = []
        for seed in seeds:
            rng = random.Random(f"{seed}:{size}")
            hits = 0
            for query in answerable:
                gold_docs = {chunk_doc[g] for g in query["gold_evidence_ids"]}
                others = sorted(set(by_doc) - gold_docs)
                extra = rng.sample(others, max(0, min(size, len(by_doc)) - len(gold_docs)))
                subset = [c for doc in sorted(gold_docs | set(extra)) for c in by_doc[doc]]
                record = run_queries(make_scorer(), subset, [query])[0]
                hits += int(record["first_gold_rank"] is not None and record["first_gold_rank"] <= 3)
            per_seed.append(hits / len(answerable))
        rows.append({
            "docs": min(size, len(by_doc)), "seeds": len(seeds),
            "hit@3_mean": round(statistics.mean(per_seed), 4),
            "hit@3_min": round(min(per_seed), 4), "hit@3_max": round(max(per_seed), 4),
            "answerable_queries": len(answerable),
        })
    return rows


# ----------------------------------------------------------------- 组装与 CLI


def build_registry(cache: EmbeddingCache | None, alpha: float) -> dict[str, Callable[[], Scorer]]:
    registry: dict[str, Callable[[], Scorer]] = {
        "bm25_ws": lambda: Bm25Scorer("bm25_ws", "whitespace"),
        "bm25_bigram": lambda: Bm25Scorer("bm25_bigram", "cjk_bigram"),
        "coverage_bigram": lambda: CoverageScorer("coverage_bigram", "cjk_bigram"),
    }
    if cache is not None:
        registry["dense"] = lambda: DenseScorer("dense", cache, BGE_ZH_QUERY_PREFIX)
        registry["hybrid"] = lambda: HybridScorer(
            "hybrid", CoverageScorer("coverage_bigram", "cjk_bigram"),
            DenseScorer("dense", cache, BGE_ZH_QUERY_PREFIX), alpha)
    return registry


def tune_alpha(cache: EmbeddingCache, dataset: Dataset, grid: Sequence[float]) -> tuple[float, list[dict[str, float]]]:
    """只用 dev 的 MRR@10 选混合权重；平局取更小的 alpha（更依赖稠密向量）。"""
    dev = [q for q in dataset.queries if q["split"] == "dev"]
    rows = []
    for alpha in grid:
        scorer = build_registry(cache, alpha)["hybrid"]()
        records = run_queries(scorer, dataset.chunks, dev)
        metrics = retrieval_metrics(records, len(dataset.chunks))
        rows.append({"alpha": alpha, "dev_mrr@10": metrics["mrr@10"]["mean"] or 0.0,
                     "dev_hit@3": metrics["hit@3"]["rate"] or 0.0})
    best = max(rows, key=lambda r: (round(r["dev_mrr@10"], 6), -r["alpha"]))
    return best["alpha"], rows


def load_sentence_transformer(model: str, revision: str | None) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model, revision=revision, device="cpu")
    return lambda texts: np.asarray(encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False))


def evaluate_system(name: str, factory: Callable[[], Scorer], dataset: Dataset, splits: Sequence[str]) -> dict[str, Any]:
    scorer = factory()
    records = run_queries(scorer, dataset.chunks, dataset.queries)
    corpus_size = len(dataset.chunks)
    dev_records = [r for r in records if r["split"] == "dev"]
    threshold = select_threshold(dev_records) if dev_records else 0.0
    result: dict[str, Any] = {"name": name, "dev_selected_threshold": round(threshold, 6)}
    for split in splits:
        subset = [r for r in records if r["split"] == split]
        result[split] = split_summary(subset, corpus_size, threshold)
    result["latency"] = latency_profile(factory(), dataset.chunks, dataset.queries)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "examples/public_health_v2")
    parser.add_argument("--systems", nargs="+", default=["bm25_ws", "bm25_bigram", "coverage_bigram", "dense", "hybrid"])
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["dev", "test"])
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--scaling-sizes", nargs="*", type=int, default=[8, 16, 32, 51])
    parser.add_argument("--scaling-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_dir)
    needs_dense = any(s in {"dense", "hybrid"} for s in args.systems)
    cache = EmbeddingCache(load_sentence_transformer(args.embedding_model, args.embedding_revision)) if needs_dense else None
    corpus_embedding_s = None
    if cache is not None:
        started = time.perf_counter()
        cache.encode([c["text"] for c in dataset.chunks])
        corpus_embedding_s = round(time.perf_counter() - started, 3)
    alpha, alpha_rows = (tune_alpha(cache, dataset, [round(i / 10, 1) for i in range(11)])
                         if cache is not None and "hybrid" in args.systems else (0.5, []))
    registry = build_registry(cache, alpha)
    unknown = [s for s in args.systems if s not in registry]
    if unknown:
        raise SystemExit(f"未知或不可用的系统: {unknown}")

    systems = {name: evaluate_system(name, registry[name], dataset, args.splits) for name in args.systems}
    scaling = {name: scaling_curve(registry[name], dataset, args.scaling_sizes, range(args.scaling_seeds))
               for name in args.systems if name in {"bm25_bigram", "dense", "hybrid"}} if args.scaling_sizes else {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "dir": str(args.dataset_dir), "chunks": len(dataset.chunks), "docs": dataset.doc_count,
            "queries": len(dataset.queries),
            "queries_by_split_bucket": {f"{s}/{b}": n for (s, b), n in sorted(
                Counter((q["split"], q["bucket"]) for q in dataset.queries).items())},
            "sha256": dataset.fingerprints,
        },
        "config": {"embedding_model": args.embedding_model if needs_dense else None,
                   "query_prefix": BGE_ZH_QUERY_PREFIX if needs_dense else None,
                   "hybrid_alpha": alpha if "hybrid" in args.systems else None,
                   "alpha_grid_dev": alpha_rows, "top_n": TOP_N,
                   "corpus_embedding_s_cpu": corpus_embedding_s, "external_model_calls": 0,
                   "selection_rule": "alpha 与弃答阈值仅在 dev 上选择；test 只报告"},
        "systems": systems,
        "scaling": scaling,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    out = args.out or ROOT / "artifacts/reports/retrieval" / (
        "retrieval-benchmark-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, system in systems.items():
        for split in args.splits:
            metrics = system[split]["retrieval"]
            print(f"{name:16s}{split:5s} hit@3={metrics['hit@3']['rate']:.3f} mrr@10={metrics['mrr@10']['mean']:.3f} "
                  f"false_answer@0={system[split]['abstention_at_production_threshold_0']['false_answer_rate']['rate']}")
    print("报告：", out)


if __name__ == "__main__":
    main()
