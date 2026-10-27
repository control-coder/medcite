"""v4 dev 混合检索矩阵评测（M8）。

在 v4 dev 上离线对比 5 种单路/融合检索（均为 parent_child 子块、无重排）：

- ``bm25_bigram``：BM25 字符二元组
- ``bm25_jieba``：BM25 jieba 分词（自定义词典由语料标题与章节路径规则生成）
- ``dense``：bge-small-zh 向量检索
- ``hybrid_bigram``：BM25 二元组 + 向量，RRF（k=60）融合
- ``hybrid_jieba``：BM25 jieba + 向量，RRF（k=60）融合

以及推荐流水线 ``pipeline``：hybrid + 交叉编码器重排（candidates 可配置）+ MMR（λ=0.7）。
指标：Hit@3、MRR@10、nDCG@10（relevant=2、partial=1）、Recall@20，以及每条查询的
检索耗时 p50/p90。只统计 answerable 的 popular_science 查询；intent 为
emergency/individual_care 的排除（按路由应直接拒答，不参与检索指标）。

标注读取依赖 M7 的 ``eval.v4_judgments``（load_judgments / chunk_relevance）；
该模块尚未就绪时直接报错退出，不静默跳过。报告输出到
``artifacts/reports/retrieval/hybrid-matrix-v4-<日期>.json``。

用法::

    python -m eval.hybrid_matrix                       # dev 全量
    python -m eval.hybrid_matrix --limit 4             # 前 4 条冒烟
    python -m eval.hybrid_matrix --rerank-candidates 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / ".cache/corpus/chunks/parent_child/chunks.jsonl"
QUERIES_PATH = ROOT / "examples/public_health_v4/queries.jsonl"
JUDGMENTS_PATH = ROOT / "examples/public_health_v4/judgments.jsonl"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# 无重排的方法 Recall 统计深度；Hit/MRR/nDCG 各自的 k 见 METRIC_DEPTHS
RECALL_DEPTH = 20
METRIC_DEPTHS = {"hit": 3, "mrr": 10, "ndcg": 10}
EXCLUDED_INTENTS = ("emergency", "individual_care", "out_of_scope")
LABEL_GAIN = {"irrelevant": 0, "partial": 1, "relevant": 2}
SINGLE_METHODS = ("bm25_bigram", "bm25_jieba", "dense", "hybrid_bigram", "hybrid_jieba")


def _fail(message: str) -> None:
    raise SystemExit(f"[hybrid_matrix] {message}")


def _load_judgments_module() -> Any:
    try:
        from eval.v4_judgments import chunk_relevance, load_judgments
    except ImportError as exc:  # M7 标注尚未就绪：给出清楚提示，不静默跳过
        _fail(f"无法导入 eval.v4_judgments（M7 标注接口）：{exc}")
    if not JUDGMENTS_PATH.exists():
        _fail(f"缺少句子级标注文件 {JUDGMENTS_PATH}；请先完成 M7 标注（python -m eval.v4_judgments --mode record）")
    return load_judgments, chunk_relevance


def load_queries() -> list[dict[str, Any]]:
    if not QUERIES_PATH.exists():
        _fail(f"缺少查询文件 {QUERIES_PATH}")
    queries = [
        json.loads(line) for line in QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        _fail(f"查询文件为空：{QUERIES_PATH}")
    return queries


def dev_popular_science(queries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """只保留 dev 中可回答的 popular_science 查询；其余意图排除并计数说明。"""
    picked = []
    for query in queries:
        if query.get("split") != "dev":
            continue
        if query.get("intent") != "popular_science":
            continue
        if query.get("answerable") is not True:
            continue  # 标注未完成时 answerable 为空；M7 冻结后为 True/False
        picked.append(query)
    return picked


def excluded_counts(queries: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for query in queries:
        if query.get("split") != "dev":
            continue
        intent = str(query.get("intent"))
        if intent != "popular_science":
            counts[intent] = counts.get(intent, 0) + 1
        elif query.get("answerable") is not True:
            counts["popular_science_unanswerable"] = counts.get("popular_science_unanswerable", 0) + 1
    return counts


class _ScoredItem:
    """mmr_select 的最小候选包装：只带 chunk_id 与 final_score。"""

    def __init__(self, result: Any, final_score: float) -> None:
        self.chunk_id = result.chunk_id
        self.final_score = final_score


def build_retrievers(chunks: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    """构建 5 种无重排方法与推荐流水线所需的检索器/重排器；模型只读本地缓存。"""
    from medidiag.rag.dense import DenseRetriever, load_local_encoder
    from medidiag.rag.hybrid import HybridRetriever
    from medidiag.rag.lexicon import build_lexicon
    from medidiag.rag.retrieval import Bm25Retriever

    lexicon = build_lexicon(chunks)
    bm25_bigram = Bm25Retriever(chunks, bm25_tokenizer="cjk_bigram")
    bm25_jieba = Bm25Retriever(chunks, bm25_tokenizer="jieba", lexicon=lexicon)
    encoder = load_local_encoder(EMBEDDING_MODEL, EMBEDDING_REVISION, device=args.device)
    dense = DenseRetriever(chunks, encoder=encoder, model_name=EMBEDDING_MODEL,
                           model_revision=EMBEDDING_REVISION)
    retrievers: dict[str, Any] = {
        "bm25_bigram": bm25_bigram,
        "bm25_jieba": bm25_jieba,
        "dense": dense,
        "hybrid_bigram": HybridRetriever(chunks, bm25=bm25_bigram, dense=dense,
                                         candidates=args.candidates, rrf_k=args.rrf_k),
        "hybrid_jieba": HybridRetriever(
            chunks,
            bm25=Bm25Retriever(chunks, bm25_tokenizer="jieba", lexicon=lexicon),
            dense=dense, candidates=args.candidates, rrf_k=args.rrf_k),
    }
    return retrievers


def evaluate_method(name: str, results_by_query: dict[str, list[tuple[str, float]]],
                    relevance: dict[str, dict[str, int]]) -> dict[str, Any]:
    """按查询名次计算 Hit@3、MRR@10、nDCG@10 与 Recall@20。"""
    hits, rr_sum, dcg_sum, recall_sum = 0, 0.0, 0.0, 0.0
    n = len(results_by_query)
    for query_id, ranked in results_by_query.items():
        labels = relevance[query_id]
        top = ranked[: RECALL_DEPTH]
        n_rel = sum(1 for _, rel in labels.items() if rel > 0)
        retrieved_rel = sum(1 for cid, _ in top if labels.get(cid, 0) > 0)
        recall_sum += retrieved_rel / n_rel if n_rel else 0.0
        if any(labels.get(cid, 0) > 0 for cid, _ in ranked[: METRIC_DEPTHS["hit"]]):
            hits += 1
        for rank, (cid, _) in enumerate(ranked[: METRIC_DEPTHS["mrr"]], 1):
            if labels.get(cid, 0) > 0:
                rr_sum += 1.0 / rank
                break
        dcg = sum(
            LABEL_GAIN["partial" if labels.get(cid, 0) == 1 else "relevant" if labels.get(cid, 0) == 2 else "irrelevant"]
            / math.log2(rank + 1)
            for rank, (cid, _) in enumerate(ranked[: METRIC_DEPTHS["ndcg"]], 1)
        )
        ideal = sorted((rel for rel in labels.values()), reverse=True)[: METRIC_DEPTHS["ndcg"]]
        idcg = sum(
            LABEL_GAIN["partial" if rel == 1 else "relevant" if rel == 2 else "irrelevant"]
            / math.log2(rank + 1)
            for rank, rel in enumerate(ideal, 1)
        )
        dcg_sum += dcg / idcg if idcg else 0.0
    return {
        "method": name,
        "queries": n,
        "hit@3": round(hits / n, 4) if n else None,
        "mrr@10": round(rr_sum / n, 4) if n else None,
        "ndcg@10": round(dcg_sum / n, 4) if n else None,
        "recall@20": round(recall_sum / n, 4) if n else None,
    }


def percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return round(ordered[index], 1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_judgments, chunk_relevance = _load_judgments_module()
    from medidiag.schemas import load_knowledge_chunks, read_jsonl

    if not CHUNKS_PATH.exists():
        _fail(f"缺少 v4 子块语料 {CHUNKS_PATH}；请按数据卡复现 M4–M6 流程")
    chunks = load_knowledge_chunks(CHUNKS_PATH)
    raw_chunks = read_jsonl(CHUNKS_PATH)
    queries = load_queries()
    picked = dev_popular_science(queries)
    if not picked:
        _fail("dev 中没有可统计的 answerable popular_science 查询；请确认 M7 已冻结 answerable 字段")
    if args.limit:
        picked = picked[: args.limit]

    judgments = load_judgments(JUDGMENTS_PATH)
    # 每条查询的相关性子块：句子级标注投影到 parent_child 子块（0/1/2）
    # chunk_relevance 逐块计算；按 doc_id 分桶后只对标注涉及的文档做投影，避免全语料 × 标注的笛卡尔积
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in raw_chunks:
        meta = chunk.get("metadata", {})
        doc_id = str(meta.get("doc_id") or chunk["source_id"])
        chunks_by_doc.setdefault(doc_id, []).append(chunk)
    relevance: dict[str, dict[str, int]] = {}
    for query in picked:
        query_id = query["sample_id"]
        rel: dict[str, int] = {}
        for doc_id, doc_labels in judgments.get(query_id, {}).items():
            for chunk in chunks_by_doc.get(doc_id, []):
                label = chunk_relevance(chunk["text"], doc_labels)
                if label != "irrelevant":
                    rel[chunk["chunk_id"]] = 2 if label == "relevant" else 1
        relevance[query_id] = rel
    picked = [q for q in picked if relevance[q["sample_id"]]]
    if not picked:
        _fail("选中查询在 judgments.jsonl 中没有正标签子块；请检查标注与语料是否对应")

    retrievers = build_retrievers(chunks, args)
    report: dict[str, Any] = {
        "schema_version": "hybrid-matrix-v1",
        "date": str(date.today()),
        "corpus": str(CHUNKS_PATH.relative_to(ROOT)),
        "queries_evaluated": len(picked),
        "excluded_dev_queries": excluded_counts(queries),
        "note": "只统计 answerable 的 popular_science 查询；emergency/individual_care/out_of_scope 排除",
        "params": {"candidates": args.candidates, "rrf_k": args.rrf_k,
                   "mmr_lambda": args.mmr_lambda, "rerank_candidates": args.rerank_candidates},
        "methods": [],
        "latency_ms": {},
    }

    for name in SINGLE_METHODS:
        retriever = retrievers[name]
        retriever.build_index()
        results_by_query: dict[str, list[tuple[str, float]]] = {}
        latencies: list[float] = []
        for query in picked:
            start = time.perf_counter()
            results = retriever.search(query["question"], top_k=RECALL_DEPTH)
            latencies.append((time.perf_counter() - start) * 1000)
            results_by_query[query["sample_id"]] = [(r.chunk_id, r.final_score) for r in results]
        report["methods"].append(evaluate_method(name, results_by_query, relevance))
        report["latency_ms"][name] = {"p50": percentile(latencies, 50), "p90": percentile(latencies, 90)}

    # 推荐流水线：hybrid（分词按 --tokenizer）+ 重排 + MMR
    from medidiag.rag.hybrid import mmr_select
    from medidiag.rag.rerank import Reranker, load_local_reranker

    pipeline_retriever = retrievers[f"hybrid_{args.tokenizer}"]
    reranker = Reranker(load_local_reranker(RERANK_MODEL, RERANK_REVISION, device=args.device),
                        model_name=RERANK_MODEL, model_revision=RERANK_REVISION,
                        candidates=args.rerank_candidates)
    results_by_query = {}
    latencies = []
    for query in picked:
        start = time.perf_counter()
        candidates = pipeline_retriever.search(query["question"], top_k=args.rerank_candidates)
        reranked = reranker.rerank(query["question"], [r for r in candidates], len(candidates))
        # mmr_select 只需要 chunk_id 与 final_score；用轻量包装避免依赖 EvidenceItem 的完整字段
        items = [_ScoredItem(r, score) for r, score in reranked]
        selected = mmr_select(items, vector_for=pipeline_retriever.vector_for,
                              top_k=RECALL_DEPTH, mmr_lambda=args.mmr_lambda)
        latencies.append((time.perf_counter() - start) * 1000)
        results_by_query[query["sample_id"]] = [(item.chunk_id, item.final_score) for item in selected]
    report["methods"].append(evaluate_method("pipeline", results_by_query, relevance))
    report["latency_ms"]["pipeline"] = {"p50": percentile(latencies, 50), "p90": percentile(latencies, 90)}

    out_dir = ROOT / "artifacts/reports/retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hybrid-matrix-v4-{date.today()}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path.relative_to(ROOT))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只评前 N 条 dev 查询")
    parser.add_argument("--tokenizer", choices=["bigram", "jieba"], default="bigram",
                        help="推荐流水线使用的 BM25 分词；单路对比两种都会输出")
    parser.add_argument("--candidates", type=int, default=20, help="混合检索每路候选数")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--rerank-candidates", type=int, default=20, help="流水线中交给重排的候选数")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"queries": report["queries_evaluated"], "report": report["report_path"],
                      "methods": [{m["method"]: {k: v for k, v in m.items() if k not in ("method", "queries")}}
                                  for m in report["methods"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
