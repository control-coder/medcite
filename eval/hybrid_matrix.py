"""v4 dev 混合检索矩阵评测（M8）。

在 v4 dev 上离线对比 5 种单路/融合检索（均为 parent_child 子块、无重排）：

- ``bm25_bigram``：BM25 字符二元组
- ``bm25_jieba``：BM25 jieba 分词（自定义词典由语料标题与章节路径规则生成）
- ``dense``：bge-small-zh 向量检索
- ``hybrid_bigram``：BM25 二元组 + 向量，RRF（k=60）融合
- ``hybrid_jieba``：BM25 jieba + 向量，RRF（k=60）融合

两种 hybrid 在 RRF 融合后再做 MMR（λ=0.7）去冗余；不含交叉编码器重排（重排见 M5）。
指标：Hit@3、MRR@10、nDCG@5（relevant=2、partial=1）、Recall@20、平均送入字符数
（top3 子块换成去重父块），以及每条查询的检索耗时 p50/p90。只统计 answerable 的 popular_science 查询；intent 为
emergency/individual_care 的排除（按路由应直接拒答，不参与检索指标）。

标注读取依赖 M7 的 ``eval.v4_judgments``（load_judgments / chunk_relevance）；
该模块尚未就绪时直接报错退出，不静默跳过。报告输出到
``artifacts/reports/retrieval/hybrid-matrix-v4-<日期>.json``。

用法::

    python -m eval.hybrid_matrix                       # dev 全量
    python -m eval.hybrid_matrix --limit 4             # 前 4 条冒烟
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
METRIC_DEPTHS = {"hit": 3, "mrr": 10, "ndcg": 5}
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
        "ndcg@5": round(dcg_sum / n, 4) if n else None,
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

    import torch

    torch.set_num_threads(2)
    retrievers = build_retrievers(chunks, args)
    report: dict[str, Any] = {
        "schema_version": "hybrid-matrix-v1",
        "date": str(date.today()),
        "corpus": str(CHUNKS_PATH.relative_to(ROOT)),
        "queries_evaluated": len(picked),
        "excluded_dev_queries": excluded_counts(queries),
        "note": "只统计 answerable 的 popular_science 查询；emergency/individual_care/out_of_scope 排除",
        "params": {"candidates": args.candidates, "rrf_k": args.rrf_k,
                   "mmr_lambda": args.mmr_lambda, "chunking": "parent_child", "split": "dev"},
        "methods": [],
        "latency_ms": {},
    }

    from medidiag.rag.hybrid import mmr_select

    by_id = {c["chunk_id"]: c for c in raw_chunks}
    parents_path = CHUNKS_PATH.with_name("parents.jsonl")
    parent_text = {c["chunk_id"]: c["text"] for c in read_jsonl(parents_path)} if parents_path.exists() else {}

    def context_chars(ranked_ids: list[str]) -> int:
        """top3 子块换成去重后的父块送入模型的字数。"""
        seen: list[str] = []
        for cid in ranked_ids[:METRIC_DEPTHS["hit"]]:
            pid = by_id[cid].get("metadata", {}).get("parent_id")
            if pid and pid not in seen:
                seen.append(pid)
        return sum(len(parent_text[pid]) for pid in seen if pid in parent_text)

    started = time.perf_counter()
    for name in SINGLE_METHODS:
        retriever = retrievers[name]
        if name.startswith("hybrid"):  # 向量只编码一次：dense 已建索引，混合检索只补建 BM25
            retriever.bm25.build_index()
        else:
            retriever.build_index()
        results_by_query: dict[str, list[tuple[str, float]]] = {}
        chars: list[int] = []
        latencies: list[float] = []
        for count, query in enumerate(picked, 1):
            start = time.perf_counter()
            results = retriever.search(query["question"], top_k=RECALL_DEPTH)
            if name.startswith("hybrid"):
                results = mmr_select(results, vector_for=retriever.vector_for, top_k=RECALL_DEPTH,
                                     mmr_lambda=args.mmr_lambda)
            latencies.append((time.perf_counter() - start) * 1000)
            ranked = [(r.chunk_id, r.final_score) for r in results]
            results_by_query[query["sample_id"]] = ranked
            chars.append(context_chars([cid for cid, _ in ranked]))
            if count % 10 == 0:
                print(f"[{name}] {count}/{len(picked)} 已用 {time.perf_counter() - started:.0f}s", flush=True)
        entry = evaluate_method(name, results_by_query, relevance)
        entry["context_chars_mean"] = round(sum(chars) / len(chars), 1)
        report["methods"].append(entry)
        report["latency_ms"][name] = {"p50": percentile(latencies, 50), "p90": percentile(latencies, 90)}
        print(f"[{name}] 完成 {entry}", flush=True)

    out_dir = ROOT / "artifacts/reports/retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hybrid-matrix-v4-{date.today()}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path.relative_to(ROOT))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只评前 N 条 dev 查询")
    parser.add_argument("--candidates", type=int, default=20, help="混合检索每路候选数")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"queries": report["queries_evaluated"], "report": report["report_path"],
                      "methods": [{m["method"]: {k: v for k, v in m.items() if k not in ("method", "queries")}}
                                  for m in report["methods"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
