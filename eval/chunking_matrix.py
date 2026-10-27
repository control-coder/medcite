"""M6 分块策略离线对比：2 种策略 × 2 种检索（BM25 cjk_bigram、bge-small-zh 向量）。

在迁移后的 v3 可回答查询上（``examples/public_health_v4/v3_gold_mapping.json``），
对比 Hit@3、MRR（前 10 名）与平均送入模型字数（top_k=3，与
``configs/application_dense_agent_rerank.yaml`` 一致）：

- 检索语料是 v4 全部子块（不是 v3 的 498 条）；父子分块只索引子块，父块不进索引。
- 递归切分按命中块本身计字数；父子分块按命中子块取其父块并去重后计字数。
- gold 映射失败的查询从分母中排除，并在报告中单列排除数与清单。
- dev 与 test 分别报告；选参只用 dev，test 只报告不调参。

不产生任何调用费用：BM25 与向量都是本地离线计算；向量编码在 GPU 可用时用 GPU，
否则 CPU；不做持久向量缓存（M9 的事），但本次运行内复用同一向量矩阵。

用法::

    python -m eval.chunking_matrix \
        --chunks .cache/corpus/chunks \
        --mapping examples/public_health_v4/v3_gold_mapping.json \
        --v3 examples/public_health_v3 \
        --out artifacts/reports/retrieval/chunking-matrix-v4-20261027.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval import retrieval_benchmark as rb  # noqa: E402
from medidiag.schemas import read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "chunking-matrix-v1"
TOP_K = 3
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def _chunk_dicts(chunks_path: Path) -> list[dict[str, Any]]:
    """读检索块 JSONL 为 retrieval_benchmark 用的 dict（只取需要字段）。"""
    return read_jsonl(chunks_path)


def _query_metrics(gold: set[str], ranked_ids: list[str]) -> dict[str, float]:
    """一条查询的 Hit@3 与 MRR（前 10）。"""
    first = next((r for r, cid in enumerate(ranked_ids, 1) if cid in gold), None)
    return {
        "hit@3": float(first is not None and first <= 3),
        "mrr@10": 1 / first if first is not None and first <= 10 else 0.0,
    }


def _context_chars(strategy: str, ranked_ids: list[str],
                   by_id: dict[str, dict[str, Any]],
                   parent_by_id: dict[str, dict[str, Any]]) -> int:
    """top_k=3 送入模型的字数。父子分块取命中子块的父块去重，递归切分按块计。"""
    top = ranked_ids[:TOP_K]
    if strategy == "parent_child":
        seen: list[str] = []
        for cid in top:
            pid = by_id[cid].get("metadata", {}).get("parent_id")
            if pid and pid not in seen:
                seen.append(pid)
        return sum(len(parent_by_id[pid]["text"]) for pid in seen if pid in parent_by_id)
    return sum(len(by_id[cid]["text"]) for cid in top)


def evaluate_strategy(
    strategy: str,
    chunks_dir: Path,
    mapping: dict[str, Any],
    queries: list[dict[str, Any]],
    cache: rb.EmbeddingCache,
) -> dict[str, Any]:
    """一种策略：BM25 与向量两路，在可用可回答查询上算指标与送入字数。"""
    chunks = _chunk_dicts(chunks_dir / strategy / "chunks.jsonl")
    by_id = {c["chunk_id"]: c for c in chunks}
    parents_path = chunks_dir / strategy / "parents.jsonl"
    parent_by_id = ({c["chunk_id"]: c for c in read_jsonl(parents_path)}
                    if parents_path.exists() else {})

    strat_map = mapping["strategies"][strategy]
    query_gold = strat_map["queries"]
    # 可用的可回答查询：v3 原可回答，且迁移后 gold 非空
    usable = [q for q in queries
              if q["gold_evidence_ids"] and query_gold[q["sample_id"]]["gold"]]

    scorers: dict[str, rb.Scorer] = {
        "bm25_bigram": rb.Bm25Scorer("bm25_bigram", "cjk_bigram"),
        "dense": rb.DenseScorer("dense", cache, rb.BGE_ZH_QUERY_PREFIX),
    }
    corpus_size = len(chunks)
    table_row = sum(1 for c in chunks if c.get("metadata", {}).get("content_kind") == "table_row")

    result: dict[str, Any] = {
        "corpus": {
            "child_chunks": corpus_size,
            "parent_chunks": len(parent_by_id),
            "table_row_chunks": table_row,
            "child_chars_mean": round(statistics.mean(len(c["text"]) for c in chunks), 1),
            "parent_chars_mean": (round(statistics.mean(len(c["text"]) for c in parent_by_id.values()), 1)
                                  if parent_by_id else 0.0),
            "documents": len({c["source_id"] for c in chunks}),
        },
        "usable_answerable": len(usable),
        "excluded_count": strat_map["excluded_count"],
        "excluded_queries": strat_map["excluded_queries"],
        "retrievers": {},
    }

    for retriever_name, scorer in scorers.items():
        scorer.fit(chunks)
        chunk_ids = [c["chunk_id"] for c in chunks]
        per_query: dict[str, dict[str, Any]] = {}
        for query in usable:
            scores = scorer.score(query["question"])
            order = sorted(range(len(chunk_ids)), key=lambda i: (-float(scores[i]), chunk_ids[i]))
            ranked_ids = [chunk_ids[i] for i in order[: rb.TOP_N]]
            gold = set(query_gold[query["sample_id"]]["gold"])
            metrics = _query_metrics(gold, ranked_ids)
            metrics["context_chars"] = float(
                _context_chars(strategy, ranked_ids, by_id, parent_by_id))
            per_query[query["sample_id"]] = metrics

        retriever_result: dict[str, Any] = {}
        for split in ("dev", "test"):
            rows = [per_query[q["sample_id"]] for q in usable if q["split"] == split]
            if not rows:
                retriever_result[split] = {"n": 0}
                continue
            hit3 = [r["hit@3"] for r in rows]
            mrr = [r["mrr@10"] for r in rows]
            chars = [r["context_chars"] for r in rows]
            retriever_result[split] = {
                "n": len(rows),
                "hit@3": rb.proportion(int(sum(hit3)), len(rows)),
                "mrr@10": {"mean": round(statistics.mean(mrr), 4),
                           "ci95": rb.bootstrap_mean_ci(mrr)},
                "context_chars_mean": round(statistics.mean(chars), 1),
                "context_chars_ci95": rb.bootstrap_mean_ci(chars),
            }
        result["retrievers"][retriever_name] = retriever_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, default=ROOT / ".cache/corpus/chunks")
    parser.add_argument("--mapping", type=Path,
                        default=ROOT / "examples/public_health_v4/v3_gold_mapping.json")
    parser.add_argument("--v3", type=Path, default=ROOT / "examples/public_health_v3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    queries = read_jsonl(args.v3 / "queries.jsonl")

    import torch  # 仅用于探测 GPU；不可用时回退 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = rb.EmbeddingCache(rb.load_sentence_transformer(EMBEDDING_MODEL, EMBEDDING_REVISION)
                              if device == "cpu" else _gpu_encoder())

    strategies = ["recursive", "parent_child"]
    results = {s: evaluate_strategy(s, args.chunks, mapping, queries, cache) for s in strategies}

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "strategies": strategies,
            "retrievers": ["bm25_bigram", "dense"],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "device": device,
            "top_k": TOP_K,
            "metrics": ["hit@3", "mrr@10", "context_chars_mean"],
            "corpus": "v4 全部子块（递归切分块 / 父子子块），父块不进索引",
            "selection_rule": "结论只按 dev 给出；test 只报告不调参",
            "external_model_calls": 0,
        },
        "dataset": {
            "mapping": str(args.mapping),
            "v3_queries": str(args.v3 / "queries.jsonl"),
        },
        "strategies": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for strat in strategies:
        r = results[strat]
        print(f"== {strat}  子块 {r['corpus']['child_chunks']} 父块 {r['corpus']['parent_chunks']}"
              f" 可用 {r['usable_answerable']} 排除 {r['excluded_count']}")
        for retriever, splits in r["retrievers"].items():
            for split in ("dev", "test"):
                row = splits[split]
                if row.get("n"):
                    print(f"  {retriever:12s} {split:4s} hit@3={row['hit@3']['rate']:.3f} "
                          f"mrr@10={row['mrr@10']['mean']:.3f} 字数={row['context_chars_mean']:.0f}")
    print("报告：", args.out)


def _gpu_encoder() -> Any:
    """GPU 可用时的向量编码器（与 dense.py 同一加载方式，local_files_only）。"""
    from medidiag.rag.dense import load_local_encoder
    return load_local_encoder(EMBEDDING_MODEL, EMBEDDING_REVISION, device="cuda", batch_size=32)


if __name__ == "__main__":
    main()
