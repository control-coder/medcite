"""检索层对照矩阵：系统 × 指标，固定在 test 集，并给出成对检验。

两类系统：

- 有完整排名的系统（BM25 双字、向量、混合）：由 ``eval.retrieval_benchmark`` 的打分器
  在 test 集上重新排序得到前 10 名，可算 Hit@1/3、Recall@3、MRR@3/10、nDCG@10。
- 只有前 3 条的系统（查询改写、检索智能体）：从模型调用记录对应的评测报告里取每条查询最终
  交给生成阶段的 3 个片段，只能算 Hit@3、Recall@3、MRR@3，不能和前者的 MRR@10、nDCG@10 比。

成对检验只用同一批 56 条可回答的 test 查询：均值差的自助法区间，加上精确符号检验
（只看两边不同的查询，双侧）。不联网，不调用任何在线模型；向量检索读本地缓存的模型。

用法::

    python -m eval.retrieval_matrix --llm-report artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from eval import retrieval_benchmark as rb

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "retrieval-matrix-v1"
# 完整排名系统的列；只有前 3 条的系统只填前三个
FULL_METRICS = ("hit@1", "hit@3", "recall@3", "mrr@3", "mrr@10", "ndcg@10")
POOL_METRICS = ("hit@3", "recall@3", "mrr@3")
# (系统甲, 系统乙, 指标)：比较“甲 − 乙”
PAIRS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("hybrid", "dense", ("hit@3", "mrr@10", "ndcg@10")),
    ("dense", "bm25_bigram", ("hit@3", "mrr@10")),
    ("bm25_rewrite", "bm25_bigram", ("hit@3", "recall@3", "mrr@3")),
    ("bm25_agent", "bm25_bigram", ("hit@3", "recall@3", "mrr@3")),
    ("bm25_agent", "bm25_rewrite", ("hit@3", "mrr@3")),
    ("dense_agent", "dense", ("hit@3", "mrr@3")),
    ("dense", "bm25_agent", ("hit@3", "mrr@3")),
)


def query_metrics(gold_ranks: Sequence[int], gold_count: int) -> dict[str, float]:
    """一条可回答查询的各指标。``gold_ranks`` 是每个正确片段在结果中的名次（从 1 起）。"""
    ranks = sorted(gold_ranks)
    first = ranks[0] if ranks else None
    record = {"gold": ["_"] * gold_count, "gold_ranks": ranks}
    return {
        "hit@1": float(first == 1),
        "hit@3": float(first is not None and first <= 3),
        "recall@3": rb._recall_at(record, 3),
        "mrr@3": 1 / first if first is not None and first <= 3 else 0.0,
        "mrr@10": 1 / first if first is not None and first <= 10 else 0.0,
        "ndcg@10": rb._ndcg_at(record, 10),
    }


def sign_test_p(wins: int, losses: int) -> float:
    """精确符号检验（双侧）：两边不同的查询里，甲胜出次数是否偏离一半。"""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(diffs: Sequence[float], resamples: int = 5000, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)]


def compare(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """同一批查询上的成对比较：甲 − 乙。"""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    wins = sum(d > 1e-12 for d in diffs)
    losses = sum(d < -1e-12 for d in diffs)
    low, high = paired_bootstrap_ci(diffs)
    return {
        "n": len(diffs), "a_mean": round(statistics.mean(a), 4), "b_mean": round(statistics.mean(b), 4),
        "diff": round(statistics.mean(diffs), 4), "diff_ci95": [low, high],
        "a_better": wins, "b_better": losses, "tied": len(diffs) - wins - losses,
        "sign_test_p": round(sign_test_p(wins, losses), 4),
    }


def summarize(per_query: Sequence[dict[str, float]], metrics: Sequence[str]) -> dict[str, Any]:
    """各指标的均值与自助法区间（Hit 类用 Wilson 区间）。"""
    out: dict[str, Any] = {}
    for name in metrics:
        values = [row[name] for row in per_query]
        if name.startswith("hit@"):
            out[name] = rb.proportion(int(sum(values)), len(values))
        else:
            out[name] = {"mean": round(statistics.mean(values), 4), "ci95": rb.bootstrap_mean_ci(values)}
    return out


def ranked_records(
    scorer_factory: Callable[[], rb.Scorer], dataset: rb.Dataset, split: str
) -> dict[str, dict[str, float]]:
    """完整排名系统：对 split 里每条可回答查询重新排序，返回 {sample_id: 指标}。"""
    queries = [q for q in dataset.queries if q["split"] == split]
    records = rb.run_queries(scorer_factory(), dataset.chunks, queries)
    return {r["sample_id"]: query_metrics(r["gold_ranks"], len(r["gold"])) for r in records if r["gold"]}


def pool_records(report: dict[str, Any], condition: str) -> dict[str, dict[str, float]]:
    """只有前 3 条的系统：从评测报告的 per_query 取最终交给生成阶段的片段。"""
    out = {}
    for row in report["per_query"]:
        if not row["gold"]:
            continue
        retrieved = row["conditions"][condition]["retrieved"]
        ranks = [i for i, chunk_id in enumerate(retrieved, 1) if chunk_id in set(row["gold"])]
        out[row["sample_id"]] = query_metrics(ranks, len(row["gold"]))
    return out


def build_matrix(
    dataset: rb.Dataset, llm_report: dict[str, Any], encoder: Callable[[list[str]], np.ndarray],
    alpha: float, split: str = "test",
) -> dict[str, Any]:
    cache = rb.EmbeddingCache(encoder)
    registry = rb.build_registry(cache, alpha)
    systems: dict[str, dict[str, dict[str, float]]] = {
        name: ranked_records(registry[name], dataset, split) for name in ("bm25_bigram", "dense", "hybrid")}
    for name in ("bm25_rewrite", "bm25_agent", "dense_agent"):
        systems[name] = pool_records(llm_report, name)

    ids = sorted(systems["dense"])
    for name, rows in systems.items():
        if sorted(rows) != ids:
            raise ValueError(f"{name} 的查询集合与 dense 不一致")

    # 一致性检查：评测报告里的 bm25 / dense 前 3 条应与这里重新排序的前 3 名同一命中结果
    cassette_bm25 = pool_records(llm_report, "bm25")
    cassette_dense = pool_records(llm_report, "dense")
    consistency = {
        "bm25_hit@3_same": sum(cassette_bm25[i]["hit@3"] == systems["bm25_bigram"][i]["hit@3"] for i in ids),
        "dense_hit@3_same": sum(cassette_dense[i]["hit@3"] == systems["dense"][i]["hit@3"] for i in ids),
        "queries": len(ids),
    }

    matrix = {}
    for name, rows in systems.items():
        metrics = FULL_METRICS if name in {"bm25_bigram", "dense", "hybrid"} else POOL_METRICS
        matrix[name] = {"ranking": "完整排名（前 10）" if metrics is FULL_METRICS else "最终 3 个片段",
                        **summarize([rows[i] for i in ids], metrics)}

    comparisons = []
    for left, right, names in PAIRS:
        for metric in names:
            comparisons.append({"a": left, "b": right, "metric": metric,
                                **compare([systems[left][i][metric] for i in ids],
                                          [systems[right][i][metric] for i in ids])})
    return {"split": split, "answerable_n": len(ids), "hybrid_alpha": alpha, "matrix": matrix,
            "paired_comparisons": comparisons, "consistency_with_llm_report": consistency}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "examples/public_health_v3")
    parser.add_argument("--llm-report", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path,
                        default=ROOT / "artifacts/reports/retrieval/retrieval-benchmark-v3-20261010.json",
                        help="取其中在 dev 上选定的混合权重 alpha")
    parser.add_argument("--embedding-model", default=rb.DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = rb.load_dataset(args.dataset_dir)
    llm_report = json.loads(args.llm_report.read_text(encoding="utf-8"))
    alpha = float(json.loads(args.retrieval_report.read_text(encoding="utf-8"))["config"]["hybrid_alpha"])
    encoder = rb.load_sentence_transformer(args.embedding_model, args.embedding_revision)
    result = build_matrix(dataset, llm_report, encoder, alpha)
    report = {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {"dir": str(args.dataset_dir), "sha256": dataset.fingerprints},
        "sources": {"llm_report": args.llm_report.name, "retrieval_report": args.retrieval_report.name,
                    "embedding_model": args.embedding_model},
        "external_model_calls": 0, **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, row in result["matrix"].items():
        cells = {k: (v["rate"] if "rate" in v else v["mean"]) for k, v in row.items() if isinstance(v, dict)}
        print(f"{name:14s}", {k: round(v, 4) for k, v in cells.items()})
    for c in result["paired_comparisons"]:
        print(f"{c['a']} − {c['b']} {c['metric']}: {c['diff']:+.4f} {c['diff_ci95']} "
              f"胜{c['a_better']}/负{c['b_better']}/平{c['tied']} p={c['sign_test_p']}")
    print("一致性:", result["consistency_with_llm_report"])
    print("报告：", args.out)


if __name__ == "__main__":
    main()
