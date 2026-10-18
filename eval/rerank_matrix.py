"""重排对照：交叉编码器重排对固定检索、查询改写和检索 Agent 的影响（检索层，不调用在线模型）。

检索 Agent 每一轮的检索语句取自已入库的端到端评测报告。重排只改变“多轮检索的候选怎么合并”，
模型每轮看到的仍是前 3 段，所以它的决策与报告里的轨迹相同，复用轨迹等价于重新运行（由
``tests/test_rerank_matrix.py`` 用离线假模型核对）。

选择规则只用 dev：

1. 对每个重排模型、每个系统，在候选深度网格里按 Hit@3、Recall@3、MRR@3 依次取最大，仍并列取更小的深度；
2. 两个模型各自在选定深度下，五个系统的 Hit@3、Recall@3、MRR@3 之和依次比较，取更大的，并列取更小的模型。

test 只在选定配置下计算一次，成对比较只用 test。

用法::

    python -m eval.rerank_matrix --llm-report artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json \\
        --out artifacts/reports/retrieval/rerank-matrix-v3-20261017.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import retrieval_benchmark as rb
from eval.retrieval_matrix import POOL_METRICS, compare, query_metrics, summarize
from medidiag.rag.rerank import Reranker, load_local_reranker
from medidiag.workflow.retrieval_agent import POOL_SIZE, merge_hits

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "rerank-matrix-v1"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
# 候选模型（按参数量从小到大）与固定版本
MODELS: dict[str, str] = {
    "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}
# 固定流程取前 N 段重排；检索 Agent 每次检索各取前 N 段，并集重排。3 表示不多取，只重排模型看过的段
FIXED_DEPTHS = (5, 10, 20)
AGENT_DEPTHS = (3, 5, 10)
RERANKED = ("bm25_rerank", "bm25_rewrite_rerank", "bm25_agent_rerank", "dense_rerank", "dense_agent_rerank")
BASELINES = ("bm25", "bm25_rewrite", "bm25_agent", "dense", "dense_agent")
# (系统甲, 系统乙)：比较“甲 − 乙”，只在 test 上
PAIRS: tuple[tuple[str, str], ...] = (
    ("bm25_agent_rerank", "bm25_agent"),
    ("bm25_agent_rerank", "bm25_rerank"),
    ("bm25_agent_rerank", "bm25_rewrite_rerank"),
    ("bm25_rewrite_rerank", "bm25_rewrite"),
    ("dense_rerank", "dense"),
    ("dense_agent_rerank", "dense_rerank"),
    ("dense_agent_rerank", "dense_agent"),
    ("bm25_agent_rerank", "dense"),
)


@dataclass(frozen=True)
class _Hit:
    chunk_id: str
    text: str


class CachedScorer:
    """同一对（问题，片段）只打一次分；网格搜索里同一候选会被多次重排。

    多线程共用时串行调用模型：各线程同时推理会各开一套计算线程，CPU 上互相争抢，反而更慢。
    """

    def __init__(self, scorer: Callable[[list[tuple[str, str]]], Sequence[float]]) -> None:
        self._scorer = scorer
        self._memo: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()
        self.pairs_scored = 0

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._lock:
            missing = list(dict.fromkeys(p for p in pairs if p not in self._memo))
            if missing:
                for pair, score in zip(missing, self._scorer(missing), strict=True):
                    self._memo[pair] = float(score)
                self.pairs_scored += len(missing)
            return [self._memo[p] for p in pairs]


class Systems:
    """十个系统在一条查询上的最终 3 段；检索 Agent 的检索语句来自轨迹。"""

    def __init__(self, dataset: rb.Dataset, llm_report: dict[str, Any], dense_encoder: Callable[[list[str]], Any]) -> None:
        self.text = {c["chunk_id"]: c["text"] for c in dataset.chunks}
        self.chunk_ids = [c["chunk_id"] for c in dataset.chunks]
        bm25 = rb.Bm25Scorer("bm25_bigram", "cjk_bigram")
        bm25.fit(dataset.chunks)
        dense = rb.DenseScorer("dense", rb.EmbeddingCache(dense_encoder), rb.BGE_ZH_QUERY_PREFIX)
        dense.fit(dataset.chunks)
        self.scorers: dict[str, rb.Scorer] = {"bm25": bm25, "dense": dense}
        self.rewrites: dict[str, str] = llm_report["rewrite"]["rewrites"]
        self.traces = {base: llm_report["agent"][base + "_agent"]["traces"] for base in ("bm25", "dense")}

    def search(self, base: str, text: str, depth: int) -> list[_Hit]:
        scorer = self.scorers[base]
        return [_Hit(cid, self.text[cid]) for cid, _ in rb.rank_chunks(scorer.score(text), self.chunk_ids,
                                                                       scorer.positive_only, depth)]

    def agent_queries(self, base: str, sample_id: str) -> list[str]:
        return [s["query"] for s in self.traces[base][sample_id]["steps"] if s["action"] == "search"]

    def final(self, system: str, query: dict[str, Any], reranker: Reranker | None = None) -> list[str]:
        question = query["question"]
        base = system.split("_")[0]
        if system.endswith("_agent") or system.endswith("_agent_rerank"):
            depth = reranker.candidates if reranker else POOL_SIZE
            lists = [self.search(base, text, depth) for text in self.agent_queries(base, query["sample_id"])]
            if reranker is None:
                return [hit.chunk_id for hit in merge_hits(lists, POOL_SIZE)]
            return [hit.chunk_id for hit, _ in reranker.rerank(question, [h for hits in lists for h in hits], POOL_SIZE)]
        text = f"{question} {self.rewrites[query['sample_id']]}".strip() if "_rewrite" in system else question
        if reranker is None:
            return [hit.chunk_id for hit in self.search(base, text, POOL_SIZE)]
        # 与应用一致：改写条件按“原问题 + 改写句”检索，也按它重排
        return [hit.chunk_id for hit, _ in reranker.rerank(text, self.search(base, text, reranker.candidates), POOL_SIZE)]


def records(systems: Systems, system: str, queries: Sequence[dict[str, Any]],
            reranker: Reranker | None = None) -> dict[str, dict[str, float]]:
    out = {}
    for query in queries:
        gold = set(query["gold_evidence_ids"])
        ranked = systems.final(system, query, reranker)
        out[query["sample_id"]] = query_metrics([i for i, cid in enumerate(ranked, 1) if cid in gold], len(gold))
    return out


def key(rows: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    return tuple(round(sum(r[m] for r in rows.values()), 6) for m in POOL_METRICS)  # type: ignore[return-value]


def select(systems: Systems, dev: Sequence[dict[str, Any]], scorers: dict[str, CachedScorer]) -> dict[str, Any]:
    grid: list[dict[str, Any]] = []
    best: dict[str, dict[str, int]] = {}
    totals: dict[str, tuple[float, float, float]] = {}
    for model, revision in MODELS.items():
        best[model] = {}
        total = [0.0, 0.0, 0.0]
        for system in RERANKED:
            depths = AGENT_DEPTHS if "_agent" in system else FIXED_DEPTHS
            scored = []
            for depth in depths:
                reranker = Reranker(scorers[model], model_name=model, model_revision=revision, candidates=depth)
                value = key(records(systems, system, dev, reranker))
                scored.append((value, depth))
                grid.append({"model": model, "system": system, "candidates": depth,
                             **dict(zip(POOL_METRICS, (round(v / len(dev), 4) for v in value), strict=True))})
            value, depth = max(scored, key=lambda item: (item[0], -item[1]))  # 指标依次取大，并列取小深度
            best[model][system.removesuffix("_rerank")] = depth
            total = [a + b for a, b in zip(total, value, strict=True)]
        totals[model] = tuple(total)  # type: ignore[assignment]
    order = list(MODELS)
    model = max(order, key=lambda name: (totals[name], -order.index(name)))
    return {"model": model, "revision": MODELS[model], "candidates": best[model],
            "dev_totals": {name: [round(v, 4) for v in totals[name]] for name in order},
            "dev_grid": grid, "rule": __doc__.split("选择规则只用 dev：")[1].split("test 只在")[0].strip()}


def build(dataset: rb.Dataset, llm_report: dict[str, Any], dense_encoder: Callable[[list[str]], Any],
          scorers: dict[str, CachedScorer]) -> dict[str, Any]:
    systems = Systems(dataset, llm_report, dense_encoder)
    split = {s: [q for q in dataset.queries if q["split"] == s and q["gold_evidence_ids"]] for s in ("dev", "test")}
    selected = select(systems, split["dev"], scorers)
    scorer = scorers[selected["model"]]
    rerankers = {name: Reranker(scorer, model_name=selected["model"], model_revision=selected["revision"],
                                candidates=selected["candidates"][name.removesuffix("_rerank")]) for name in RERANKED}
    per_split: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    latency: dict[str, list[float]] = {name: [] for name in RERANKED}
    for name, queries in split.items():
        per_split[name] = {s: records(systems, s, queries) for s in BASELINES}
        for system in RERANKED:
            started = time.perf_counter()
            per_split[name][system] = records(systems, system, queries, rerankers[system])
            latency[system].append((time.perf_counter() - started) / len(queries))
    ids = sorted(per_split["test"]["dense"])
    matrix = {name: {system: summarize([rows[i] for i in sorted(rows)], POOL_METRICS)
                     for system, rows in per_split[name].items()} for name in per_split}
    comparisons = [{"a": a, "b": b, "metric": metric,
                    **compare([per_split["test"][a][i][metric] for i in ids], [per_split["test"][b][i][metric] for i in ids])}
                   for a, b in PAIRS for metric in ("hit@3", "recall@3", "mrr@3")]
    return {"selected": {k: v for k, v in selected.items() if k != "dev_grid"}, "dev_grid": selected["dev_grid"],
            "matrix": matrix, "paired_comparisons_test": comparisons,
            "per_query_test": {system: {i: rows[i] for i in ids} for system, rows in per_split["test"].items()},
            # 打分已缓存，这里是本机 CPU 上包含检索在内的平均每条查询耗时，只用于量级比较
            "seconds_per_query_cpu": {k: round(statistics.mean(v), 3) for k, v in latency.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "examples/public_health_v3")
    parser.add_argument("--llm-report", type=Path, required=True, help="提供检索 Agent 轨迹与查询改写句")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = rb.load_dataset(args.dataset_dir)
    llm_report = json.loads(args.llm_report.read_text(encoding="utf-8"))
    encoder = rb.load_sentence_transformer(rb.DEFAULT_EMBEDDING_MODEL, EMBEDDING_REVISION)
    scorers = {model: CachedScorer(load_local_reranker(model, revision)) for model, revision in MODELS.items()}
    result = build(dataset, llm_report, encoder, scorers)
    report = {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {"dir": str(args.dataset_dir), "sha256": dataset.fingerprints},
        "sources": {"llm_report": args.llm_report.name, "embedding_model": rb.DEFAULT_EMBEDDING_MODEL,
                    "embedding_revision": EMBEDDING_REVISION, "rerank_models": MODELS},
        "external_model_calls": 0, "pairs_scored": {m: s.pairs_scored for m, s in scorers.items()}, **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("选定：", report["selected"])
    for split in ("dev", "test"):
        for system, row in result["matrix"][split].items():
            print(f"{split:5s}{system:21s}", {k: v.get("rate", v.get("mean")) for k, v in row.items()})
    for c in result["paired_comparisons_test"]:
        print(f"{c['a']} − {c['b']} {c['metric']}: {c['diff']:+.4f} {c['diff_ci95']} "
              f"胜{c['a_better']}/负{c['b_better']}/平{c['tied']} p={c['sign_test_p']}")
    print("报告：", args.out)


if __name__ == "__main__":
    main()
