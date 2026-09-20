"""固定模拟检索对照；只测软件路径，不计算医学准确率，也不下载模型。"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.rag.retrieval import Retriever
from medidiag.schemas import KnowledgeChunk

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "rag_regression"
WEIGHTS = {"w1_bm25": 1.0, "w2_embedding": 0.0, "w3_evidence_level": 0.0, "w4_term_overlap": 0.0}


def retrieval_metrics(retrieved: list[str], relevant: list[str], k: int) -> dict:
    """Hit 为至少命中一次，Recall 为命中相关条目占比；空金标不混入均值。"""
    expected = set(relevant)
    found = len(set(retrieved[:k]) & expected)
    return {"hit": int(found > 0) if expected else None,
            "recall": found / len(expected) if expected else None}


def make_retriever() -> Retriever:
    chunks = [KnowledgeChunk(**item) for item in json.loads((FIXTURE / "chunks.json").read_text(encoding="utf-8"))]
    return Retriever(chunks, weights=WEIGHTS, evidence_level_scores={"level_5_other": 0.0},
                     embedding_model="disabled", rerank_model="disabled", device="cpu",
                     normalizer=TerminologyNormalizer())


def run_regression() -> dict:
    cases = json.loads((FIXTURE / "queries.json").read_text(encoding="utf-8"))
    runs = {}
    for name, normalize in (("bm25", False), ("bm25_term_normalized", True)):
        retriever = make_retriever()
        config = {"use_bm25": True, "use_embedding": False, "use_rerank": False,
                  "use_evidence_weighting": False, "use_term_normalization": normalize}
        started = time.perf_counter()
        retriever.build_index(use_bm25=True, use_embedding=False)
        build_ms = (time.perf_counter() - started) * 1000
        records = []
        for case in cases:
            started = time.perf_counter()
            results = retriever.search(case["question"], top_k=3, experiment_config=config)
            latency_ms = (time.perf_counter() - started) * 1000
            ids = [item.chunk_id for item in results]
            records.append({**case, "retrieved_ids": ids, "latency_ms": round(latency_ms, 4),
                            "at_1": retrieval_metrics(ids, case["relevant_chunk_ids"], 1),
                            "at_3": retrieval_metrics(ids, case["relevant_chunk_ids"], 3)})
        positive = [item for item in records if item["relevant_chunk_ids"]]
        metrics = {f"{metric}@{k}": statistics.mean(item[f"at_{k}"][metric] for item in positive)
                   for k in (1, 3) for metric in ("hit", "recall")}
        runs[name] = {"config": config, "index_build_ms": round(build_ms, 4), "metrics": metrics,
                      "median_query_ms": statistics.median(item["latency_ms"] for item in records),
                      "empty_results": sum(not item["retrieved_ids"] for item in records),
                      "miss_ids_at_3": [item["query_id"] for item in positive if not item["at_3"]["hit"]],
                      "queries": records}
    return {"schema_version": "application-rag-regression-v1", "generated_at": datetime.now(UTC).isoformat(),
            "dataset": "examples/rag_regression", "synthetic": True, "query_count": len(cases),
            "positive_query_count": sum(bool(item["relevant_chunk_ids"]) for item in cases),
            "weights": WEIGHTS, "top_k": 3, "tokenizer": "lowercase_whitespace",
            "versions": {"python": platform.python_version(), "rank-bm25": version("rank-bm25"), "numpy": version("numpy")},
            "external_model_calls": 0, "external_model_cost_usd": 0,
            "limitations": ["模拟主题词匹配，无医学事实或临床金标。", "查询刻意覆盖词典别名，不代表真实分布。",
                            "中文发热未被英文词典归一化，保留失败样例。", "单次进程内计时；不含初始化、网络或生成，不能作性能宣传。",
                            "空相关集合单独计数，不纳入 Hit/Recall 均值；不运行向量与重排。"],
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "implementation" / "rag-regression.json")
    args = parser.parse_args()
    report = run_regression()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, run in report["runs"].items():
        print(name, run["metrics"], "未命中：", run["miss_ids_at_3"])
    print("报告：", args.output)


if __name__ == "__main__":
    main()
