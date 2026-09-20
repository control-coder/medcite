"""固定 WHO 中文短引的两组词法对照；不联网、不调参、不覆盖历史报告。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.schemas import read_jsonl
from medidiag.workflow.application import load_application_config

ROOT = Path(__file__).resolve().parents[1]


def metrics(ids: list[str], gold: list[str], k: int) -> dict:
    hits = set(ids[:k]) & set(gold)
    return {"hit": int(bool(hits)) if gold else None,
            "recall": len(hits) / len(set(gold)) if gold else None}


def run_regression() -> dict:
    config = load_application_config(ROOT / "configs/application.yaml")
    cases = read_jsonl(ROOT / config["dataset"]["rag_eval_set_path"])
    runs = {}
    for tokenizer in ("whitespace", "cjk_bigram"):
        current = copy.deepcopy(config)
        current["retrieval"]["bm25_tokenizer"] = tokenizer
        started = time.perf_counter()
        rag = RuntimeMedicalRAG.from_config(current, root=ROOT)
        build_ms = (time.perf_counter() - started) * 1000
        records = []
        for case in cases:
            started = time.perf_counter()
            query = rag.normalize(case["question"])["normalized_query"]
            results = rag.retriever.search(query, top_k=3, experiment_config=rag.experiment_config)
            elapsed = (time.perf_counter() - started) * 1000
            ids = [item.chunk_id for item in results]
            records.append({**case, "retrieved_ids": ids,
                            "ranking": [{"chunk_id": item.chunk_id, "score": item.final_score} for item in results],
                            "latency_ms": round(elapsed, 4),
                            "at_1": metrics(ids, case["gold_evidence_ids"], 1),
                            "at_3": metrics(ids, case["gold_evidence_ids"], 3)})
        splits = {}
        for split in ("dev", "check"):
            subset = [r for r in records if r["split"] == split]
            positive = [r for r in subset if r["gold_evidence_ids"]]
            empty = [r for r in subset if not r["gold_evidence_ids"]]
            splits[split] = {
                "positive_count": len(positive),
                "metrics": {f"{metric}@{k}": statistics.mean(r[f"at_{k}"][metric] for r in positive)
                            for k in (1, 3) for metric in ("hit", "recall")},
                "no_evidence_count": len(empty),
                "correct_empty_count": sum(not r["retrieved_ids"] for r in empty),
                "false_hit_ids": [r["sample_id"] for r in empty if r["retrieved_ids"]],
                "miss_ids_at_3": [r["sample_id"] for r in positive if not r["at_3"]["hit"]],
                "median_query_ms": statistics.median(r["latency_ms"] for r in subset),
            }
        runs[tokenizer] = {"splits": splits, "queries": records, "build_ms": round(build_ms, 4),
                           "corpus_hash": rag.corpus_hash, "config_hash": rag.retrieval_config_hash}
    return {"schema_version": "public-health-rag-regression-v1", "generated_at": datetime.now(UTC).isoformat(),
            "dataset": config["dataset"], "query_count": len(cases),
            "query_sha256": hashlib.sha256((ROOT / config["dataset"]["rag_eval_set_path"]).read_bytes()).hexdigest(),
            "top_k": 3, "weights": config["retrieval"]["weights"], "runs": runs,
            "versions": {"python": platform.python_version(), "rank-bm25": version("rank-bm25"), "numpy": version("numpy")},
            "external_model_calls": 0,
            "limitations": ["8 篇 WHO 中文科普页面的短引，16 条 AI 辅助工程查询，非医学金标或盲测。",
                            "只改变 BM25 切分规则；check 结果不用于调参。",
                            "字面相关不等于能回答；无证据参考上的误命中单独保留。",
                            "单进程微型检索耗时不含生成/网络，不作吞吐、P95 或临床效果宣传。"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/reports/application" /
                        ("public-rag-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"))
    args = parser.parse_args()
    report = run_regression()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 使用排他写入，避免误覆盖先前固定数据上的证据。
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, run in report["runs"].items():
        print(name, json.dumps(run["splits"], ensure_ascii=False))
    print("报告：", args.output)


if __name__ == "__main__":
    main()
