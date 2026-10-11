"""由失败归因报告生成回归集 ``examples/rag_regression/cases.jsonl``。

每个失败过的查询一行：问题、正确片段、哪些条件在哪一类原因上失败，以及应用 BM25 检索器此刻的前 3 个片段。
只读本地文件，不联网；已存在时不覆盖。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from medidiag.rag.retrieval import Bm25Retriever
from medidiag.schemas import KnowledgeChunk, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "examples/public_health_v3"


def build_cases(attribution: dict, queries: dict[str, dict], retriever: Bm25Retriever) -> list[dict]:
    failing: dict[str, list[dict]] = defaultdict(list)
    for item in attribution["failures"]:
        if item["trial"] == 0:  # 每个条件取第 0 次采样，与归因表同口径
            failing[item["sample_id"]].append({"condition": item["condition"], "label": item["label"]})
    cases = []
    for sample_id in sorted(failing):
        query = queries[sample_id]
        top3 = [hit.chunk_id for hit in retriever.search(query["question"], top_k=3)]
        gold = query["gold_evidence_ids"]
        cases.append({
            "sample_id": sample_id, "bucket": query["bucket"], "question": query["question"],
            "gold_evidence_ids": gold, "failures": failing[sample_id],
            "bm25_top3": top3, "bm25_gold_in_top3": sorted(set(gold) & set(top3)),
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "examples/rag_regression/cases.jsonl")
    args = parser.parse_args()
    attribution = json.loads(args.attribution.read_text(encoding="utf-8"))
    queries = {q["sample_id"]: q for q in read_jsonl(DATASET / "queries.jsonl")}
    retriever = Bm25Retriever([KnowledgeChunk(**row) for row in read_jsonl(DATASET / "chunks.jsonl")], "cjk_bigram")
    retriever.build_index()
    cases = build_cases(attribution, queries, retriever)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:  # 排他写入，不覆盖已有回归集
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"写入 {len(cases)} 条：{args.out}")


if __name__ == "__main__":
    main()
