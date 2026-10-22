"""拒答取舍曲线：在已入库的端到端结果之前加一道“证据分数门槛”，看误答与过度拒答如何随阈值 τ 变化。

现有每个条件只有一种拒答行为（空检索不调用模型、模型自己判断证据不足）。这里不重新调用模型：
对每条查询，在本地重算门槛会看到的证据分数，分数低于 τ 就直接拒答，否则沿用报告里记录的流水线结果
（带原因重试一次之后的最终结果，与 docs/evaluation.md 第五节的口径相同）。τ = −∞ 即现有系统。

门槛分数（两个候选，只在 v3 dev 上按 AUROC 选一个）：

- ``rerank_top1``：交给生成的第 1 段证据与原问题的交叉编码器分数（``bge-reranker-v2-m3``，与流水线重排
  用的是同一个模型、同一个问题，所以就是重排时排第 1 的那个分数）。它直接看问题和最终证据，不受检索
  Agent 换了几次说法的影响。
- ``dense_top1``：原问题与交给生成的 3 段之间的最大向量余弦（``bge-small-zh-v1.5``，带检索前缀）。

证据的重建：报告的 ``per_query`` 只有 test 的最终 3 段，没有分数，也没有 dev。检索 Agent 每次检索的语句
记录在 ``agent.<条件>.traces`` 里（dev 与 test 都有），所以按同一检索代码重做每次检索（各取前 N 段），
并集按原问题重排取前 3，重排分数同时得到。test 上重建的 3 段必须与报告记录逐条一致，否则直接报错。

选择规则只用 v3 dev（dev 没有生成结果，所以只能用分数本身选）：

1. 门槛分数：dev 上“可回答 vs 不可回答”的 AUROC 较大者，并列取 ``rerank_top1``；
2. 两个阈值：dev 可回答问题里被门槛挡住的比例不超过 5% / 10% 时可取的最大 τ。
3. 另给“只用分数”的对照（分数 ≥ τ 就作答，不看模型判断），τ 按 dev 上“有用作答率 − 误答率”最大选择。

test 与扩充测试集只在选定的分数与阈值下报告；完整曲线在 test 上是描述性的，不用于选阈值。

用法::

    python -m eval.refusal_curve --out artifacts/reports/llm/refusal-curve-v3-YYYYMMDD.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import retrieval_benchmark as rb
from eval.rerank_matrix import CachedScorer
from medidiag.rag.rerank import PairScorer, Reranker, load_local_reranker
from medidiag.workflow.retrieval_agent import POOL_SIZE, merge_hits

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "refusal-curve-v1"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
SCORES = ("rerank_top1", "dense_top1")
RETRY_KEY = "feedback_retry"
BUDGETS = (0.05, 0.10)  # dev 可回答问题被门槛挡住的比例上限
BLOCKED = {"invalid", "error"}


@dataclass(frozen=True)
class _Hit:
    chunk_id: str
    text: str


# ----------------------------------------------------------------- 证据重建与分数

def agent_queries(trace: dict[str, Any]) -> list[str]:
    return [s["query"] for s in trace["steps"] if s["action"] == "search"]


def merge_method(trace: dict[str, Any]) -> str:
    """不重排的条件没有合并步骤，按倒数排名融合处理；重排失败时轨迹里记为 rrf。"""
    return next((s["method"] for s in trace["steps"] if s["action"] == "merge"), "rrf")


def reconstruct(question: str, trace: dict[str, Any], search: Callable[[str, int], list[_Hit]],
                reranker: Reranker) -> list[tuple[str, float]]:
    """按轨迹重做检索 Agent 的各次检索，返回交给生成的前 3 段及其按原问题的重排分数（降序）。"""
    lists = [search(text, reranker.candidates) for text in agent_queries(trace)]
    if merge_method(trace) == "rerank":
        return [(hit.chunk_id, score) for hit, score in
                reranker.rerank(question, [h for hits in lists for h in hits], POOL_SIZE)]
    # 退回倒数排名融合的情况：顺序以融合为准，分数仍按原问题打（门槛看的是第 1 段）
    merged = merge_hits(lists, POOL_SIZE)
    scored = dict((hit.chunk_id, score) for hit, score in reranker.rerank(question, merged, POOL_SIZE))
    return [(hit.chunk_id, scored[hit.chunk_id]) for hit in merged]


def final_outcome(row: dict[str, Any], condition: str, retry_key: str | None = RETRY_KEY) -> dict[str, Any]:
    first = row["conditions"][condition]["trials"][0]
    return first[retry_key] if retry_key and retry_key in first else first


def build_items(dataset: rb.Dataset, report: dict[str, Any], condition: str, search: Callable[[str, int], list[_Hit]],
                reranker: Reranker, dense_cosine: Callable[[str, Sequence[str]], float]) -> list[dict[str, Any]]:
    """每条查询一项：证据、两种门槛分数，test 上另带记录的流水线结果。"""
    traces = report["agent"][condition]["traces"]
    rows = {r["sample_id"]: r for r in report["per_query"]}
    text = {c["chunk_id"]: c["text"] for c in dataset.chunks}
    items = []
    for query in dataset.queries:
        sid = query["sample_id"]
        evidence = reconstruct(query["question"], traces[sid], search, reranker)
        ids = [cid for cid, _ in evidence]
        item: dict[str, Any] = {
            "sample_id": sid, "split": query["split"], "bucket": query["bucket"], "gold": query["gold_evidence_ids"],
            "evidence": ids, "rerank_scores": [round(s, 6) for _, s in evidence],
            "rerank_top1": round(evidence[0][1], 6) if evidence else None,
            "dense_top1": round(dense_cosine(query["question"], [text[c] for c in ids]), 6) if ids else None,
        }
        if sid in rows:
            recorded = rows[sid]["conditions"][condition]["retrieved"]
            if recorded != ids:  # 重建必须与入库报告逐条一致，否则门槛看到的不是同一批证据
                raise ValueError(f"{sid} 重建的证据 {ids} 与报告记录 {recorded} 不一致")
            outcome = final_outcome(rows[sid], condition)
            item["decision"], item["selected"] = outcome["decision"], outcome.get("selected", [])
        items.append(item)
    return items


# ----------------------------------------------------------------- 曲线

def _score(item: dict[str, Any], score: str) -> float:
    value = item[score]
    return -math.inf if value is None else float(value)


def _tau_json(tau: float) -> float | str:
    return round(tau, 6) if math.isfinite(tau) else ("-inf" if tau < 0 else "+inf")


def gated_point(items: Sequence[dict[str, Any]], score: str, tau: float) -> dict[str, Any]:
    """分数 < τ 时直接拒答（不调用模型），否则沿用记录的流水线结果。"""
    answerable = [i for i in items if i["gold"]]
    unanswerable = [i for i in items if not i["gold"]]

    def decision(item: dict[str, Any]) -> str:
        return "abstained" if _score(item, score) < tau else str(item["decision"])

    def correct(item: dict[str, Any]) -> bool:
        return decision(item) == "answered" and bool(set(item["selected"]) & set(item["gold"]))

    useful = sum(correct(i) for i in answerable)
    without_gold = sum(1 for i in answerable if decision(i) == "answered" and not correct(i))
    false_answers = sum(1 for i in unanswerable if decision(i) == "answered")
    answered = useful + without_gold + false_answers
    return {
        "tau": _tau_json(tau),
        "gate_blocked": sum(1 for i in items if _score(i, score) < tau),
        "answered": answered,
        # 不可回答却作答
        "false_answers": rb.proportion(false_answers, len(unanswerable)),
        # 可回答却拒答（模型拒答或门槛挡住），输出被拦截的不计入，与第一节口径相同
        "over_refusals": rb.proportion(sum(1 for i in answerable if decision(i) == "abstained"), len(answerable)),
        # 可回答、作答，但引用的片段都不是正确片段
        "answered_without_gold": rb.proportion(without_gold, len(answerable)),
        "useful_answers": rb.proportion(useful, len(answerable)),
        "blocked_by_validation": sum(1 for i in items if decision(i) in BLOCKED),
        "answer_precision": rb.proportion(useful, answered),
    }


def score_only_point(items: Sequence[dict[str, Any]], score: str, tau: float) -> dict[str, Any]:
    """只用分数的对照：分数 ≥ τ 就作答，前 3 段含正确片段算有用作答，不看模型判断。"""
    answerable = [i for i in items if i["gold"]]
    unanswerable = [i for i in items if not i["gold"]]
    answered = [i for i in answerable if _score(i, score) >= tau]
    useful = sum(1 for i in answered if set(i["evidence"]) & set(i["gold"]))
    return {
        "tau": _tau_json(tau),
        "false_answers": rb.proportion(sum(1 for i in unanswerable if _score(i, score) >= tau), len(unanswerable)),
        "over_refusals": rb.proportion(len(answerable) - len(answered), len(answerable)),
        "answered_without_gold": rb.proportion(len(answered) - useful, len(answerable)),
        "useful_answers": rb.proportion(useful, len(answerable)),
    }


def thresholds(items: Sequence[dict[str, Any]], score: str) -> list[float]:
    """−∞（现有系统）、每个出现过的分数（分数等于 τ 的那条仍作答），以及 +∞（全部拒答）。"""
    values = sorted({_score(i, score) for i in items if math.isfinite(_score(i, score))})
    return [-math.inf, *values, math.inf]


def curve(items: Sequence[dict[str, Any]], score: str,
          point: Callable[[Sequence[dict[str, Any]], str, float], dict[str, Any]]) -> list[dict[str, Any]]:
    return [point(items, score, tau) for tau in thresholds(items, score)]


def score_auroc(items: Sequence[dict[str, Any]], score: str, *, require_hit: bool = False) -> float | None:
    """正例为可回答问题（``require_hit`` 时只取前 3 段含正确片段的），负例为不可回答问题。"""
    positives = [_score(i, score) for i in items if i["gold"] and (not require_hit or set(i["evidence"]) & set(i["gold"]))]
    negatives = [_score(i, score) for i in items if not i["gold"]]
    value = rb.auroc(positives, negatives)
    return None if value is None else round(value, 4)


def select_score(dev: Sequence[dict[str, Any]]) -> str:
    values = {s: score_auroc(dev, s) or 0.0 for s in SCORES}
    return max(SCORES, key=lambda s: (values[s], -SCORES.index(s)))


def select_budget_tau(dev: Sequence[dict[str, Any]], score: str, budget: float) -> float:
    """dev 可回答问题中分数 < τ 的条数不超过 ⌊budget × n⌋ 时可取的最大 τ（取相邻两个分数的中点）。"""
    values = sorted(_score(i, score) for i in dev if i["gold"])
    allowed = math.floor(budget * len(values))
    if allowed <= 0:
        return -math.inf
    if allowed >= len(values):
        return math.inf
    # 恰好挡住最低的 allowed 条：τ 落在第 allowed 条与第 allowed+1 条之间
    low, high = values[allowed - 1], values[allowed]
    return (low + high) / 2 if high > low else low  # 并列时不能多挡，退到并列值本身（它们仍作答）


def select_youden_tau(dev: Sequence[dict[str, Any]], score: str) -> float:
    """只用分数的对照：dev 上“有用作答率 − 误答率”最大，并列取更小的 τ（覆盖更大）。"""
    best_tau, best = -math.inf, -math.inf
    for tau in thresholds(dev, score):
        point = score_only_point(dev, score, tau)
        utility = (point["useful_answers"]["rate"] or 0.0) - (point["false_answers"]["rate"] or 0.0)
        if utility > best + 1e-12:
            best_tau, best = tau, utility
    return best_tau


def evaluate(items: Sequence[dict[str, Any]], score: str, selected: dict[str, float]) -> dict[str, Any]:
    """一个测试集：现有系统、选定阈值下的结果、两种分数的 AUROC 与完整曲线。"""
    by_bucket = {b: sum(1 for i in items if i["bucket"] == b) for b in sorted({i["bucket"] for i in items})}
    return {
        "queries": len(items), "by_bucket": by_bucket,
        "auroc": {s: {"answerable_vs_unanswerable": score_auroc(items, s),
                      "evidence_hit_vs_unanswerable": score_auroc(items, s, require_hit=True)} for s in SCORES},
        "operating_point": gated_point(items, score, -math.inf),
        "at_dev_selected": {name: (score_only_point if name == "score_only_youden" else gated_point)(items, score, tau)
                            for name, tau in selected.items()},
        "gated_curve": curve(items, score, gated_point),
        "score_only_curve": curve(items, score, score_only_point),
    }


def build(dev_items: Sequence[dict[str, Any]], tests: dict[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    score = select_score(dev_items)
    selected = {f"gated_budget_{int(b * 100)}pct": select_budget_tau(dev_items, score, b) for b in BUDGETS}
    selected["score_only_youden"] = select_youden_tau(dev_items, score)
    return {
        "selection": {
            "split": "dev", "score": score,
            "dev_auroc": {s: {"answerable_vs_unanswerable": score_auroc(dev_items, s),
                              "evidence_hit_vs_unanswerable": score_auroc(dev_items, s, require_hit=True)} for s in SCORES},
            "thresholds": {k: _tau_json(v) for k, v in selected.items()},
            "dev_blocked_answerable": {k: sum(1 for i in dev_items if i["gold"] and _score(i, score) < v)
                                       for k, v in selected.items()},
            "dev_blocked_unanswerable": {k: sum(1 for i in dev_items if not i["gold"] and _score(i, score) < v)
                                         for k, v in selected.items()},
            "dev_counts": {"answerable": sum(1 for i in dev_items if i["gold"]),
                           "unanswerable": sum(1 for i in dev_items if not i["gold"])},
            "rule": __doc__.split("选择规则只用 v3 dev")[1].split("test 与扩充")[0].strip(),
        },
        "evaluations": {name: evaluate(items, score, selected) for name, items in tests.items()},
    }


# ----------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", default="dense_agent_rerank", choices=["dense_agent_rerank"])
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "examples/public_health_v3")
    parser.add_argument("--llm-report", type=Path, default=ROOT / "artifacts/reports/llm/llm-pipeline-v3-rerank-20261017.json",
                        help="提供 dev 与 test 的检索 Agent 轨迹，以及 test 的流水线结果")
    parser.add_argument("--ext-dataset-dir", type=Path, default=ROOT / "examples/public_health_v3_ext")
    parser.add_argument("--ext-report", type=Path, default=ROOT / "artifacts/reports/llm/llm-pipeline-v3-ext-20261017.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="重排模型设备；入库报告的重排在 GPU 上完成，用同一设备重建以逐条核对")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.llm_report.read_text(encoding="utf-8"))
    ext_report = json.loads(args.ext_report.read_text(encoding="utf-8"))
    selected = report["config"]["rerank"]
    if ext_report["config"]["rerank"]["candidates"] != selected["candidates"]:
        raise ValueError("两份报告的重排配置不同")
    dataset, ext = rb.load_dataset(args.dataset_dir), rb.load_dataset(args.ext_dataset_dir)
    if dataset.fingerprints != report["config"]["dataset_sha256"] or ext.fingerprints != ext_report["config"]["dataset_sha256"]:
        raise ValueError("数据集指纹与报告不一致")
    scorer: PairScorer = CachedScorer(load_local_reranker(selected["model"], selected["revision"], device=args.device))
    reranker = Reranker(scorer, model_name=selected["model"], model_revision=selected["revision"],
                        candidates=int(selected["candidates"][args.condition.removesuffix("_rerank")]))
    cache = rb.EmbeddingCache(rb.load_sentence_transformer(rb.DEFAULT_EMBEDDING_MODEL, EMBEDDING_REVISION))
    dense = rb.DenseScorer("dense", cache, rb.BGE_ZH_QUERY_PREFIX)
    dense.fit(dataset.chunks)  # 两个测试集共用同一份语料（chunks.jsonl 指纹相同）
    chunk_ids = [c["chunk_id"] for c in dataset.chunks]
    text = {c["chunk_id"]: c["text"] for c in dataset.chunks}

    def search(query: str, depth: int) -> list[_Hit]:
        return [_Hit(cid, text[cid]) for cid, _ in rb.rank_chunks(dense.score(query), chunk_ids, False, depth)]

    def dense_cosine(question: str, passages: Sequence[str]) -> float:
        q = cache.encode([rb.BGE_ZH_QUERY_PREFIX + question])[0]
        return float(max(cache.encode(list(passages)) @ q))

    items = build_items(dataset, report, args.condition, search, reranker, dense_cosine)
    ext_items = build_items(ext, ext_report, args.condition, search, reranker, dense_cosine)
    result = build([i for i in items if i["split"] == "dev"],
                   {"v3_test": [i for i in items if i["split"] == "test"], "v3_ext": ext_items})
    payload = {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
        "condition": args.condition, "external_model_calls": 0,
        "sources": {"llm_report": args.llm_report.name, "ext_report": args.ext_report.name,
                    "dataset_sha256": {"v3": dataset.fingerprints, "v3_ext": ext.fingerprints},
                    "rerank_model": selected["model"], "rerank_revision": selected["revision"],
                    "rerank_candidates": reranker.candidates, "rerank_device": args.device,
                    "embedding_model": rb.DEFAULT_EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
                    "pipeline_outcome": "带原因重试一次之后的最终结果（trial 0）"},
        "evidence_check": "test 与扩充测试集上重建的前 3 段与报告记录逐条一致（不一致时脚本报错）",
        **result,
        # 每条查询的分数与记录结果，测试可只靠本报告与原报告重算曲线，不需要模型
        "per_query": {"v3": items, "v3_ext": ext_items},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    sel = result["selection"]
    print("门槛分数：", sel["score"], "dev AUROC：", sel["dev_auroc"])
    print("dev 选定阈值：", sel["thresholds"], "dev 挡住可回答：", sel["dev_blocked_answerable"])
    for name, ev in result["evaluations"].items():
        print(name, "AUROC", ev["auroc"])
        for label, point in [("现有系统", ev["operating_point"]), *ev["at_dev_selected"].items()]:
            print(f"  {label}: τ={point['tau']} 误答 {point['false_answers']['k']}/{point['false_answers']['n']} "
                  f"过度拒答 {point['over_refusals']['k']}/{point['over_refusals']['n']} "
                  f"正确作答 {point['useful_answers']['k']} 作答无正确片段 {point['answered_without_gold']['k']}")
    print("报告：", args.out)


if __name__ == "__main__":
    main()
