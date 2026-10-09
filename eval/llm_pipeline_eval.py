"""真实模型接入后的端到端评测：LLM 弃答、查询改写与重复性（v2 语料）。

回答三个问题，全部可用录像带零费用回放：

1. 现网只有“有正分命中就作答”，把 top3 证据交给应用里的受约束生成（``MimoGroundedWorkflowProvider``）后，
   模型自己的弃答能否降低“不可回答问题仍被作答”的比例？代价是多少可回答问题被拒？
2. 预算受限的查询改写（每个问题最多 1 次额外调用，不看语料）能否补上 BM25 在口语同义改写上的漏检？
3. 同一请求重复采样，模型的作答/弃答与选段是否稳定？

模式：``record`` 需要真实凭据并占用持久预算；``replay`` 只读录像带，绝不联网，未命中即失败。

用法::

    python -m eval.llm_pipeline_eval --mode record --max-calls 800
    python -m eval.llm_pipeline_eval --mode replay --conditions bm25 bm25_rewrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.retrieval_benchmark import (
    BGE_ZH_QUERY_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    Bm25Scorer,
    DenseScorer,
    EmbeddingCache,
    load_dataset,
    load_sentence_transformer,
    proportion,
    rank_chunks,
)
from medidiag.errors import MediDiagError
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport
from medidiag.llm.openai_compatible import OpenAICompatibleProvider
from medidiag.llm.profiles import get_provider_profile
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
from medidiag.workflow.query_rewrite import REWRITE_PROMPT, build_rewrite_request, parse_rewrite

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "llm-pipeline-eval-v1"
TOP_K = 3
RETRYABLE = {"PROVIDER_RATE_LIMITED", "PROVIDER_UNAVAILABLE", "LLM_TIMEOUT", "PROVIDER_NETWORK_ERROR"}
CONTRACT_VIOLATIONS = {"STRUCTURED_OUTPUT_INVALID", "PROVIDER_SCHEMA_INVALID"}


# ----------------------------------------------------------------- 统计

def binomial_two_sided(k: int, n: int) -> float:
    """H0: p=0.5 的精确双侧二项检验，用于“改写后新命中 vs 新丢失”的配对比较。"""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def call_with_retry(fn: Callable[[], Any], tries: int = 3, sleep: Callable[[float], None] = time.sleep) -> Any:
    for attempt in range(tries):
        try:
            return fn()
        except MediDiagError as exc:
            if exc.code not in RETRYABLE or attempt == tries - 1:
                raise
            sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


# ----------------------------------------------------------------- LLM 步骤

RETRY_BASE = 100  # 重采样用 trial 号 RETRY_BASE + t，与首次采样的录像互不覆盖
FEEDBACK_BASE = 200  # 带原因的第二次请求用 trial 号 FEEDBACK_BASE + t

def make_provider(transport: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(get_provider_profile("mimo_v25"), post=transport,
                                    max_retries=0, max_structured_retries=0)


def rewrite_query(provider: OpenAICompatibleProvider, question: str) -> dict[str, Any]:
    """返回 {"query": 改写句, "status": ok|invalid|error}；失败时调用方回退到原问题。"""
    request = build_rewrite_request(question)
    try:
        result = call_with_retry(lambda: provider.generate(request, timeout_s=45, idempotency_key="rw-" + hashlib.sha256(question.encode()).hexdigest()[:24]))
    except MediDiagError as exc:
        return {"query": "", "status": "invalid" if exc.code in CONTRACT_VIOLATIONS else "error", "code": exc.code}
    text = parse_rewrite(result.parsed_json)
    if text is None:
        return {"query": "", "status": "invalid", "code": "REWRITE_SHAPE"}
    return {"query": text, "status": "ok", "usage": result.usage}


def grounded_decision(provider: OpenAICompatibleProvider, question: str, ranked: Sequence[tuple[str, str]],
                      feedback: str | None = None) -> dict[str, Any]:
    """走应用的真实生成路径。decision: answered | abstained | invalid | error。"""
    app = MimoGroundedWorkflowProvider(rag_stage=None, llm=provider)  # type: ignore[arg-type]
    retrieval = {"chunks": [{"chunk_id": cid, "text": text} for cid, text in ranked]}
    try:
        response = call_with_retry(lambda: app.generate(question, retrieval, {}, feedback=feedback))
    except MediDiagError as exc:
        return {"decision": "invalid" if exc.code in CONTRACT_VIOLATIONS else "error", "code": exc.code, "detail": exc.detail, "selected": []}
    claims = response.payload["claims"]
    return {"decision": "answered" if claims else "abstained",
            "selected": [c["citation_chunk_ids"][0] for c in claims],
            "usage": response.metadata.get("usage", {})}


# ----------------------------------------------------------------- 指标

def generation_metrics(rows: Sequence[dict[str, Any]], condition: str, trial: int, *,
                       with_retry: bool | str = False) -> dict[str, Any]:
    answerable = [r for r in rows if r["gold"]]
    unanswerable = [r for r in rows if not r["gold"]]

    def outcome(row: dict[str, Any]) -> dict[str, Any]:
        first = row["conditions"][condition]["trials"][trial]
        # with_retry：首次违反契约的请求，以重采样那一次的结果为准（最多一次）
        key = "retry" if with_retry is True else with_retry
        return first[key] if key and key in first else first

    correct = sum(1 for r in answerable if outcome(r)["decision"] == "answered" and set(outcome(r)["selected"]) & set(r["gold"]))
    wrong = sum(1 for r in answerable if outcome(r)["decision"] == "answered" and not set(outcome(r)["selected"]) & set(r["gold"]))
    refused = sum(1 for r in answerable if outcome(r)["decision"] == "abstained")
    invalid = sum(1 for r in rows if outcome(r)["decision"] in {"invalid", "error"})
    false_answers = sum(1 for r in unanswerable if outcome(r)["decision"] == "answered")
    answered_total = correct + wrong + false_answers
    return {
        "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        # 不可回答却作答
        "false_answer_rate": proportion(false_answers, len(unanswerable)),
        # 可回答却主动弃答（不含契约违规）
        "over_abstention_rate": proportion(refused, len(answerable)),
        # 可回答、作答且选中了金标准片段
        "useful_answer_rate": proportion(correct, len(answerable)),
        # 可回答、作答，但选的片段都不是金标准（可能是等价的另一段，也可能是错的）
        "answered_without_gold_rate": proportion(wrong, len(answerable)),
        # 作答的回答里，选中金标准的比例
        "precision_when_answered": proportion(correct, answered_total),
        "contract_violations_or_errors": invalid,
        # 契约违规发生在哪类问题上：不可回答问题上的违规若未被拦截，本会是一次误答
        "contract_violations_by_kind": {
            "answerable": sum(1 for r in answerable if outcome(r)["decision"] in {"invalid", "error"}),
            "unanswerable": sum(1 for r in unanswerable if outcome(r)["decision"] in {"invalid", "error"}),
            "details": dict(Counter(outcome(r).get("detail", "") for r in rows if outcome(r)["decision"] in {"invalid", "error"})),
        },
        "false_answer_by_bucket": {
            bucket: proportion(
                sum(1 for r in unanswerable if r["bucket"] == bucket and outcome(r)["decision"] == "answered"),
                sum(1 for r in unanswerable if r["bucket"] == bucket))
            for bucket in ("near_miss", "uncovered", "out_of_scope")},
    }


def answer_f1(metrics: dict[str, Any]) -> dict[str, float | None]:
    """由 ``generation_metrics`` 的结果算出回答的精确率、召回率和 F1，不改动入库报告的字段。

    精确率 = 作答且选中金标准 / 全部作答（含不该答却答了的）；召回率 = 作答且选中金标准 / 可回答问题数。
    """
    precision = metrics["precision_when_answered"]["rate"]
    recall = metrics["useful_answer_rate"]["rate"]
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (
        0.0 if precision is not None or recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": None if f1 is None else round(f1, 4)}


def repeatability(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    decisions_agree = selections_agree = 0
    flips: list[str] = []
    for row in rows:
        trials = row["conditions"][condition]["trials"]
        kinds = {t["decision"] for t in trials}
        picks = {tuple(sorted(t["selected"])) for t in trials}
        decisions_agree += len(kinds) == 1
        selections_agree += len(picks) == 1
        if len(kinds) > 1:
            flips.append(row["sample_id"])
    n = len(rows)
    return {"trials": len(rows[0]["conditions"][condition]["trials"]) if rows else 0,
            "decision_all_agree": proportion(decisions_agree, n),
            "selection_all_agree": proportion(selections_agree, n), "flipped_samples": flips}


def retrieval_hits(retrieval: dict[str, list[dict[str, Any]]], condition: str, queries: Sequence[dict[str, Any]]) -> dict[str, bool]:
    return {q["sample_id"]: bool(set(q["gold_evidence_ids"]) & set(retrieval[condition][q["sample_id"]]))
            for q in queries if q["gold_evidence_ids"]}


def rewrite_effect(queries: Sequence[dict[str, Any]], retrieval: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = retrieval_hits(retrieval, "bm25", queries)
    new = retrieval_hits(retrieval, "bm25_rewrite", queries)
    out: dict[str, Any] = {}
    for label, predicate in [("all", lambda q: True), *[(b, lambda q, b=b: q["bucket"] == b)
                                                       for b in ("direct", "paraphrase", "multi")]]:
        for split in ("dev", "test", "both"):
            ids = [q["sample_id"] for q in queries if q["gold_evidence_ids"] and predicate(q)
                   and (split == "both" or q["split"] == split)]
            gained = sum(1 for i in ids if new[i] and not base[i])
            lost = sum(1 for i in ids if base[i] and not new[i])
            out[f"{label}/{split}"] = {
                "n": len(ids), "bm25_hit@3": proportion(sum(base[i] for i in ids), len(ids)),
                "rewrite_hit@3": proportion(sum(new[i] for i in ids), len(ids)),
                "gained": gained, "lost": lost, "sign_test_p": round(binomial_two_sided(min(gained, lost), gained + lost), 4)}
    return out


def rewrite_status(queries: Sequence[dict[str, Any]], rewrites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status = Counter(r["status"] for r in rewrites.values())
    return {"status": dict(status), "queries": len(queries)}


# ----------------------------------------------------------------- 编排

def build_retrievers(conditions: Sequence[str], dataset: Any, embedding_revision: str | None) -> dict[str, Callable[[dict[str, Any], str], list[tuple[str, float]]]]:
    chunk_ids = [c["chunk_id"] for c in dataset.chunks]
    bm25 = Bm25Scorer("bm25_bigram", "cjk_bigram")
    bm25.fit(dataset.chunks)
    retrievers: dict[str, Callable[[dict[str, Any], str], list[tuple[str, float]]]] = {
        "bm25": lambda q, text: rank_chunks(bm25.score(text), chunk_ids, True, TOP_K)}
    if "dense" in conditions:
        cache = EmbeddingCache(load_sentence_transformer(DEFAULT_EMBEDDING_MODEL, embedding_revision))
        dense = DenseScorer("dense", cache, BGE_ZH_QUERY_PREFIX)
        dense.fit(dataset.chunks)
        retrievers["dense"] = lambda q, text: rank_chunks(dense.score(text), chunk_ids, False, TOP_K)
    return retrievers


def run(args: argparse.Namespace, transports: dict[int, Any], dataset: Any) -> dict[str, Any]:
    conditions: list[str] = args.conditions
    chunk_text = {c["chunk_id"]: c["text"] for c in dataset.chunks}
    queries = list(dataset.queries)
    if args.limit:
        queries = [q for s in ("dev", "test") for q in [x for x in queries if x["split"] == s][: args.limit]]
    retrievers = build_retrievers(conditions, dataset, args.embedding_revision)
    providers = {trial: make_provider(t) for trial, t in transports.items()}
    retry_enabled = bool(getattr(args, "retry_on_violation", False))
    feedback_enabled = bool(getattr(args, "feedback_retry", False))

    started = time.perf_counter()
    rewrites: dict[str, dict[str, Any]] = {}
    if "bm25_rewrite" in conditions:
        with ThreadPoolExecutor(args.workers) as pool:
            for query, result in zip(queries, pool.map(lambda q: rewrite_query(providers[0], q["question"]), queries), strict=True):
                rewrites[query["sample_id"]] = result

    def retrieved(condition: str, query: dict[str, Any]) -> list[tuple[str, float]]:
        if condition == "bm25_rewrite":
            rewritten = rewrites[query["sample_id"]]["query"]
            return retrievers["bm25"](query, f"{query['question']} {rewritten}".strip())
        return retrievers[condition](query, query["question"])

    retrieval: dict[str, dict[str, list[str]]] = {
        c: {q["sample_id"]: [cid for cid, _ in retrieved(c, q)] for q in queries} for c in conditions}
    gen_queries = [q for q in queries if q["split"] == "test"]
    trials_for = {c: (args.repeat_trials if c == "bm25" else 1) for c in conditions}

    jobs = [(c, q, t) for c in conditions for q in gen_queries for t in range(trials_for[c])]

    def execute(job: tuple[str, dict[str, Any], int]) -> dict[str, Any]:
        condition, query, trial = job
        ranked = [(cid, chunk_text[cid]) for cid in retrieval[condition][query["sample_id"]]]
        if not ranked:  # 与应用一致：空检索不调用模型
            return {"decision": "abstained", "selected": [], "model_called": False}
        first = {**grounded_decision(providers[trial], query["question"], ranked), "model_called": True}
        if retry_enabled and first["decision"] == "invalid":  # 只对契约违规重采样一次；网络类错误不算
            first["retry"] = {**grounded_decision(providers[RETRY_BASE + trial], query["question"], ranked), "model_called": True}
        if feedback_enabled and first["decision"] == "invalid":  # 同上，但第二次请求带上被拒原因
            first["feedback_retry"] = {**grounded_decision(providers[FEEDBACK_BASE + trial], query["question"], ranked,
                                                           feedback=first.get("detail")), "model_called": True}
        return first

    with ThreadPoolExecutor(args.workers) as pool:
        outcomes = list(pool.map(execute, jobs))
    rows = []
    by_job = {(c, q["sample_id"], t): o for (c, q, t), o in zip(jobs, outcomes, strict=True)}
    for query in gen_queries:
        rows.append({
            "sample_id": query["sample_id"], "bucket": query["bucket"], "gold": query["gold_evidence_ids"],
            "conditions": {c: {"retrieved": retrieval[c][query["sample_id"]],
                               "trials": [by_job[(c, query["sample_id"], t)] for t in range(trials_for[c])]}
                           for c in conditions}})

    report: dict[str, Any] = {
        "generation": {c: {f"trial_{t}": generation_metrics(rows, c, t) for t in range(trials_for[c])} for c in conditions},
        "repeatability": {c: repeatability(rows, c) for c in conditions if trials_for[c] > 1},
        "per_query": rows, "wall_seconds": round(time.perf_counter() - started, 1),
    }
    if retry_enabled:
        report["generation_with_retry"] = {
            c: {f"trial_{t}": generation_metrics(rows, c, t, with_retry=True) for t in range(trials_for[c])} for c in conditions}
        report["retry_summary"] = {
            c: {"retried": sum(1 for r in rows for o in r["conditions"][c]["trials"] if "retry" in o),
                "recovered": sum(1 for r in rows for o in r["conditions"][c]["trials"]
                                 if "retry" in o and o["retry"]["decision"] in {"answered", "abstained"}),
                "still_invalid": sum(1 for r in rows for o in r["conditions"][c]["trials"]
                                     if "retry" in o and o["retry"]["decision"] in {"invalid", "error"})}
            for c in conditions}
    if feedback_enabled:
        report["generation_with_feedback_retry"] = {
            c: {f"trial_{t}": generation_metrics(rows, c, t, with_retry="feedback_retry") for t in range(trials_for[c])}
            for c in conditions}
        report["feedback_retry_summary"] = {
            c: {"retried": sum(1 for r in rows for o in r["conditions"][c]["trials"] if "feedback_retry" in o),
                "recovered": sum(1 for r in rows for o in r["conditions"][c]["trials"]
                                 if "feedback_retry" in o and o["feedback_retry"]["decision"] in {"answered", "abstained"}),
                "still_invalid": sum(1 for r in rows for o in r["conditions"][c]["trials"]
                                     if "feedback_retry" in o and o["feedback_retry"]["decision"] in {"invalid", "error"})}
            for c in conditions}
    if "bm25_rewrite" in conditions:
        report["rewrite"] = {
            "effect_on_retrieval": rewrite_effect(queries, retrieval),
            "rewrite_status": rewrite_status(queries, rewrites),
            "examples": [{"sample_id": q["sample_id"], "bucket": q["bucket"], "question": q["question"],
                          "rewrite": rewrites[q["sample_id"]]["query"]}
                         for q in queries if q["bucket"] == "paraphrase"][:12],
            "rewrites": {k: v["query"] for k, v in rewrites.items()},
        }
    return report


def sha256_of(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "examples/public_health_v2")
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/llm-pipeline-v2.jsonl")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/llm-pipeline-ledger.db")
    parser.add_argument("--max-calls", type=int, default=800, help="record 模式的持久调用上限（含失败与重试）")
    parser.add_argument("--conditions", nargs="+", default=["bm25", "bm25_rewrite", "dense"],
                        choices=["bm25", "bm25_rewrite", "dense"])
    parser.add_argument("--repeat-trials", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retry-cassette", type=Path, default=ROOT / "eval/cassettes/llm-pipeline-v2-retry.jsonl",
                        help="重采样录像带，与主录像带分开，不改变已入库录像带的指纹")
    parser.add_argument("--retry-on-violation", action="store_true",
                        help="首次违反输出契约时重采样一次（用 trial 号 100+t，需要额外调用）")
    parser.add_argument("--feedback-cassette", type=Path,
                        default=ROOT / "eval/cassettes/llm-pipeline-v2-feedback.jsonl",
                        help="带原因的第二次请求的录像带，与前两个分开")
    parser.add_argument("--feedback-retry", action="store_true",
                        help="首次违反输出契约时，带上被拒原因再请求一次（trial 号 200+t，需要额外调用）")
    parser.add_argument("--limit", type=int, default=0, help="冒烟：每个 split 只取前 N 条查询")
    parser.add_argument("--embedding-revision", default="7999e1d3359715c523056ef9478215996d62a620")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_dir)
    cassette = Cassette(args.cassette)
    trials = max(args.repeat_trials, 1)
    transports: dict[int, Any]
    spent: dict[str, Any] | None = None
    if args.mode == "record":
        transports = {t: RecordingTransport(cassette, args.ledger, max_calls=args.max_calls, trial=t) for t in range(trials)}
        if args.retry_on_violation:
            retry_cassette = Cassette(args.retry_cassette)
            transports.update({RETRY_BASE + t: RecordingTransport(retry_cassette, args.ledger, max_calls=args.max_calls, trial=RETRY_BASE + t)
                               for t in range(trials)})
        if args.feedback_retry:
            feedback_cassette = Cassette(args.feedback_cassette)
            transports.update({FEEDBACK_BASE + t: RecordingTransport(feedback_cassette, args.ledger, max_calls=args.max_calls,
                                                                     trial=FEEDBACK_BASE + t) for t in range(trials)})
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        transports = {t: ReplayTransport(cassette, trial=t) for t in range(trials)}
        if args.retry_on_violation:
            retry_cassette = Cassette(args.retry_cassette)
            transports.update({RETRY_BASE + t: ReplayTransport(retry_cassette, trial=RETRY_BASE + t) for t in range(trials)})
        if args.feedback_retry:
            feedback_cassette = Cassette(args.feedback_cassette)
            transports.update({FEEDBACK_BASE + t: ReplayTransport(feedback_cassette, trial=FEEDBACK_BASE + t)
                               for t in range(trials)})
    report = run(args, transports, dataset)
    if args.mode == "record":
        spent = transports[0].spent()
    report = {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
        "config": {"model": "mimo-v2.5", "mode": args.mode, "conditions": args.conditions, "top_k": TOP_K,
                   "repeat_trials_bm25": args.repeat_trials, "limit": args.limit or None,
                   "rewrite_prompt": REWRITE_PROMPT, "temperature": "供应商默认（该 profile 不支持设置）",
                   "dataset_sha256": dataset.fingerprints, "cassette_entries": len(cassette),
                   "cassette_sha256": sha256_of(args.cassette),
                   "retry_on_violation": bool(args.retry_on_violation),
                   "retry_cassette_sha256": sha256_of(args.retry_cassette) if args.retry_on_violation else None,
                   "feedback_retry": bool(args.feedback_retry),
                   "feedback_cassette_sha256": sha256_of(args.feedback_cassette) if args.feedback_retry else None},
        "spend": spent, **report}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    for cond, trials_report in report["generation"].items():
        for name, metrics in trials_report.items():
            print(f"{cond:13s}{name:8s} false_answer={metrics['false_answer_rate']['rate']} "
                  f"over_abstain={metrics['over_abstention_rate']['rate']} useful={metrics['useful_answer_rate']['rate']} "
                  f"violations={metrics['contract_violations_or_errors']}")
    if spent:
        print("花费：", spent)


if __name__ == "__main__":
    main()
