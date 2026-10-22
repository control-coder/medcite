"""多轮追问评测：同一条追问分别按四种方式变成检索与生成的问题，比较检索命中、正确作答与拒答。

流水线固定为推荐配置（向量检索 + 检索 Agent + 重排，``configs/application_dense_agent_rerank.yaml``），
带原因重试一次，模型 ``mimo-v2.6-flash``。四个条件只差“交给流水线的问题”：

- ``last_turn_only``：只用追问原文（单轮系统的做法）；
- ``concat_history``：把历史各轮的问题和追问按顺序用空格拼成一句（不调用模型的朴素做法）；
- ``context_rewrite``：应用里的问题补全（``medidiag.workflow.context_rewrite``，提示词 ``context-rewrite-v1``），
  历史为数据集里固定下来的第一轮问题与回答；补全失败时回退到追问原文；
- ``oracle_standalone``：源查询本身（已标注的完整问题），作为上限参照。它的请求与 v3 / v3-ext 端到端评测中
  向量 Agent 加重排条件逐字相同，所以直接读那几份调用记录（``--base-cassette``，只读），不产生新调用。

生成只在 test 对话上进行。``--rewrite-only`` 只做问题补全，用于在 dev 对话上冒烟检查提示词，不检索也不生成。

模式：``record`` 需要真实凭据并占用持久预算；``replay`` 只读调用记录，绝不联网，未命中即失败。

用法::

    python -m eval.multiturn_eval --mode record --max-calls 780 --rerank-device cuda --out ...
    python -m eval.multiturn_eval --mode replay --rerank-device cuda --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.llm_pipeline_eval import (
    FEEDBACK_BASE,
    TOP_K,
    agent_summary,
    answer_f1,
    build_retrievers,
    generation_metrics,
    grounded_decision,
    make_provider,
    sha256_of,
)
from eval.rerank_matrix import CachedScorer
from eval.retrieval_benchmark import load_dataset, proportion
from eval.retrieval_matrix import POOL_METRICS, compare, query_metrics, summarize
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport
from medidiag.llm.models import ACTIVE_MIMO_MODEL, ALLOWED_MIMO_MODELS
from medidiag.rag.rerank import PairScorer, Reranker, load_local_reranker
from medidiag.workflow.context_rewrite import (
    CONTEXT_REWRITE_PROMPT,
    PROMPT_VERSION,
    Turn,
    contextualize,
)
from medidiag.workflow.retrieval_agent import SearchAgentResult, run_search_agent

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "multiturn-eval-v1"
CONDITIONS = ("last_turn_only", "concat_history", "context_rewrite", "oracle_standalone")
# (条件甲, 条件乙)：比较“甲 − 乙”
PAIRS: tuple[tuple[str, str], ...] = (
    ("context_rewrite", "last_turn_only"),
    ("context_rewrite", "concat_history"),
    ("concat_history", "last_turn_only"),
    ("oracle_standalone", "context_rewrite"),
)
RETRY_KEY = "feedback_retry"


@dataclass(frozen=True)
class _Hit:
    chunk_id: str
    text: str


def load_dialogues(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def turns_of(dialogue: dict[str, Any]) -> list[Turn]:
    """数据集里固定的历史，与 worker 交给补全步骤的结构相同。"""
    return [{"question": t["question"], "outcome": t["outcome"], "answer": list(t["answer"])} for t in dialogue["history"]]


def concat_question(dialogue: dict[str, Any]) -> str:
    return " ".join([t["question"] for t in dialogue["history"]] + [dialogue["follow_up"]])


def questions_for(dialogue: dict[str, Any], condition: str, rewrites: dict[str, tuple[str, dict[str, Any]]]) -> str:
    if condition == "last_turn_only":
        return str(dialogue["follow_up"])
    if condition == "concat_history":
        return concat_question(dialogue)
    if condition == "context_rewrite":
        return rewrites[dialogue["sample_id"]][0]
    return str(dialogue["source_question"])


def rewrite_summary(dialogues: Sequence[dict[str, Any]], rewrites: dict[str, tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    usage: Counter[str] = Counter()
    for _, info in rewrites.values():
        usage.update(info.get("usage", {}))
    by_id = {d["sample_id"]: d for d in dialogues}
    return {
        "dialogues": len(rewrites),
        "status": dict(Counter(info["status"] for _, info in rewrites.values())),
        "fallback_codes": dict(Counter(info.get("code", "") for _, info in rewrites.values() if info["status"] == "fallback")),
        # 补全句与源查询逐字相同的条数，只说明措辞接近，不是正确性指标
        "identical_to_source_question": sum(1 for sid, (q, _) in rewrites.items() if q == by_id[sid]["source_question"]),
        "model_calls": len(rewrites), "usage": dict(usage), "prompt_version": PROMPT_VERSION,
    }


def run(args: argparse.Namespace, transports: dict[int, Any], dialogues: Sequence[dict[str, Any]], dataset: Any,
        rerank_scorer: PairScorer | None) -> dict[str, Any]:
    conditions: list[str] = list(args.conditions)
    selected = [d for d in dialogues if d["split"] in args.splits]
    if args.limit:
        selected = selected[: args.limit]
    providers = {trial: make_provider(t) for trial, t in transports.items()}
    started = time.perf_counter()

    rewrites: dict[str, tuple[str, dict[str, Any]]] = {}
    if "context_rewrite" in conditions:
        with ThreadPoolExecutor(args.workers) as pool:
            done = pool.map(lambda d: contextualize(providers[0], d["follow_up"], turns_of(d), args.model), selected)
            rewrites = {d["sample_id"]: r for d, r in zip(selected, done, strict=True)}
    if args.rewrite_only:
        return {"rewrite": rewrite_summary(selected, rewrites),
                "rewrites": {sid: {"standalone_question": q, **info} for sid, (q, info) in rewrites.items()},
                "wall_seconds": round(time.perf_counter() - started, 1)}

    if rerank_scorer is None:
        raise ValueError("需要重排模型")
    chunk_text = {c["chunk_id"]: c["text"] for c in dataset.chunks}
    dense = build_retrievers(["dense"], dataset, args.embedding_revision)["dense"]
    reranker = Reranker(rerank_scorer, model_name=args.rerank_selected["model"],
                        model_revision=args.rerank_selected["revision"],
                        candidates=int(args.rerank_selected["candidates"]["dense_agent"]))

    def agent_for(question: str) -> SearchAgentResult:
        # 与应用和 llm_pipeline_eval 的 dense_agent_rerank 一致：模型每次只看前 3 段，各次检索的前 N 段按问题重排
        def search(text: str) -> list[_Hit]:
            return [_Hit(cid, chunk_text[cid]) for cid, _ in dense(None, text, reranker.candidates)]

        def merge(result_lists: Sequence[Sequence[_Hit]]) -> list[_Hit]:
            return [hit for hit, _ in reranker.rerank(question, [h for hits in result_lists for h in hits], TOP_K)]
        return run_search_agent(providers[0], search, question, search(question), args.model, observe=TOP_K, merge=merge)

    jobs = [(c, d) for c in conditions for d in selected]
    asked = {(c, d["sample_id"]): questions_for(d, c, rewrites) for c, d in jobs}
    with ThreadPoolExecutor(args.workers) as pool:
        runs = dict(zip([(c, d["sample_id"]) for c, d in jobs], pool.map(lambda j: agent_for(asked[(j[0], j[1]["sample_id"])]), jobs),
                        strict=True))

    def execute(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        condition, dialogue = job
        question = asked[(condition, dialogue["sample_id"])]
        ranked = [(hit.chunk_id, hit.text) for hit in runs[(condition, dialogue["sample_id"])].chunks]
        if not ranked:  # 与应用一致：空检索不调用模型
            return {"decision": "abstained", "selected": [], "model_called": False}
        first = {**grounded_decision(providers[0], question, ranked, args.model), "model_called": True}
        if first["decision"] == "invalid":  # 与应用的 on_invalid: feedback 一致，带上被拒原因再请求一次
            first[RETRY_KEY] = {**grounded_decision(providers[FEEDBACK_BASE], question, ranked, args.model,
                                                    feedback=first.get("detail")), "model_called": True}
        return first

    with ThreadPoolExecutor(args.workers) as pool:
        outcomes = dict(zip([(c, d["sample_id"]) for c, d in jobs], pool.map(execute, jobs), strict=True))

    rows = [{"sample_id": d["sample_id"], "bucket": d["bucket"], "gold": d["gold_evidence_ids"],
             "source_sample_id": d["source_sample_id"],
             "conditions": {c: {"question": asked[(c, d["sample_id"])],
                                "retrieved": [hit.chunk_id for hit in runs[(c, d["sample_id"])].chunks],
                                "trials": [outcomes[(c, d["sample_id"])]]} for c in conditions}}
            for d in selected]
    report = summarize_rows(rows, conditions)
    report["agent"] = {c: {"summary": agent_summary({d["sample_id"]: runs[(c, d["sample_id"])] for d in selected}),
                           "traces": {d["sample_id"]: {"status": runs[(c, d["sample_id"])].status,
                                                       "model_calls": runs[(c, d["sample_id"])].model_calls,
                                                       "steps": runs[(c, d["sample_id"])].steps} for d in selected}}
                       for c in conditions}
    if rewrites:
        report["rewrite"] = rewrite_summary(selected, rewrites)
        report["rewrites"] = {sid: {"standalone_question": q, **info} for sid, (q, info) in rewrites.items()}
    report["per_query"] = rows
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    return report


def summarize_rows(rows: Sequence[dict[str, Any]], conditions: Sequence[str]) -> dict[str, Any]:
    """只由逐条结果算出的指标；入库报告可以不用模型、不联网重算（``tests/test_multiturn_eval.py``）。"""
    answerable = [r for r in rows if r["gold"]]
    retrieval = {c: {r["sample_id"]: query_metrics([i for i, cid in enumerate(r["conditions"][c]["retrieved"], 1)
                                                   if cid in set(r["gold"])], len(r["gold"])) for r in answerable}
                 for c in conditions}

    def useful(row: dict[str, Any], condition: str) -> float:
        first = row["conditions"][condition]["trials"][0]
        final = first.get(RETRY_KEY, first)
        return float(final["decision"] == "answered" and bool(set(final["selected"]) & set(row["gold"])))

    generation = {c: generation_metrics(rows, c, 0, with_retry=RETRY_KEY) for c in conditions}
    ids = [r["sample_id"] for r in answerable]
    by_id = {r["sample_id"]: r for r in answerable}
    return {
        "n_dialogues": len(rows), "n_answerable": len(answerable), "n_unanswerable": len(rows) - len(answerable),
        "retrieval": {c: summarize([retrieval[c][i] for i in ids], POOL_METRICS) for c in conditions},
        "generation_first_pass": {c: generation_metrics(rows, c, 0) for c in conditions},
        "generation_with_feedback_retry": generation,
        "answer_f1": {c: answer_f1(generation[c]) for c in conditions},
        # 生成步骤的模型请求数：首次请求加带原因的重试
        "model_calls_generation": {c: sum(int(bool(r["conditions"][c]["trials"][0].get("model_called")))
                                          + int(RETRY_KEY in r["conditions"][c]["trials"][0]) for r in rows)
                                   for c in conditions},
        "paired_comparisons": [
            {"a": a, "b": b, "metric": metric,
             **compare([retrieval[a][i]["hit@3"] if metric == "hit@3" else useful(by_id[i], a) for i in ids],
                       [retrieval[b][i]["hit@3"] if metric == "hit@3" else useful(by_id[i], b) for i in ids])}
            for a, b in PAIRS if a in conditions and b in conditions and ids for metric in ("hit@3", "useful_answer")],
        "false_answers": {c: generation[c]["false_answer_rate"] for c in conditions},
        "over_refusals": {c: generation[c]["over_abstention_rate"] for c in conditions},
        "useful_answers": {c: generation[c]["useful_answer_rate"] for c in conditions},
        "answerable_by_bucket": dict(Counter(r["bucket"] for r in answerable)),
        "hit@3_by_bucket": {c: {b: proportion(int(sum(retrieval[c][r["sample_id"]]["hit@3"] for r in answerable
                                                      if r["bucket"] == b)),
                                              sum(1 for r in answerable if r["bucket"] == b))
                                for b in sorted({r["bucket"] for r in answerable})} for c in conditions},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", choices=ALLOWED_MIMO_MODELS, default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--dialogues", type=Path, default=ROOT / "examples/public_health_v3_multiturn/dialogues.jsonl")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "examples/public_health_v3")
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/multiturn-v3.jsonl")
    parser.add_argument("--feedback-cassette", type=Path, default=ROOT / "eval/cassettes/multiturn-v3-feedback.jsonl")
    parser.add_argument("--base-cassette", type=Path, nargs="*", default=[
        ROOT / "eval/cassettes/llm-pipeline-v3-rerank.jsonl", ROOT / "eval/cassettes/llm-pipeline-v3-flash.jsonl",
        ROOT / "eval/cassettes/llm-pipeline-v3-ext.jsonl"], help="只读的旧调用记录，依次查找；新记录只写入 --cassette")
    parser.add_argument("--base-feedback-cassette", type=Path, nargs="*", default=[
        ROOT / "eval/cassettes/llm-pipeline-v3-rerank-feedback.jsonl", ROOT / "eval/cassettes/llm-pipeline-v3-flash-feedback.jsonl",
        ROOT / "eval/cassettes/llm-pipeline-v3-ext-feedback.jsonl"])
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/multiturn-ledger.db")
    parser.add_argument("--max-calls", type=int, default=780, help="record 模式的持久调用上限（含失败）")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["dev", "test"])
    parser.add_argument("--rewrite-only", action="store_true", help="只做问题补全（dev 冒烟用），不检索也不生成")
    parser.add_argument("--rerank-report", type=Path, default=ROOT / "artifacts/reports/retrieval/rerank-matrix-v3-20261017.json")
    parser.add_argument("--rerank-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--embedding-revision", default="7999e1d3359715c523056ef9478215996d62a620")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只取前 N 条对话")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dialogues = load_dialogues(args.dialogues)
    dataset = load_dataset(args.corpus_dir)

    def chain(path: Path, bases: Sequence[Path]) -> Cassette:
        base: Cassette | None = None
        for item in reversed(bases):
            base = Cassette(item, base=base)
        return Cassette(path, base=base)

    cassette, feedback = chain(args.cassette, args.base_cassette), chain(args.feedback_cassette, args.base_feedback_cassette)
    args.rerank_selected = json.loads(args.rerank_report.read_text(encoding="utf-8"))["selected"]
    scorer: PairScorer | None = None
    if not args.rewrite_only:
        scorer = CachedScorer(load_local_reranker(args.rerank_selected["model"], args.rerank_selected["revision"],
                                                  device=args.rerank_device))
    transports: dict[int, Any]
    if args.mode == "record":
        transports = {0: RecordingTransport(cassette, args.ledger, max_calls=args.max_calls, trial=0),
                      FEEDBACK_BASE: RecordingTransport(feedback, args.ledger, max_calls=args.max_calls, trial=FEEDBACK_BASE)}
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        transports = {0: ReplayTransport(cassette, trial=0), FEEDBACK_BASE: ReplayTransport(feedback, trial=FEEDBACK_BASE)}
    result = run(args, transports, dialogues, dataset, scorer)
    spent = transports[0].spent() if args.mode == "record" else None
    report = {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(UTC).isoformat(),
        "config": {"model": args.model, "mode": args.mode, "conditions": args.conditions, "splits": args.splits,
                   "limit": args.limit or None, "rewrite_only": args.rewrite_only, "top_k": TOP_K,
                   "pipeline": "dense_agent_rerank（带原因重试一次）", "context_rewrite_prompt": CONTEXT_REWRITE_PROMPT,
                   "context_rewrite_prompt_version": PROMPT_VERSION, "temperature": "供应商默认（该 profile 不支持设置）",
                   "dialogues_sha256": hashlib.sha256(args.dialogues.read_bytes()).hexdigest(),
                   "corpus_sha256": dataset.fingerprints["chunks.jsonl"],
                   "cassette_sha256": sha256_of(args.cassette), "feedback_cassette_sha256": sha256_of(args.feedback_cassette),
                   "base_cassette_sha256": {p.name: sha256_of(p) for p in [*args.base_cassette, *args.base_feedback_cassette]},
                   "rerank": {**args.rerank_selected, "report": args.rerank_report.name, "device": args.rerank_device}},
        "spend": spent, **result}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:  # 排他写入，不覆盖已有报告
            handle.write(text + "\n")
    if "rewrite" in result:
        print("补全：", result["rewrite"])
    for condition in args.conditions if not args.rewrite_only else []:
        g = result["generation_with_feedback_retry"][condition]
        print(f"{condition:18s} hit@3={result['retrieval'][condition]['hit@3']['k']}/{result['n_answerable']} "
              f"正确作答={g['useful_answer_rate']['k']} 过度拒答={g['over_abstention_rate']['k']} "
              f"误答={g['false_answer_rate']['k']}/{g['false_answer_rate']['n']} 拦截={g['contract_violations_or_errors']}")
    if spent:
        print("花费：", spent)


if __name__ == "__main__":
    main()
