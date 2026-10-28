"""检索 Agent v2 与 v1 的端到端对照（M11）：v4 dev 中 intent=popular_science 的全部查询。

两个条件使用同一检索配置（v4 混合检索 + 重排、top3、parent_child 子块，不开查询路由），生成都是抽取式回答（代码中称 constrained excerpt）
（``MimoGroundedWorkflowProvider.generate``，``on_invalid=feedback``）：

- ``agent_v1``：``RetrievalAgentWorkflowProvider``，只会换说法再检索；
- ``agent_v2``：``RetrievalAgentV2WorkflowProvider``，多工具、带检索评估（CRAG）、并行与预算上限。

指标定义（A 为 answerable=true 的查询，U 为 answerable=false 的查询；标注口径同 ``eval.hybrid_matrix``，
``relevant`` 与 ``partial`` 都算相关，没有任何相关块的 A 查询按未命中计）：

- 最终证据 Hit@3：A 中，最终交给生成的证据（最多 3 段）含相关块的比例；
- 正确回答率：A 中，作答（至少一条摘录）且至少一段被引用的块相关的比例；
- 误答率分两类：作答但被引用的块全都不相关（分母 A）；U 被作答（分母 U）；
- 过度拒答率：A 中拒答的比例。生成阶段报错的查询单列，不算拒答；
- 平均模型调用数（决策、HyDE、生成分开列）、工具调用数（v1 为额外检索次数）、token；
- 每题端到端时延 p50 / p90：record 时为实测整题耗时；replay 时为该题所有调用记录中 elapsed 之和
  （不含本地检索与重排，同一轮并行的 HyDE 按串行相加，略偏高）。record 报告同时给出同口径的 ``llm_ms``。

``record`` 需要真实凭据并占用持久预算账本；``replay`` 只读调用记录，绝不联网，并与 record 报告核对结果是否一致。

用法::

    python -m eval.agent_v2_eval --mode record --limit 3   # 冒烟，不写报告
    python -m eval.agent_v2_eval --mode record             # 全量
    python -m eval.agent_v2_eval --mode replay             # 回放并核对
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.hybrid_matrix import (  # noqa: E402
    CHUNKS_PATH,
    JUDGMENTS_PATH,
    QUERIES_PATH,
    load_queries,
    percentile,
)
from eval.llm_pipeline_eval import call_with_retry, make_provider  # noqa: E402
from eval.router_eval import build_relevance  # noqa: E402
from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import (  # noqa: E402
    Cassette,
    RecordingTransport,
    ReplayTransport,
    request_key,
)
from medidiag.llm.models import ACTIVE_MIMO_MODEL  # noqa: E402
from medidiag.workflow.retrieval_agent import RetrievalAgentWorkflowProvider  # noqa: E402
from medidiag.workflow.retrieval_agent_v2 import RetrievalAgentV2WorkflowProvider  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "agent-v2-eval-v1"
REPORT_DIR = ROOT / "artifacts/reports/retrieval"
CONFIG = "configs/application_v4_agent.yaml"
CONDITIONS = ("agent_v1", "agent_v2")
# 费用估算口径同 M3：每百万输入 Token 1 元、输出 2 元（未核实实际计费，仅供量级参考）
YUAN_PER_M_PROMPT = 1.0
YUAN_PER_M_COMPLETION = 2.0
CALL_KINDS = {"retrieval-agent-v1": "agent", "retrieval-agent-v2": "agent", "hyde-v1": "hyde", "mimo-grounded-v1": "generation"}


class _QueryTransport:
    """包一层传输：每次调用后从调用记录里取该请求的 elapsed，累加到当前这道题上。"""

    def __init__(self, inner: Callable[..., Any], cassette: Cassette) -> None:
        self.inner, self.cassette = inner, cassette
        self.elapsed_ms: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float, **kw: Any) -> Any:
        response = self.inner(url, headers=headers, json=json, timeout=timeout, **kw)
        entry = self.cassette.get(request_key(json, 0))
        if entry is not None:
            with self._lock:
                self.elapsed_ms.append(int(entry.get("elapsed_ms") or 0))
        return response


class _QueryLLM:
    """一道题专用的模型入口：可重试错误自动重试，按调用类型统计次数、错误码和 token。"""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.calls: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()
        self._lock = threading.Lock()

    def generate(self, request: Any, *, timeout_s: float, idempotency_key: str) -> Any:
        kind = CALL_KINDS.get(request.prompt_version, request.prompt_version)
        with self._lock:
            self.calls[kind] += 1
        try:
            result = call_with_retry(lambda: self.provider.generate(
                request, timeout_s=timeout_s, idempotency_key=idempotency_key))
        except MediDiagError as exc:
            with self._lock:
                self.errors[exc.code] += 1
            raise
        with self._lock:
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                self.tokens[name] += int((result.usage or {}).get(name) or 0)
        return result


def dev_queries() -> list[dict[str, Any]]:
    return [q for q in load_queries() if q.get("split") == "dev" and q.get("intent") == "popular_science"]


def build_rag(device: str) -> Any:
    """按应用配置建检索（混合检索 + 重排）；评测里 device 用 cuda 覆盖配置里的 cpu，向量缓存在 .cache/index/。"""
    from medidiag.rag.runtime import RuntimeMedicalRAG, _hybrid_factory
    from medidiag.workflow.application import _rerank_factory, load_application_config

    config = load_application_config(ROOT / CONFIG)
    config["runtime"] = {"device": device, "batch_size": 64 if device == "cuda" else 8}
    started = time.time()
    rag = RuntimeMedicalRAG.from_config(config, root=ROOT, retriever_factory=_hybrid_factory,
                                        reranker_factory=_rerank_factory)
    print(f"[rag] 检索就绪（{device}，子块 {len(rag.chunks)}，{time.time() - started:.0f}s）", flush=True)
    return rag, config


def serialize_retrieval(rag: Any) -> None:
    """评测里 4 个线程共用同一份模型与索引；检索与重排串行执行，并行只发生在模型调用上。"""
    lock = threading.RLock()

    def locked(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return fn(*args, **kwargs)
        return wrapper

    for name in ("search_evidence", "search_filtered", "rerank_evidence"):
        setattr(rag, name, locked(getattr(rag, name)))


def make_base(condition: str, rag: Any, model: str) -> Any:
    if condition == "agent_v1":
        return RetrievalAgentWorkflowProvider(rag, llm=None, on_invalid="feedback", model=model)
    provider = RetrievalAgentV2WorkflowProvider(rag, llm=None, on_invalid="feedback", model=model)
    provider._parent_map()  # 先加载父块，之后每题复制出的 provider 共用同一份
    return provider


def classify(query: dict[str, Any], answered: bool, error: bool, cited_relevant: bool) -> str:
    if error:
        return "generation_error"
    if query.get("answerable") is True:
        if not answered:
            return "over_refusal"
        return "correct" if cited_relevant else "answered_irrelevant"
    return "answered_unanswerable" if answered else "correct_abstain"


def run_query(base: Any, query: dict[str, Any], positives: set[str], inner: Callable[..., Any],
              cassette: Cassette, mode: str) -> dict[str, Any]:
    transport = _QueryTransport(inner, cassette)
    llm = _QueryLLM(make_provider(transport))
    provider = replace(base, llm=llm)
    started = time.perf_counter()
    question = provider.normalize(query["question"])["normalized_query"]
    error: str | None = None
    claims: list[dict[str, Any]] = []
    retrieval: dict[str, Any] = {"chunks": [], "search_agent": {}}
    try:
        retrieval = provider.retrieve(question)
        response = provider.generate(question, retrieval, provider.plan(question, retrieval))
        claims = list(response.payload.get("claims", []))
    except MediDiagError as exc:
        error = exc.code
    wall_ms = round((time.perf_counter() - started) * 1000)
    llm_ms = sum(transport.elapsed_ms)
    evidence_ids = [c["chunk_id"] for c in retrieval["chunks"]]
    cited = [cid for c in claims for cid in c["citation_chunk_ids"]]
    agent = retrieval.get("search_agent") or {}
    steps = agent.get("steps", [])
    if "tool_calls" in agent:  # v2
        tool_calls = int(agent["tool_calls"])
    else:  # v1：首轮检索之外的检索次数
        tool_calls = max(0, sum(1 for s in steps if s.get("action") == "search") - 1)
    answered = bool(claims)
    return {
        "sample_id": query["sample_id"], "answerable": query.get("answerable"),
        "evidence_ids": evidence_ids, "hit3": any(cid in positives for cid in evidence_ids[:3]),
        "answered": answered, "cited_ids": cited, "cited_relevant": any(cid in positives for cid in cited),
        "outcome": classify(query, answered, error is not None, any(cid in positives for cid in cited)),
        "generation_error": error, "model_calls": dict(llm.calls), "tool_calls": tool_calls,
        "tokens": dict(llm.tokens), "call_errors": dict(llm.errors),
        "latency_ms": wall_ms if mode == "record" else llm_ms, "wall_ms": wall_ms, "llm_ms": llm_ms,
        "agent": {k: v for k, v in agent.items() if k != "usage"}}


def run_condition(condition: str, base: Any, queries: Sequence[dict[str, Any]], relevance: dict[str, set[str]],
                  inner: Callable[..., Any], cassette: Cassette, mode: str, workers: int) -> list[dict[str, Any]]:
    """并发跑一个条件，按 5 条或 30 秒打印进度。"""
    rows: list[dict[str, Any]] = []
    started = last_print = time.time()

    def one(query: dict[str, Any]) -> dict[str, Any]:
        return run_query(base, query, relevance[query["sample_id"]], inner, cassette, mode)

    with ThreadPoolExecutor(workers) as pool:
        for done, row in enumerate(pool.map(one, queries), 1):
            rows.append(row)
            if done % 5 == 0 or done == len(queries) or time.time() - last_print > 30:
                last_print = time.time()
                print(f"[{condition}] {done}/{len(queries)}（{last_print - started:.0f}s）", flush=True)
    return rows


def rate(count: int, total: int) -> dict[str, Any]:
    return {"count": f"{count}/{total}", "rate": round(count / total, 4) if total else None}


def mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
    answerable = [r for r in rows if r["answerable"] is True]
    unanswerable = [r for r in rows if r["answerable"] is not True]
    outcomes = Counter(r["outcome"] for r in rows)
    kinds = ("agent", "hyde", "generation")
    calls = {k: mean([r["model_calls"].get(k, 0) for r in rows]) for k in kinds}
    total_calls = mean([sum(r["model_calls"].values()) for r in rows])
    latency = [float(r["latency_ms"]) for r in rows]
    return {
        "n": len(rows), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        "evidence_hit@3": rate(sum(r["hit3"] for r in answerable), len(answerable)),
        "correct_answer": rate(outcomes["correct"], len(answerable)),
        "wrong_answer_irrelevant_citation": rate(outcomes["answered_irrelevant"], len(answerable)),
        "wrong_answer_unanswerable_answered": rate(outcomes["answered_unanswerable"], len(unanswerable)),
        "over_refusal": rate(outcomes["over_refusal"], len(answerable)),
        "correct_abstain_unanswerable": rate(outcomes["correct_abstain"], len(unanswerable)),
        "generation_errors": outcomes["generation_error"], "outcomes": dict(outcomes),
        "avg_model_calls": {"total": total_calls, **calls},
        "avg_tool_calls": mean([r["tool_calls"] for r in rows]),
        "avg_tokens": {k: mean([r["tokens"].get(k, 0) for r in rows]) for k in ("input_tokens", "output_tokens", "total_tokens")},
        "latency_ms": {"p50": percentile(latency, 50), "p90": percentile(latency, 90),
                       "source": "wall_ms（实测整题）" if mode == "record" else "llm_ms（调用记录 elapsed 之和）"},
        "llm_ms": {"p50": percentile([float(r["llm_ms"]) for r in rows], 50),
                   "p90": percentile([float(r["llm_ms"]) for r in rows], 90)}}


def v2_extras(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tools: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    status: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    parallel_rounds = 0
    for row in rows:
        agent = row["agent"]
        status[agent.get("status", "none")] += 1
        stops[str(agent.get("stop_reason"))] += 1
        parallel_rounds += int(agent.get("parallel_rounds", 0))
        labels.update(agent.get("labels", {}).values())
        if agent.get("code"):
            codes[agent["code"]] += 1
        for step in agent.get("steps", []):
            if step.get("action") == "tool":
                tools[step["tool"]] += 1
            elif step.get("action") == "skip":
                tools["skipped:" + str(step.get("reason"))] += 1
    return {"tool_usage": dict(tools), "parallel_rounds": parallel_rounds,
            "queries_with_parallel_round": sum(1 for r in rows if r["agent"].get("parallel_rounds")),
            "fallbacks": status["fallback"], "fallback_codes": dict(codes), "insufficient": status["insufficient"],
            "agent_status": dict(status), "stop_reasons": dict(stops), "crag_labels": dict(labels),
            "queries_with_hyde": sum(1 for r in rows if r["model_calls"].get("hyde"))}


def consistency_view(report: dict[str, Any]) -> dict[str, Any]:
    """回放核对用的视图：去掉时延，其余逐题字段必须与 record 一致。"""
    skip = {"latency_ms", "wall_ms", "llm_ms"}
    return {name: {r["sample_id"]: {k: v for k, v in r.items() if k not in skip} for r in rows}
            for name, rows in report["per_query"].items()}


def run(args: argparse.Namespace, transport: Any | None = None) -> dict[str, Any]:
    from medidiag.schemas import read_jsonl

    for path in (CHUNKS_PATH, QUERIES_PATH, JUDGMENTS_PATH):
        if not path.exists():
            raise SystemExit(f"[agent_v2_eval] 缺少 {path}")
    queries = dev_queries()
    if args.limit:
        queries = queries[: args.limit]
    answerable = sum(1 for q in queries if q.get("answerable") is True)
    print(f"[load] dev 科普查询 {len(queries)} 条（可回答 {answerable}，不可回答 {len(queries) - answerable}）", flush=True)
    relevance = build_relevance(queries, read_jsonl(CHUNKS_PATH))
    import torch

    torch.set_num_threads(2)
    rag, _ = build_rag("cuda" if torch.cuda.is_available() else "cpu")
    serialize_retrieval(rag)

    cassette = Cassette(args.cassette)
    if transport is not None:
        inner: Any = transport
    elif args.mode == "record":
        inner = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        inner = ReplayTransport(cassette)

    per_query: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        base = make_base(condition, rag, args.model)
        per_query[condition] = run_condition(condition, base, queries, relevance, inner, cassette, args.mode, args.workers)
        print(f"[{condition}] 完成 " + json.dumps({k: v for k, v in summarize(per_query[condition], args.mode).items()
                                                   if k in ("evidence_hit@3", "correct_answer", "over_refusal")},
                                                  ensure_ascii=False), flush=True)

    errors: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    calls = 0
    for rows in per_query.values():
        for row in rows:
            errors.update(row["call_errors"])
            tokens.update(row["tokens"])
            calls += sum(row["model_calls"].values())
    cost = {"model_calls": calls, "call_errors": dict(errors), "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
            "est_yuan": round((tokens["input_tokens"] * YUAN_PER_M_PROMPT
                               + tokens["output_tokens"] * YUAN_PER_M_COMPLETION) / 1e6, 4)}
    if args.mode == "record" and hasattr(inner, "spent"):
        cost["spend"] = inner.spent()
    return {
        "schema_version": SCHEMA_VERSION, "date": str(date.today()), "mode": args.mode, "model": args.model,
        "split": "dev", "intent": "popular_science", "queries": len(queries), "cost": cost,
        "params": {"config": CONFIG, "retrieval": "hybrid + rerank + MMR，parent_child 子块，top3，不开查询路由",
                   "generation": "抽取式回答，on_invalid=feedback",
                   "agent_v2_budget": {"tool_calls": 4, "model_calls": 5, "total_tokens": 12000}},
        "summary": {name: summarize(rows, args.mode) for name, rows in per_query.items()},
        "agent_v2": v2_extras(per_query["agent_v2"]),
        "agent_v1_extras": {"avg_extra_searches": mean([r["tool_calls"] for r in per_query["agent_v1"]]),
                            "fallbacks": sum(1 for r in per_query["agent_v1"] if r["agent"].get("status") == "fallback")},
        "note": "标注由 LLM 完成、未经人工复核；相关 = relevant 或 partial；样本量小，只作工程对照；"
                "引用块相关只说明摘录来自相关句子所在的块，不等于医学正确。",
        "per_query": per_query}


def replay_check(report: dict[str, Any], path: Path) -> dict[str, Any]:
    """回放结果与已写入的 record 报告对照：逐题的证据、引用、调用数与 token 必须一致。"""
    if not path.exists():
        return {"checked": False, "reason": "没有找到 record 报告"}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(report, ensure_ascii=False))
    before, after = consistency_view(recorded), consistency_view(current)
    differing = [f"{name}/{sid}" for name in before for sid in before[name] if before[name][sid] != after.get(name, {}).get(sid)]
    errors = sum(report["cost"]["call_errors"].values())
    calls = report["cost"]["model_calls"]
    return {"checked": True, "consistent": not differing and before.keys() == after.keys(), "differing": differing[:10],
            "cassette_hit_rate": f"{calls - errors}/{calls}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/v4-agent-v2.jsonl")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/v4-m11-ledger.db")
    parser.add_argument("--max-calls", type=int, default=700)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只评前 N 条查询，不写报告")
    args = parser.parse_args()

    report = run(args)
    path = REPORT_DIR / f"agent-v2-v4-{date.today():%Y%m%d}.json"
    brief = {k: report[k] for k in ("queries", "cost", "summary")}
    if args.limit:
        print(json.dumps({**brief, "agent_v2": report["agent_v2"]}, ensure_ascii=False, indent=2))
        return
    if args.mode == "replay":
        report["replay_check"] = replay_check(report, path)
        print(json.dumps(report["replay_check"], ensure_ascii=False, indent=2))
        if report["replay_check"].get("checked"):
            # 一致时保留 record 报告并附上核对结果与回放时延；不一致时不覆盖，交给人排查
            if report["replay_check"]["consistent"]:
                recorded = json.loads(path.read_text(encoding="utf-8"))
                recorded["replay_check"] = report["replay_check"]
                recorded["replay_latency_ms"] = {n: s["latency_ms"] for n, s in report["summary"].items()}
                path.write_text(json.dumps(recorded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**brief, "agent_v2": report["agent_v2"]}, ensure_ascii=False, indent=2))
    print(f"[done] 报告 {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
