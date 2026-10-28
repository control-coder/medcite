"""最终对照（M15）：v4 test 全部 202 条，三个条件各跑一次。

条件（配置驱动，检索与回答装配与应用一致）：

1. ``v3_recommended_on_v4``：v3 推荐配置（向量检索 + 检索 Agent v1 + 重排候选 5 + 抽取式，无路由），语料换成 v4 父子子块；
2. ``v4_extractive``：``configs/application_v4_recommended.yaml``（混合检索 + 重排 + 路由 + 检索 Agent v2 + 抽取式）；
3. ``v4_generative``：``configs/application_v4_recommended_generative.yaml``，只读复用条件 2 的路由与检索 Agent 调用记录。

指标定义（口径同 M11 ``eval.agent_v2_eval`` 与 M13 ``eval.generation_quality``；标注为 LLM 句子级标注，relevant 与 partial 都算相关）：

- 检索：可回答科普题的最终证据 Hit@3（被路由拒答的计未命中，另列被路由拒答数）；
- 回答：正确回答率（作答且至少一段被引用的块相关）、作答但引用全不相关（分母可回答题）、
  无答案被作答（answerable=false 的科普题，分母该类题）、过度拒答（可回答却拒答，另列其中由路由造成的）；
- 非科普意图（individual_care / out_of_scope / emergency）：未作答的比例、给出路由模板的比例；急症是否给出急症提示；
- 生成侧（仅条件 3）：引用召回、引用精确率、忠实度、答案相关性、上下文精确率、降级数、删句数、partial 数（M13 评审 eval-nli-v1）；
- 成本：分类型的模型调用数（含命中旧记录的）、token、估算费用（口径同 M11）、端到端时延。

时延：``wall_ms`` 为实测整题耗时；复用旧记录或回放时调用瞬间返回，所以主口径 ``latency_ms`` = 整题耗时 - 调用实际耗时
+ 调用记录里的 elapsed，各条件可比（并行的 HyDE 按串行相加，略偏高；4 个线程共用 GPU 检索，含排队）。

v4 结果不与 v2/v3 结果混比；标注未经人工复核；部分 v3 dev 查询落入 v4 test（对基线略有利，见 status M7 阶段 A）；单次运行。

用法（三个条件分开跑，可断点续跑，请求相同命中调用记录、不重复计费）::

    python -m eval.final_comparison --mode record --split dev --limit 3 --conditions v3_recommended_on_v4,v4_extractive,v4_generative
    python -m eval.final_comparison --mode record --conditions v3_recommended_on_v4
    python -m eval.final_comparison --mode record --conditions v4_extractive
    python -m eval.final_comparison --mode record --conditions v4_generative
    python -m eval.final_comparison --mode replay
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval import generation_quality as gq  # noqa: E402
from eval.agent_v2_eval import classify, serialize_retrieval  # noqa: E402
from eval.hybrid_matrix import (  # noqa: E402
    CHUNKS_PATH,
    JUDGMENTS_PATH,
    QUERIES_PATH,
    load_queries,
    percentile,
)
from eval.llm_pipeline_eval import make_provider  # noqa: E402
from eval.router_eval import build_relevance  # noqa: E402
from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import (  # noqa: E402
    Cassette,
    RecordingTransport,
    ReplayTransport,
    request_key,
)
from medidiag.llm.models import ACTIVE_MIMO_MODEL  # noqa: E402
from medidiag.workflow.application import load_application_config  # noqa: E402
from medidiag.workflow.generative_answer import ENTAILMENT_VERSION, PROMPT_VERSION  # noqa: E402
from medidiag.workflow.query_router import PROMPT_VERSIONS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "final-comparison-v1"
REPORT_DIR = ROOT / "artifacts/reports/final"
PARTS_DIR = ROOT / ".cache/final-parts"
SMOKE_DIR = ROOT / ".cache/final-smoke"
CASSETTE_DIR = ROOT / "eval/cassettes"
LEDGER = ROOT / ".cache/v4-m15-ledger.db"
V3_CONFIG = "configs/application_dense_agent_rerank.yaml"
CONDITIONS = ("v3_recommended_on_v4", "v4_extractive", "v4_generative")
CONFIG_FILES = {"v4_extractive": "configs/application_v4_recommended.yaml",
                "v4_generative": "configs/application_v4_recommended_generative.yaml"}
NON_POPULAR = ("individual_care", "out_of_scope", "emergency")
# 费用估算口径同 M11：每百万输入 Token 1 元、输出 2 元（未核实实际计费，仅供量级参考）
YUAN_PER_M_PROMPT = 1.0
YUAN_PER_M_COMPLETION = 2.0
BASE_KINDS = {"query-router-v1": "router", "query-router-v2": "router", "retrieval-agent-v1": "agent", "retrieval-agent-v2": "agent",
              "hyde-v1": "hyde"}
EXTRACTIVE_KINDS = {**BASE_KINDS, "mimo-grounded-v1": "generation"}
GENERATIVE_KINDS = {**BASE_KINDS, PROMPT_VERSION: "generation", ENTAILMENT_VERSION: "entailment",
                    "mimo-grounded-v1": "extractive"}
TIME_KEYS = {"wall_ms", "latency_ms"}


# ----------------------------------------------------------------- 配置与装配

def condition_config(condition: str) -> dict[str, Any]:
    """条件对应的应用配置：条件 1 是 v3 推荐配置换成 v4 父子子块语料，其余直接读 v4 推荐配置（均经 load_application_config 校验）。"""
    if condition == "v3_recommended_on_v4":
        config: dict[str, Any] = copy.deepcopy(load_application_config(ROOT / V3_CONFIG))
        v4: dict[str, Any] = load_application_config(ROOT / CONFIG_FILES["v4_extractive"])
        config["dataset"] = copy.deepcopy(v4["dataset"])
        return config
    loaded: dict[str, Any] = load_application_config(ROOT / CONFIG_FILES[condition])
    return loaded


def build_rag(config: dict[str, Any], device: str) -> Any:
    """按配置建检索（评测里 device 用 cuda 覆盖配置里的 cpu）。"""
    from medidiag.rag.runtime import RuntimeMedicalRAG, _hybrid_factory
    from medidiag.workflow.application import _dense_factory, _rerank_factory

    config = copy.deepcopy(config)
    config["runtime"] = {"device": device, "batch_size": 64 if device == "cuda" else 8}
    factory = _hybrid_factory if config["retrieval"]["method"] == "hybrid" else _dense_factory
    started = time.time()
    rag = RuntimeMedicalRAG.from_config(config, root=ROOT, retriever_factory=factory,
                                        reranker_factory=_rerank_factory if config["retrieval"].get("rerank") else None)
    print(f"[rag] {config['retrieval']['method']} 检索就绪（{device}，子块 {len(rag.chunks)}，{time.time() - started:.0f}s）",
          flush=True)
    return rag


def make_base(config: dict[str, Any], rag: Any, model: str) -> Any:
    """按配置装配工作流（与 build_application_provider 的 mimo_grounded 分支一致，模型入口之后按题注入）。"""
    from medidiag.workflow.retrieval_agent import RetrievalAgentWorkflowProvider
    from medidiag.workflow.retrieval_agent_v2 import RetrievalAgentV2WorkflowProvider

    generation = config.get("generation", {})
    common: dict[str, Any] = {"llm": None, "on_invalid": generation.get("on_invalid", "none"), "model": model,
                              "query_router": bool(generation.get("query_router", False)),
                              "router_prompt_version": PROMPT_VERSIONS[generation.get("router_prompt", "v1")],  # M15 记录基于 v1
                              "answer_mode": generation.get("answer_mode", "extractive")}
    if not generation.get("agent", False):
        raise ValueError("最终对照的三个条件都必须开启检索 Agent")
    if generation.get("agent_version", 1) == 2:
        provider = RetrievalAgentV2WorkflowProvider(rag, **common)
        provider._parent_map()  # 先加载父块，之后每题复制出的 provider 共用同一份
        return provider
    return RetrievalAgentWorkflowProvider(rag, **common)


# ----------------------------------------------------------------- 单题流程

class _TimedTransport:
    """记录一道题内每次调用的实际耗时与调用记录里的 elapsed，用来把复用旧记录的时延还原成可比口径。"""

    def __init__(self, inner: Callable[..., Any], cassette: Cassette) -> None:
        self.inner, self.cassette = inner, cassette
        self.actual_ms = 0.0
        self.recorded_ms = 0
        self._lock = threading.Lock()

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float, **kw: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.inner(url, headers=headers, json=json, timeout=timeout, **kw)
        finally:
            actual = (time.perf_counter() - started) * 1000
            entry = self.cassette.get(request_key(json, 0))
            with self._lock:
                self.actual_ms += actual
                self.recorded_ms += int((entry or {}).get("elapsed_ms") or 0)


def _route_info(retrieval: Mapping[str, Any]) -> dict[str, Any] | None:
    route = retrieval.get("query_router")
    if not isinstance(route, dict):
        return None
    return {k: route.get(k) for k in ("intent", "source", "refused", "rule", "fallback_reason")}


def run_query(base: Any, query: dict[str, Any], positives: set[str], inner: Callable[..., Any], cassette: Cassette,
              model: str, generative: bool) -> dict[str, Any]:
    transport = _TimedTransport(inner, cassette)
    provider_llm = make_provider(transport)
    kinds = GENERATIVE_KINDS if generative else EXTRACTIVE_KINDS
    pipeline, judge = gq._LLM(provider_llm, kinds), gq._LLM(provider_llm, gq.JUDGE_KINDS)
    provider = replace(base, llm=pipeline)
    started = time.perf_counter()
    question = provider.normalize(query["question"])["normalized_query"]
    error: str | None = None
    retrieval: dict[str, Any] = {"chunks": []}
    payload: dict[str, Any] = {"claims": []}
    try:
        retrieval = provider.retrieve(question)
        payload = provider.generate(question, retrieval, provider.plan(question, retrieval)).payload
    except MediDiagError as exc:
        error = exc.code
    wall_ms = round((time.perf_counter() - started) * 1000)
    latency_ms = max(0, round(wall_ms - transport.actual_ms + transport.recorded_ms))
    claims = list(payload.get("claims", []))
    trace = payload.get("generative_trace") or {}
    evidence = {c["chunk_id"]: c["text"] for c in retrieval["chunks"]}
    evidence_ids = list(evidence)
    cited = [cid for c in claims for cid in c.get("citation_chunk_ids", [])]
    route = _route_info(retrieval)
    routed_out = bool(route and route["refused"])
    answered = bool(claims)
    degraded = payload.get("degraded_from") == "generative"
    shown = generative and answered and payload.get("answer_mode") == "generative" and not degraded
    outcome = ("error" if error else "degraded_extractive" if degraded else "generative" if shown
               else "extractive" if answered else "refused")
    sentences = [{"text": c["text"], "entailment": c.get("entailment"), "suspicious": c.get("suspicious"),
                  "citations": c["citations"]} for c in claims] if shown else []
    judged: dict[str, Any] = {"sentences": [], "questions": None, "errors": []}
    if sentences:
        judged = gq.judge_answer(judge, model, evidence, sentences)
    is_a = query.get("answerable") is True
    return {
        "sample_id": query["sample_id"], "intent": query["intent"], "answerable": query.get("answerable"),
        "partial_only": bool(query.get("partial_only")), "question": question, "route": route, "routed_out": routed_out,
        "evidence_ids": evidence_ids, "hit3": any(c in positives for c in evidence_ids[:3]),
        "answered": answered, "cited_ids": cited, "cited_relevant": any(c in positives for c in cited),
        "classification": classify(query, answered, error is not None, any(c in positives for c in cited)),
        "outcome": outcome, "generation_error": error,
        "context_precision": gq.context_precision(evidence_ids, positives) if is_a else None,
        "sentences": sentences, "extractive_claims": [c["text"] for c in claims] if degraded else [],
        "dropped": trace.get("dropped", []), "invalid_reasons": trace.get("invalid_reasons", []),
        "entailment_error": trace.get("entailment_error"), "degraded": trace.get("degraded"),
        "judge": judged["sentences"], "generated_questions": judged["questions"], "judge_errors": judged["errors"],
        "pipeline_calls": dict(pipeline.calls), "judge_calls": dict(judge.calls),
        "call_errors": dict(pipeline.errors + judge.errors),
        "tokens": dict(pipeline.tokens), "judge_tokens": dict(judge.tokens),
        "wall_ms": wall_ms, "latency_ms": latency_ms}


def run_all(base: Any, queries: Sequence[dict[str, Any]], relevance: Mapping[str, set[str]], inner: Any,
            cassette: Cassette, model: str, generative: bool, workers: int, label: str) -> list[dict[str, Any]]:
    """并发跑一个条件，每 10 条或 30 秒打印进度（含账本已用调用数）。"""
    rows: list[dict[str, Any]] = []
    started = last_print = time.time()

    def one(query: dict[str, Any]) -> dict[str, Any]:
        return run_query(base, query, relevance.get(query["sample_id"], set()), inner, cassette, model, generative)

    with ThreadPoolExecutor(workers) as pool:
        for done, row in enumerate(pool.map(one, queries), 1):
            rows.append(row)
            if done % 10 == 0 or done == len(queries) or time.time() - last_print > 30:
                last_print = time.time()
                spent = inner.spent()["calls"] if hasattr(inner, "spent") else "-"
                print(f"[{label}] {done}/{len(queries)}（{last_print - started:.0f}s，账本累计调用 {spent}）", flush=True)
    return rows


# ----------------------------------------------------------------- 汇总（纯函数）

def rate(count: int, total: int) -> dict[str, Any]:
    return {"count": f"{count}/{total}", "rate": round(count / total, 4) if total else None}


def summarize_popular(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """科普题（intent=popular_science）的检索与回答指标。"""
    pop = [r for r in rows if r["intent"] == "popular_science"]
    a = [r for r in pop if r["answerable"] is True]
    u = [r for r in pop if r["answerable"] is not True]
    over = [r for r in a if r["classification"] == "over_refusal"]
    answered_u = [r for r in u if r["classification"] == "answered_unanswerable"]
    return {
        "n": len(pop), "n_answerable": len(a), "n_unanswerable": len(u),
        "n_partial_only": sum(1 for r in u if r["partial_only"]),
        "evidence_hit@3": rate(sum(r["hit3"] for r in a), len(a)),
        "routed_out_answerable": rate(sum(r["routed_out"] for r in a), len(a)),
        "routed_out_unanswerable": rate(sum(r["routed_out"] for r in u), len(u)),
        "correct_answer": rate(sum(r["classification"] == "correct" for r in a), len(a)),
        "wrong_answer_irrelevant_citation": rate(sum(r["classification"] == "answered_irrelevant" for r in a), len(a)),
        "wrong_answer_unanswerable_answered": rate(len(answered_u), len(u)),
        "unanswerable_answered_partial_only": rate(sum(r["partial_only"] for r in answered_u), sum(r["partial_only"] for r in u)),
        "unanswerable_answered_no_answer": rate(sum(not r["partial_only"] for r in answered_u),
                                                sum(not r["partial_only"] for r in u)),
        "over_refusal": rate(len(over), len(a)),
        "over_refusal_by_router": rate(sum(r["routed_out"] for r in over), len(a)),
        "generation_errors": sum(1 for r in pop if r["generation_error"]),
        "context_precision": {"mean": _mean([r["context_precision"] for r in a if r["context_precision"] is not None]),
                              "n": len(a)}}


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize_non_popular(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """非科普意图：是否拒答、是否给出路由模板；急症是否给出急症提示。"""
    other = [r for r in rows if r["intent"] in NON_POPULAR]
    by_intent = {}
    for intent in NON_POPULAR:
        sub = [r for r in other if r["intent"] == intent]
        by_intent[intent] = {
            "n": len(sub), "not_answered": rate(sum(not r["answered"] for r in sub), len(sub)),
            "template_given": rate(sum(r["routed_out"] for r in sub), len(sub)),
            "answered": rate(sum(r["answered"] for r in sub), len(sub))}
    emergency = [r for r in other if r["intent"] == "emergency"]
    return {
        "n": len(other), "not_answered": rate(sum(not r["answered"] for r in other), len(other)),
        "template_given": rate(sum(r["routed_out"] for r in other), len(other)), "by_intent": by_intent,
        "emergency_prompt_given": rate(sum(bool(r["route"] and r["route"]["intent"] == "emergency" and r["routed_out"])
                                           for r in emergency), len(emergency)),
        "emergency_answered": rate(sum(r["answered"] for r in emergency), len(emergency)),
        "generation_errors": sum(1 for r in other if r["generation_error"])}


def summarize_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls: Counter[str] = Counter()
    judge_calls: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    judge_tokens: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for r in rows:
        calls.update(r["pipeline_calls"])
        judge_calls.update(r["judge_calls"])
        tokens.update(r["tokens"])
        judge_tokens.update(r["judge_tokens"])
        errors.update(r["call_errors"])

    def yuan(t: Mapping[str, int]) -> float:
        return (t.get("input_tokens", 0) * YUAN_PER_M_PROMPT + t.get("output_tokens", 0) * YUAN_PER_M_COMPLETION) / 1e6

    grouped = {"router": calls["router"], "agent": calls["agent"], "hyde": calls["hyde"],
               "generation": calls["generation"] + calls["extractive"], "entailment": calls["entailment"],
               "judge": sum(judge_calls.values())}
    total_tokens = tokens + judge_tokens
    latency = [float(r["latency_ms"]) for r in rows]
    wall = [float(r["wall_ms"]) for r in rows]
    return {
        "model_calls_by_kind": grouped, "model_calls_total": sum(grouped.values()),
        "model_calls_raw": {**dict(calls), **{"judge:" + k: v for k, v in judge_calls.items()}},
        "call_errors": dict(errors), "input_tokens": total_tokens["input_tokens"],
        "output_tokens": total_tokens["output_tokens"], "judge_tokens": dict(judge_tokens),
        "est_yuan": round(yuan(total_tokens), 4), "est_yuan_excl_judge": round(yuan(tokens), 4),
        "latency_ms": {"p50": percentile(latency, 50), "p90": percentile(latency, 90)},
        "wall_ms": {"p50": percentile(wall, 50), "p90": percentile(wall, 90)}}


def summarize_condition(rows: Sequence[dict[str, Any]], generative: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), "popular": summarize_popular(rows),
                           "non_popular": summarize_non_popular(rows), "cost": summarize_cost(rows)}
    if generative:
        out["generation_quality"] = gq.summarize([r for r in rows if r["intent"] == "popular_science"])
    return out


def comparison_table(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """三条件对照表（纯文本）。"""
    names = [n for n in CONDITIONS if n in summaries]

    def cell(path: Sequence[str], n: str) -> str:
        node: Any = summaries[n]
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
            if node is None:
                return "-"
        if isinstance(node, Mapping) and "count" in node:
            return f"{node['count']} ({node['rate']})"
        return str(node)

    rows = [
        ("Hit@3（可回答科普）", ("popular", "evidence_hit@3")),
        ("被路由拒答（可回答科普）", ("popular", "routed_out_answerable")),
        ("正确回答率", ("popular", "correct_answer")),
        ("作答但引用全不相关", ("popular", "wrong_answer_irrelevant_citation")),
        ("无答案被作答", ("popular", "wrong_answer_unanswerable_answered")),
        ("过度拒答", ("popular", "over_refusal")),
        ("  其中路由造成", ("popular", "over_refusal_by_router")),
        ("非科普未作答", ("non_popular", "not_answered")),
        ("非科普给出模板", ("non_popular", "template_given")),
        ("急症给出急症提示", ("non_popular", "emergency_prompt_given")),
        ("急症被作答", ("non_popular", "emergency_answered")),
        ("引用召回", ("generation_quality", "citation_recall")),
        ("引用精确率", ("generation_quality", "citation_precision")),
        ("忠实度", ("generation_quality", "faithfulness")),
        ("答案相关性(均值)", ("generation_quality", "answer_relevancy", "mean")),
        ("上下文精确率(均值)", ("popular", "context_precision", "mean")),
        ("降级为抽取式数", ("generation_quality", "degraded_extractive")),
        ("删句数", ("generation_quality", "sentences_dropped")),
        ("partial 句数", ("generation_quality", "sentences_partial")),
        ("模型调用总数", ("cost", "model_calls_total")),
        ("  路由/Agent/HyDE", None),
        ("  生成/蕴含/评审", None),
        ("输入/输出 token", None),
        ("估算费用(元,含评审)", ("cost", "est_yuan")),
        ("估算费用(元,不含评审)", ("cost", "est_yuan_excl_judge")),
        ("时延 p50(ms)", ("cost", "latency_ms", "p50")),
        ("时延 p90(ms)", ("cost", "latency_ms", "p90"))]
    lines = ["指标".ljust(26) + "".join(n.ljust(44) for n in names)]
    for label, path in rows:
        if path is None:
            kinds = {n: summaries[n]["cost"].get("model_calls_by_kind", {}) for n in names}
            if label.startswith("  路由"):
                cells = [f"{k.get('router', 0)}/{k.get('agent', 0)}/{k.get('hyde', 0)}" for k in kinds.values()]
            elif label.startswith("  生成"):
                cells = [f"{k.get('generation', 0)}/{k.get('entailment', 0)}/{k.get('judge', 0)}" for k in kinds.values()]
            else:
                cells = [f"{summaries[n]['cost']['input_tokens']}/{summaries[n]['cost']['output_tokens']}" for n in names]
        else:
            cells = [cell(path, n) for n in names]
        lines.append(label.ljust(26) + "".join(c.ljust(44) for c in cells))
    return "\n".join(lines)


# ----------------------------------------------------------------- 主流程

LIMITATIONS = [
    "v4 结果与 v2/v3 结果分开报告，不混算、不替换。",
    "相关性标注由 LLM 完成、未经人工复核（relevant 与 partial 都算相关）；引用块相关不等于医学正确。",
    "部分 v3 dev 查询落入 v4 test，对条件 1（v3 推荐配置）略有利，见 status M7 阶段 A。",
    "单次运行，不重复、不调参、不做消融；样本量小，只作工程对照，不作显著性结论。",
    "条件 3 复用条件 2 的路由与检索 Agent 调用记录，因此两者的检索与路由完全一致；时延按调用记录还原，口径见脚本说明。",
    "评审与被评系统同源（均为 MiMo），有偏差；判定不等于医学正确。",
    "估算费用口径同 M11（输入 1 元/百万、输出 2 元/百万 token），未核实实际计费。"]


def select_queries(split: str, limit: int) -> list[dict[str, Any]]:
    queries = [q for q in load_queries() if q.get("split") == split]
    return queries[:limit] if limit else queries


def cassette_paths(condition: str, smoke: bool) -> Path:
    return (SMOKE_DIR if smoke else CASSETTE_DIR) / (f"v4-final-{condition}.jsonl")


def run_condition(condition: str, args: argparse.Namespace, queries: Sequence[dict[str, Any]],
                  relevance: Mapping[str, set[str]], rags: dict[str, Any], smoke: bool) -> dict[str, Any]:
    import torch

    config = condition_config(condition)
    generative = config["generation"].get("answer_mode", "extractive") == "generative"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = json.dumps([config["retrieval"], config["dataset"]], sort_keys=True)
    if key not in rags:
        # 同一进程依次跑多个条件时只保留当前检索配置的索引，避免两套语料索引与模型同时驻留内存
        rags.clear()
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        rags[key] = build_rag(config, device)
        serialize_retrieval(rags[key])
    rag = rags[key]

    base_cassette = Cassette(cassette_paths("v4_extractive", smoke)) if condition == "v4_generative" else None
    if condition == "v4_generative" and (base_cassette is None or len(base_cassette) == 0):
        raise SystemExit("[final] 条件 3 需要先完成条件 2 的调用记录（只读复用其路由与检索 Agent 请求）")
    cassette = Cassette(cassette_paths(condition, smoke), base=base_cassette)
    if args.mode == "record":
        inner: Any = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        inner = ReplayTransport(cassette)

    own_before = len(cassette)
    started = time.time()
    rows = run_all(make_base(config, rag, args.model), queries, relevance, inner, cassette, args.model, generative,
                   args.workers, condition)
    if generative:
        from medidiag.rag.dense import load_local_encoder

        embedding = config["embedding"]
        encoder = load_local_encoder(embedding["model"], embedding["revision"], device=device,
                                     batch_size=64 if device == "cuda" else 8)
        gq.add_relevancy([r for r in rows], encoder)
    summary = summarize_condition(rows, generative)
    spend = inner.spent() if args.mode == "record" else None
    result = {"condition": condition, "config": CONFIG_FILES.get(condition, V3_CONFIG), "generative": generative,
              "summary": summary, "rows": rows,
              "run": {"mode": args.mode, "seconds": round(time.time() - started), "own_cassette_entries": len(cassette),
                      "new_entries_this_run": len(cassette) - own_before, "ledger_spend_cumulative": spend}}
    pop = summary["popular"]
    print(f"[{condition}] 完成 Hit@3 {pop['evidence_hit@3']['count']}，正确回答 {pop['correct_answer']['count']}，"
          f"模型调用 {summary['cost']['model_calls_total']}，调用错误 {summary['cost']['call_errors']}", flush=True)
    return result


def consistency_view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {r["sample_id"]: {k: v for k, v in r.items() if k not in TIME_KEYS} for r in rows}


def replay_check(recorded: Mapping[str, Any], replayed: Mapping[str, Any]) -> dict[str, Any]:
    before, after = consistency_view(recorded["rows"]), consistency_view(replayed["rows"])
    differing = [sid for sid in before if before[sid] != after.get(sid)]
    errors = sum(replayed["summary"]["cost"]["call_errors"].values())
    calls = replayed["summary"]["cost"]["model_calls_total"]
    return {"consistent": not differing and before.keys() == after.keys(), "differing": differing[:10],
            "n_differing": len(differing), "replay_call_errors": errors, "cassette_hit_rate": f"{calls - errors}/{calls}"}


def assemble(parts: Mapping[str, Mapping[str, Any]], mode: str, n_queries: int, model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "date": str(date.today()), "mode": mode, "model": model, "split": "test",
        "queries": n_queries, "conditions": list(parts),
        "params": {"v3_recommended_on_v4": "v3 推荐配置（向量检索 + 检索 Agent v1 + 重排候选 5 + 抽取式，无路由）换 v4 父子子块语料",
                   "v4_extractive": CONFIG_FILES["v4_extractive"], "v4_generative": CONFIG_FILES["v4_generative"],
                   "judge": {"nli_prompt": gq.NLI_VERSION, "question_prompt": gq.QGEN_VERSION},
                   "est_yuan_rate": {"input_per_m": YUAN_PER_M_PROMPT, "output_per_m": YUAN_PER_M_COMPLETION}},
        "summary": {n: p["summary"] for n, p in parts.items()},
        "run": {n: p["run"] for n, p in parts.items()}, "limitations": LIMITATIONS,
        "per_query": {n: p["rows"] for n, p in parts.items()}}


def part_path(split: str, mode: str, condition: str) -> Path:
    return PARTS_DIR / split / mode / f"{condition}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--conditions", default=",".join(CONDITIONS), help="逗号分隔，默认全部")
    parser.add_argument("--model", default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--max-calls", type=int, default=3500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只评前 N 条 dev 查询，不写报告；test 不允许")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions or any(c not in CONDITIONS for c in conditions):
        raise SystemExit(f"[final] --conditions 只能取 {CONDITIONS}")
    smoke = bool(args.limit)
    if smoke and args.split != "dev":
        raise SystemExit("[final] test 只跑一次全量，冒烟只能用 --split dev --limit N")
    if args.split == "dev" and not smoke:
        raise SystemExit("[final] dev 仅用于冒烟，请带 --limit")
    if smoke:  # 冒烟用独立的账本与调用记录，不污染正式记录
        args.ledger = SMOKE_DIR / "ledger.db"
        args.max_calls = 300
    for path in (CHUNKS_PATH, QUERIES_PATH, JUDGMENTS_PATH):
        if not path.exists():
            raise SystemExit(f"[final] 缺少 {path}")

    from medidiag.schemas import read_jsonl

    queries = select_queries(args.split, args.limit)
    popular = [q for q in queries if q["intent"] == "popular_science"]
    print(f"[load] {args.split} 查询 {len(queries)} 条（科普 {len(popular)}，可回答 "
          f"{sum(q.get('answerable') is True for q in popular)}）", flush=True)
    relevance = build_relevance(popular, read_jsonl(CHUNKS_PATH))
    empty = [q["sample_id"] for q in popular if q.get("answerable") is True and not relevance[q["sample_id"]]]
    print(f"[load] 可回答科普题中没有相关块标注的 {len(empty)} 条（按未命中计）", flush=True)
    import torch

    torch.set_num_threads(2)

    rags: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        results[condition] = run_condition(condition, args, queries, relevance, rags, smoke)
        if not smoke:
            path = part_path(args.split, args.mode, condition)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(results[condition], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if smoke:
        print(comparison_table({n: r["summary"] for n, r in results.items()}))
        return
    parts = {n: json.loads(part_path(args.split, args.mode, n).read_text(encoding="utf-8"))
             for n in CONDITIONS if part_path(args.split, args.mode, n).exists()}
    print(comparison_table({n: p["summary"] for n, p in parts.items()}))
    if len(parts) < len(CONDITIONS):
        print(f"[final] 已完成 {list(parts)}，其余条件跑完后自动汇总报告", flush=True)
        return
    report_path = REPORT_DIR / f"final-comparison-v4-test-{date.today():%Y%m%d}.json"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "record":
        report = assemble(parts, "record", len(queries), args.model)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[done] 报告 {report_path.relative_to(ROOT)}", flush=True)
        return
    recorded = {n: json.loads(part_path("test", "record", n).read_text(encoding="utf-8")) for n in CONDITIONS
                if part_path("test", "record", n).exists()}
    check = {n: replay_check(recorded[n], parts[n]) for n in parts if n in recorded}
    print(json.dumps(check, ensure_ascii=False, indent=2))
    existing = sorted(REPORT_DIR.glob("final-comparison-v4-test-*.json"))
    if existing and all(c["consistent"] for c in check.values()) and len(check) == len(CONDITIONS):
        target = existing[-1]
        report = json.loads(target.read_text(encoding="utf-8"))
        report["replay_check"] = check
        report["replay_latency_ms"] = {n: p["summary"]["cost"]["latency_ms"] for n, p in parts.items()}
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[done] 回放一致，已写入 {target.relative_to(ROOT)}", flush=True)
    else:
        print("[final] 回放与 record 不一致或缺少 record 部分，不覆盖报告，请排查", flush=True)


if __name__ == "__main__":
    main()
