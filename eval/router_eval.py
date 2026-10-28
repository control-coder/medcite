"""v4 dev 查询路由与元数据过滤评测（M10）。

只跑 v4 dev 60 条：

1. 路由：规则级 + 一次 LLM 调用（``query-router-v2``，``--prompt-version v1`` 回放 M10 的旧记录），输出 gold 意图 × 预测意图的混淆矩阵（标明规则 / LLM / 降级来源）；
   急症必须全部召回。
2. 过滤抽取：带 ``expected_filters`` 的查询上，预测过滤条件与期望条件逐键对照（主题按文档集合有交集算命中，
   周期的 latest 解析为语料最大月份）；另统计不该有过滤条件的科普查询上出现的多余条件。
3. 检索：在带 ``expected_filters`` 且 answerable 的 dev 科普查询上，用向量检索（parent_child 子块、top3、无重排）
   对比「不过滤 / 预测过滤 / gold 过滤」的 Hit@3（过滤后不足 3 条按顺序放宽重查）；
   并在全部 answerable 科普查询上对比「不过滤 / 预测过滤」，检查多余过滤是否伤害不带条件的问题。

``record`` 需要真实凭据并占用持久预算账本；``replay`` 只读调用记录，绝不联网，并与已写入的 record 报告对照结果是否一致。
标注读取沿用 ``eval.v4_judgments`` 的句子级标注（relevant / partial 都算命中）。

用法::

    python -m eval.router_eval --mode record --limit 4   # 冒烟，不写报告
    python -m eval.router_eval --mode record             # 全量
    python -m eval.router_eval --mode replay             # 回放并核对
    python -m eval.router_eval --mode replay --prompt-version v1   # 回放 v1 旧记录，只核对不写报告

v2 默认使用 ``eval/cassettes/v4-router-v2.jsonl``、账本 ``.cache/v4-router-v2-ledger.db``，报告
``router-v2-v4-<日期>.json`` 内带与 v1 报告的并列对照（``comparison``）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, cast

# 向量模型只读本地缓存，不联网探测（须在 transformers/sentence_transformers 导入前设置）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.hybrid_matrix import (  # noqa: E402
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    JUDGMENTS_PATH,
    QUERIES_PATH,
    _load_judgments_module,
    load_queries,
)
from eval.llm_pipeline_eval import call_with_retry, make_provider  # noqa: E402
from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport  # noqa: E402
from medidiag.llm.models import ACTIVE_MIMO_MODEL, ALLOWED_MIMO_MODELS  # noqa: E402
from medidiag.rag.metadata import MetadataIndex, search_with_relaxation  # noqa: E402
from medidiag.workflow.query_router import (  # noqa: E402
    INTENTS,
    PROMPT_VERSIONS,
    RouteDecision,
    route_query,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "router-eval-v1"
REPORT_DIR = ROOT / "artifacts/reports/retrieval"
TOP_K = 3
COMPARED_KEYS = ("topic", "source_type", "period")  # expected_filters 里出现的键；section_type 没有 gold，不对照
# 费用估算口径同 M3：每百万输入 Token 1 元、输出 2 元（未核实实际计费，仅供量级参考）
YUAN_PER_M_PROMPT = 1.0
YUAN_PER_M_COMPLETION = 2.0


class _CountingLLM:
    """包装 provider：可重试错误自动重试，统计调用数与错误码（回放时错误即调用记录未命中）。"""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.calls = 0
        self.errors: Counter[str] = Counter()
        self._lock = threading.Lock()

    def generate(self, request: Any, *, timeout_s: float, idempotency_key: str) -> Any:
        with self._lock:
            self.calls += 1
        try:
            return call_with_retry(lambda: self.provider.generate(
                request, timeout_s=timeout_s, idempotency_key=idempotency_key))
        except MediDiagError as exc:
            with self._lock:
                self.errors[exc.code] += 1
            raise


def route_all(queries: Sequence[dict[str, Any]], llm: _CountingLLM, index: MetadataIndex, model: str,
              workers: int, prompt_version: str) -> dict[str, RouteDecision]:
    """对每条查询路由；按 10 条或 30 秒打印进度。"""
    decisions: dict[str, RouteDecision] = {}
    started = last_print = time.time()

    def one(query: dict[str, Any]) -> tuple[str, RouteDecision]:
        return query["sample_id"], route_query(query["question"], llm=cast(Any, llm), metadata=index, model=model,
                                               prompt_version=prompt_version)

    with ThreadPoolExecutor(workers) as pool:
        for done, (sample_id, decision) in enumerate(pool.map(one, queries), 1):
            decisions[sample_id] = decision
            if done % 10 == 0 or done == len(queries) or time.time() - last_print > 30:
                last_print = time.time()
                print(f"[route] {done}/{len(queries)}（{last_print - started:.0f}s，LLM 调用 {llm.calls}，"
                      f"错误 {dict(llm.errors)}）", flush=True)
    return decisions


def confusion(queries: Sequence[dict[str, Any]], decisions: dict[str, RouteDecision]) -> dict[str, Any]:
    """gold 意图 × 预测意图，每格再按来源（rule / llm / fallback）拆分。"""
    matrix: dict[str, dict[str, dict[str, int]]] = {}
    for query in queries:
        decision = decisions[query["sample_id"]]
        cell = matrix.setdefault(query["intent"], {}).setdefault(decision.intent, {})
        cell[decision.source] = cell.get(decision.source, 0) + 1
    correct = sum(1 for q in queries if decisions[q["sample_id"]].intent == q["intent"])
    emergency = [q for q in queries if q["intent"] == "emergency"]
    caught = [q["sample_id"] for q in emergency if decisions[q["sample_id"]].intent == "emergency"]
    return {"matrix": matrix, "accuracy": f"{correct}/{len(queries)}",
            "emergency_recall": f"{len(caught)}/{len(emergency)}",
            "emergency_missed": [q["sample_id"] for q in emergency if q["sample_id"] not in caught],
            "source_counts": dict(Counter(decisions[q["sample_id"]].source for q in queries)),
            "rule_hits": dict(Counter(d.rule for d in decisions.values() if d.rule)),
            "fallbacks": {sid: d.fallback_reason for sid, d in decisions.items() if d.source == "fallback"}}


def over_refusal(queries: Sequence[dict[str, Any]], decisions: dict[str, RouteDecision]) -> dict[str, Any]:
    """gold 为科普的查询被路由拒答（个体诊疗 / 超范围 / 急症）的数量与编号，按预测意图拆分。"""
    ids = {q["sample_id"]: decisions[q["sample_id"]].intent for q in queries
           if q["intent"] == "popular_science" and decisions[q["sample_id"]].refused}
    return {"count": len(ids), "of": sum(1 for q in queries if q["intent"] == "popular_science"),
            "by_predicted": dict(Counter(ids.values())), "ids": ids}


def over_refusal_from_routes(report: dict[str, Any]) -> dict[str, Any]:
    """旧报告没有 over_refusal 字段时，从逐条路由与 gold 意图还原（v1 报告用）。"""
    gold = {q["sample_id"]: q["intent"] for q in load_queries() if q.get("split") == "dev"}
    ids = {sid: r["intent"] for sid, r in report["routes"].items()
           if gold.get(sid) == "popular_science" and r["refused"]}
    return {"count": len(ids), "of": sum(1 for v in gold.values() if v == "popular_science"),
            "by_predicted": dict(Counter(ids.values())), "ids": ids}


def compare_with_v1(report: dict[str, Any], v1: dict[str, Any]) -> dict[str, Any]:
    """v2 与 v1 报告并列：准确率、混淆矩阵、急症召回、科普误拒、Hit@3；v1 报告不改动。"""
    def side(r: dict[str, Any]) -> dict[str, Any]:
        retrieval = r["retrieval"]["all_answerable"]
        variants = r.get("filter_variants", {}).get("all_answerable", {})
        return {"prompt_version": r["prompt_version"], "accuracy": r["routing"]["accuracy"],
                "matrix": r["routing"]["matrix"], "emergency_recall": r["routing"]["emergency_recall"],
                "popular_over_refusal": r.get("over_refusal") or over_refusal_from_routes(r),
                "hit@3_n": retrieval["n"], "hit@3_no_filter": retrieval["none"]["hit@3"],
                "hit@3_applied_filters_source_type_period": retrieval["predicted"]["hit@3"],
                "hit@3_variants": {k: variants[k]["hit@3"] for k in variants if k != "n"},
                "filters_key_match_rate": r["filters"]["key_match_rate"], "cost": r["cost"]}
    return {"v1": side(v1), "v2": side(report),
            "note": "同一批 v4 dev 60 条；v2 只跑一次。M15 test 上的路由数字基于 v1，未用 v2 重跑。"}


def compare_filters(expected: dict[str, Any], decision: RouteDecision, index: MetadataIndex) -> dict[str, Any]:
    """逐键对照期望与预测的过滤条件。"""
    gold, _ = index.validate(expected)
    keys: dict[str, str] = {}
    for key in COMPARED_KEYS:
        if key not in expected:
            continue
        want, got = gold.get(key), decision.filters.get(key)
        if got is None:
            keys[key] = "missing"
        elif want is None:
            keys[key] = "gold_unresolvable"
        elif key == "topic":
            keys[key] = "match" if index.resolve_topic(want) & index.resolve_topic(got) else "mismatch"
        else:
            keys[key] = "match" if want == got else "mismatch"
    extra = [k for k in COMPARED_KEYS if k in decision.filters and k not in expected]
    return {"expected": expected, "predicted": decision.filters, "dropped": decision.dropped_filters,
            "keys": keys, "extra": extra}


def filter_summary(queries: Sequence[dict[str, Any]], decisions: dict[str, RouteDecision],
                   index: MetadataIndex) -> dict[str, Any]:
    with_expected = [q for q in queries if q.get("expected_filters")]
    rows = {q["sample_id"]: compare_filters(q["expected_filters"], decisions[q["sample_id"]], index)
            for q in with_expected}
    key_counts = Counter(status for row in rows.values() for status in row["keys"].values())
    all_match = [sid for sid, row in rows.items() if row["keys"] and set(row["keys"].values()) == {"match"}]
    clean = [sid for sid in all_match if not rows[sid]["extra"]]
    spurious = [q["sample_id"] for q in queries
                if q["intent"] == "popular_science" and not q.get("expected_filters")
                and any(k in decisions[q["sample_id"]].filters for k in COMPARED_KEYS)]
    section = {q["sample_id"]: decisions[q["sample_id"]].filters["section_type"]
               for q in queries if "section_type" in decisions[q["sample_id"]].filters}
    n_plain = sum(1 for q in queries if q["intent"] == "popular_science" and not q.get("expected_filters"))
    return {"queries_with_expected_filters": len(with_expected),
            "key_status": dict(key_counts),
            "key_match_rate": f"{key_counts['match']}/{sum(key_counts.values())}",
            "queries_all_keys_matched": f"{len(all_match)}/{len(with_expected)}",
            "queries_all_matched_without_extra": f"{len(clean)}/{len(with_expected)}",
            "spurious_filter_queries": {"count": len(spurious), "of": n_plain, "ids": spurious},
            "predicted_section_type": section, "per_query": rows}


def build_relevance(picked: Sequence[dict[str, Any]], raw_chunks: list[dict[str, Any]]) -> dict[str, set[str]]:
    """每条查询的正例子块集合（relevant 与 partial），投影方式同 eval.hybrid_matrix。"""
    load_judgments, chunk_labels_for_query, recover_sentence_texts = _load_judgments_module()
    from medidiag.schemas import read_jsonl

    judgments = load_judgments(JUDGMENTS_PATH)
    strategies = {"parent_child": raw_chunks}
    recursive_path = CHUNKS_PATH.parent.parent / "recursive" / "chunks.jsonl"
    if recursive_path.exists():
        strategies["recursive"] = read_jsonl(recursive_path)
    sentence_texts = recover_sentence_texts(strategies, judgments)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in raw_chunks:
        by_doc.setdefault(str(chunk.get("metadata", {}).get("doc_id") or chunk["source_id"]), []).append(chunk)
    out: dict[str, set[str]] = {}
    for query in picked:
        qid = query["sample_id"]
        doc_chunks = [c for doc_id in judgments.get(qid, {}) for c in by_doc.get(doc_id, [])]
        labels = chunk_labels_for_query(qid, doc_chunks, judgments, sentence_texts)
        out[qid] = {cid for cid, label in labels.items() if label != "irrelevant"}
    return out


def build_dense(chunks: list[Any], device: str) -> Any:
    """向量检索器：GPU 编码，向量缓存到 .cache/index/（同语料同模型版本不重复编码）。"""
    from medidiag.rag.dense import INDEX_DIR, DenseRetriever, load_local_encoder
    from medidiag.rag.store import cache_base

    encoder = load_local_encoder(EMBEDDING_MODEL, EMBEDDING_REVISION, device=device,
                                 batch_size=64 if device == "cuda" else 8)
    dense = DenseRetriever(chunks, encoder=encoder, model_name=EMBEDDING_MODEL, model_revision=EMBEDDING_REVISION,
                           store_base=cache_base(INDEX_DIR, chunks, EMBEDDING_REVISION))
    started = time.time()
    dense.build_index()
    print(f"[dense] 索引就绪（缓存命中={dense.cache_hit}，{time.time() - started:.0f}s）", flush=True)
    return dense


def retrieval_hit(dense: Any, index: MetadataIndex, question: str, filters: dict[str, Any] | None,
                  positives: set[str]) -> dict[str, Any]:
    """一次带过滤（不足 top3 时放宽）的向量检索，返回是否命中与放宽记录。"""
    if filters:
        results, trace = search_with_relaxation(
            lambda allowed: dense.search(question, top_k=TOP_K, allowed_ids=allowed), index, filters, TOP_K)
        info: dict[str, Any] = {"applied": trace.applied, "relaxed": trace.relaxed, "dropped": trace.dropped}
    else:
        results = dense.search(question, top_k=TOP_K)
        info = {"applied": {}, "relaxed": [], "dropped": []}
    ids = [r.chunk_id for r in results]
    return {"hit": any(cid in positives for cid in ids), "chunk_ids": ids, **info}


def retrieval_comparison(queries: Sequence[dict[str, Any]], decisions: dict[str, RouteDecision],
                         index: MetadataIndex, dense: Any, relevance: dict[str, set[str]]) -> dict[str, Any]:
    """答得出的科普查询上对比 不过滤 / 预测过滤 / gold 过滤 的 Hit@3。"""
    answerable = [q for q in queries if q["intent"] == "popular_science" and q.get("answerable") is True]
    with_filters = [q for q in answerable if q.get("expected_filters")]
    rows: dict[str, Any] = {}
    for count, query in enumerate(answerable, 1):
        qid, question = query["sample_id"], query["question"]
        decision = decisions[qid]
        row: dict[str, Any] = {"none": retrieval_hit(dense, index, question, None, relevance[qid])}
        # 路由拒答的科普查询在应用里不会检索，按未命中计；单独标出便于区分过滤与误拒
        row["predicted"] = ({"hit": False, "routed_out": decision.intent, "chunk_ids": []} if decision.refused
                            else retrieval_hit(dense, index, question, decision.filters, relevance[qid]))
        if query.get("expected_filters"):
            row["gold"] = retrieval_hit(dense, index, question, query["expected_filters"], relevance[qid])
        rows[qid] = row
        if count % 10 == 0 or count == len(answerable):
            print(f"[retrieval] {count}/{len(answerable)}", flush=True)

    def rate(ids: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
        hits = sum(1 for q in ids if rows[q["sample_id"]][arm]["hit"])
        return {"hit@3": f"{hits}/{len(ids)}", "rate": round(hits / len(ids), 4) if ids else None}

    wins = [qid for qid in rows if "gold" in rows[qid] and rows[qid]["gold"]["hit"] and not rows[qid]["none"]["hit"]]
    losses = [qid for qid in rows if "gold" in rows[qid] and rows[qid]["none"]["hit"] and not rows[qid]["gold"]["hit"]]
    return {
        "with_expected_filters": {
            "n": len(with_filters), "none": rate(with_filters, "none"),
            "predicted": rate(with_filters, "predicted"), "gold": rate(with_filters, "gold"),
            "gold_gained_vs_none": wins, "gold_lost_vs_none": losses,
            "queries_without_positive_chunks": [q["sample_id"] for q in with_filters if not relevance[q["sample_id"]]]},
        "all_answerable": {"n": len(answerable), "none": rate(answerable, "none"),
                           "predicted": rate(answerable, "predicted"),
                           "routed_out": [q["sample_id"] for q in answerable if decisions[q["sample_id"]].refused]},
        "per_query": rows}


VARIANTS: dict[str, tuple[str, ...]] = {
    "all_keys": ("topic", "section_type", "source_type", "period"),
    "no_section_type": ("topic", "source_type", "period"),
    "source_period_only": ("source_type", "period"),
}


def filter_variants(queries: Sequence[dict[str, Any]], decisions: dict[str, RouteDecision],
                    index: MetadataIndex, dense: Any, relevance: dict[str, set[str]]) -> dict[str, Any]:
    """预测过滤的键子集变体（不新增 LLM 调用）：全部键 / 去掉 section_type / 只留 source_type 与 period。

    路由拒答的科普题按未命中计，口径同 ``retrieval_comparison``。
    """
    answerable = [q for q in queries if q["intent"] == "popular_science" and q.get("answerable") is True]
    hits: dict[str, dict[str, bool]] = {name: {} for name in VARIANTS}
    for count, query in enumerate(answerable, 1):
        qid = query["sample_id"]
        decision = decisions[qid]
        for name, keys in VARIANTS.items():
            if decision.refused:
                hits[name][qid] = False
                continue
            subset = {k: v for k, v in decision.filters.items() if k in keys}
            hits[name][qid] = retrieval_hit(dense, index, query["question"], subset, relevance[qid])["hit"]
        if count % 10 == 0 or count == len(answerable):
            print(f"[variants] {count}/{len(answerable)}", flush=True)

    def rate(ids: Sequence[str], name: str) -> dict[str, Any]:
        n_hit = sum(hits[name][i] for i in ids)
        return {"hit@3": f"{n_hit}/{len(ids)}", "rate": round(n_hit / len(ids), 4) if ids else None}

    all_ids = [q["sample_id"] for q in answerable]
    cond_ids = [q["sample_id"] for q in answerable if q.get("expected_filters")]
    return {"note": "同一批预测过滤的键子集对照，不新增 LLM 调用；原数字见 retrieval 字段。",
            "variants": {name: list(keys) for name, keys in VARIANTS.items()},
            "all_answerable": {"n": len(all_ids), **{name: rate(all_ids, name) for name in VARIANTS}},
            "with_expected_filters": {"n": len(cond_ids), **{name: rate(cond_ids, name) for name in VARIANTS}}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from medidiag.schemas import load_knowledge_chunks, read_jsonl

    for path in (CHUNKS_PATH, QUERIES_PATH, JUDGMENTS_PATH):
        if not path.exists():
            raise SystemExit(f"[router_eval] 缺少 {path}")
    queries = [q for q in load_queries() if q.get("split") == "dev"]
    if args.limit:
        queries = queries[: args.limit]
    chunks = load_knowledge_chunks(CHUNKS_PATH)
    index = MetadataIndex(chunks)
    print(f"[load] dev 查询 {len(queries)} 条，子块 {len(chunks)}，统计周期 {index.periods[0]}~{index.periods[-1]}",
          flush=True)

    cassette = Cassette(args.cassette)
    if args.mode == "record":
        transport: Any = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        transport = ReplayTransport(cassette)
    llm = _CountingLLM(make_provider(transport))
    prompt_version = PROMPT_VERSIONS[args.prompt_version]
    decisions = route_all(queries, llm, index, args.model, args.workers, prompt_version)

    routing = confusion(queries, decisions)
    filters = filter_summary(queries, decisions, index)
    tokens: Counter[str] = Counter()
    for decision in decisions.values():
        for name in ("input_tokens", "output_tokens"):
            tokens[name] += int(decision.usage.get(name) or 0)
    llm_decisions = [d for d in decisions.values() if d.source != "rule"]
    cost = {"llm_calls": llm.calls, "llm_decisions": len(llm_decisions),
            "call_errors": dict(llm.errors), "input_tokens": tokens["input_tokens"],
            "output_tokens": tokens["output_tokens"],
            "est_yuan": round((tokens["input_tokens"] * YUAN_PER_M_PROMPT
                               + tokens["output_tokens"] * YUAN_PER_M_COMPLETION) / 1e6, 4)}
    if args.mode == "record":
        cost["spend"] = transport.spent()
    print(f"[routing] 准确率 {routing['accuracy']}，急症召回 {routing['emergency_recall']}，"
          f"来源 {routing['source_counts']}", flush=True)
    print(f"[filters] 键命中 {filters['key_match_rate']}，全键命中 {filters['queries_all_keys_matched']}，"
          f"多余过滤 {filters['spurious_filter_queries']['count']}", flush=True)

    import torch

    torch.set_num_threads(2)
    raw_chunks = read_jsonl(CHUNKS_PATH)
    relevance = build_relevance([q for q in queries if q["intent"] == "popular_science"], raw_chunks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense = build_dense(chunks, device)
    retrieval = retrieval_comparison(queries, decisions, index, dense, relevance)
    print(f"[retrieval] 带条件查询 {json.dumps({k: retrieval['with_expected_filters'][k] for k in ('n', 'none', 'predicted', 'gold')}, ensure_ascii=False)}",
          flush=True)

    variants = filter_variants(queries, decisions, index, dense, relevance)
    print("[variants] " + json.dumps({k: variants[k] for k in ("all_answerable", "with_expected_filters")},
                                     ensure_ascii=False), flush=True)

    return {
        "filter_variants": variants,
        "schema_version": SCHEMA_VERSION, "date": str(date.today()), "mode": args.mode, "model": args.model,
        "prompt_version": prompt_version, "split": "dev", "queries": len(queries),
        "intents": list(INTENTS), "cost": cost, "routing": routing, "filters": filters, "retrieval": retrieval,
        "params": {"retrieval": "dense bge-small-zh parent_child top3 无重排", "top_k": TOP_K,
                   "relax_order": ["section_type", "topic", "source_type", "period"]},
        "note": "标注由 LLM 完成、未经人工复核；Hit@3 的 relevant 与 partial 都算命中；样本量小，只作工程对照。",
        "routes": {sid: d.to_dict() for sid, d in decisions.items()},
        "over_refusal": over_refusal(queries, decisions),
        "emergency_hints": [sid for sid, d in decisions.items() if d.emergency_hint],
    }


def replay_check(report: dict[str, Any], path: Path) -> dict[str, Any]:
    """回放结果与已写入的 record 报告对照：路由决定与检索命中必须一致。"""
    if not path.exists():
        return {"checked": False, "reason": "没有找到 record 报告"}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(report, ensure_ascii=False))  # 与落盘报告同一序列化形态再比较
    keys = ("routes", "routing", "filters", "retrieval")
    same = {key: recorded.get(key) == current[key] for key in keys}
    calls = report["cost"]["llm_calls"]
    errors = sum(report["cost"]["call_errors"].values())
    return {"checked": True, "consistent": all(same.values()), "sections": same,
            "cassette_hit_rate": f"{calls - errors}/{calls}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", choices=ALLOWED_MIMO_MODELS, default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--prompt-version", choices=sorted(PROMPT_VERSIONS), default="v2",
                        help="路由提示词版本；v1 只用于回放 M10 的调用记录，回放时不改动旧报告")
    parser.add_argument("--cassette", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只评前 N 条 dev 查询，不写报告")
    args = parser.parse_args()
    v1 = args.prompt_version == "v1"
    args.cassette = args.cassette or ROOT / ("eval/cassettes/v4-router.jsonl" if v1 else "eval/cassettes/v4-router-v2.jsonl")
    args.ledger = args.ledger or ROOT / (".cache/v4-m10-ledger.db" if v1 else ".cache/v4-router-v2-ledger.db")
    args.max_calls = args.max_calls or (150 if v1 else 100)

    report = run(args)
    path = REPORT_DIR / (f"router-v4-{date.today():%Y%m%d}.json" if v1 else f"router-v2-v4-{date.today():%Y%m%d}.json")
    if v1 and args.mode == "replay" and not args.limit:  # v1 回放只核对，不改动旧报告
        print(json.dumps(replay_check(report, path), ensure_ascii=False, indent=2))
        return
    v1_path = REPORT_DIR / "router-v4-20261028.json"
    if not v1 and not args.limit and args.mode == "record" and v1_path.exists():
        report["comparison"] = compare_with_v1(report, json.loads(v1_path.read_text(encoding="utf-8")))
    if args.limit:
        print(json.dumps({k: report[k] for k in ("queries", "cost", "routing")}, ensure_ascii=False, indent=2))
        return
    if args.mode == "replay":
        report["replay_check"] = replay_check(report, path)
        print(json.dumps(report["replay_check"], ensure_ascii=False, indent=2))
        if report["replay_check"].get("checked"):
            # 一致时保留 record 报告并附上核对结果；不一致时不覆盖，交给人排查
            if report["replay_check"]["consistent"]:
                recorded = json.loads(path.read_text(encoding="utf-8"))
                recorded["replay_check"] = report["replay_check"]
                recorded["filter_variants"] = report["filter_variants"]
                path.write_text(json.dumps(recorded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in ("queries", "cost")}
    summary["routing"] = {k: v for k, v in report["routing"].items() if k != "matrix"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] 报告 {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
