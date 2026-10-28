"""生成式回答（M12）真实冒烟：10 道 v4 dev 题走应用 provider 的完整阶段（检索→计划→生成→仲裁→审核→报告）。

题目：v4 dev 中 intent=popular_science 且 answerable=true 的题按序号等距取 8 道，加 1 道 answerable=false 的科普题
和 1 道带剂量风险的题（dev 里没有带剂量的科普题，取 individual_care 的退烧药剂量题）。配置 ``configs/application_v4_generative.yaml``
（混合检索 + 重排 + 生成式回答，固定链路）。这是冒烟，不是评测：不调提示词、不做指标对照。

``record`` 需要真实凭据并占用持久预算账本（封顶 60 次）；所有调用写调用记录。``replay`` 只读调用记录，不联网，
并核对与 record 报告的每题结果一致。

用法::

    python -m eval.generative_smoke --mode record
    python -m eval.generative_smoke --mode replay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport  # noqa: E402
from medidiag.llm.models import ACTIVE_MIMO_MODEL  # noqa: E402
from medidiag.llm.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from medidiag.llm.profiles import get_provider_profile  # noqa: E402
from medidiag.workflow.generative_answer import ENTAILMENT_VERSION, PROMPT_VERSION  # noqa: E402
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/application_v4_generative.yaml"
QUERIES = ROOT / "examples/public_health_v4/queries.jsonl"
CASSETTE = ROOT / "eval/cassettes/v4-generative-smoke.jsonl"
LEDGER = ROOT / ".cache/v4-m12-ledger.db"
REPORT_DIR = ROOT / "artifacts/reports/generation"
MAX_CALLS = 60
SCHEMA_VERSION = "generative-smoke-v1"
CALL_KINDS = {PROMPT_VERSION: "generation", ENTAILMENT_VERSION: "entailment", "mimo-grounded-v1": "extractive"}
# 不进入一致性核对的字段（耗时随运行变化）
VOLATILE = {"latency_ms"}


class CountingLLM:
    """一道题专用的模型入口：按调用类型统计次数、错误码和 token。"""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.calls: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()

    def generate(self, request: Any, *, timeout_s: float, idempotency_key: str) -> Any:
        self.calls[CALL_KINDS.get(request.prompt_version, request.prompt_version)] += 1
        try:
            result = self.provider.generate(request, timeout_s=timeout_s, idempotency_key=idempotency_key)
        except MediDiagError as exc:
            self.errors[exc.code] += 1
            raise
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            self.tokens[name] += int((result.usage or {}).get(name) or 0)
        return result


def select_queries() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in QUERIES.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev = sorted((r for r in rows if r.get("split") == "dev"), key=lambda r: r["sample_id"])
    answerable = [r for r in dev if r["intent"] == "popular_science" and r["answerable"] is True]
    stride = len(answerable) // 8
    chosen = [{**r, "category": "answerable"} for r in answerable[::stride][:8]]
    by_id = {r["sample_id"]: r for r in dev}
    chosen.append({**by_id["public_q15"], "category": "unanswerable"})
    chosen.append({**by_id["v4_a58"], "category": "dosage_risk"})
    return chosen


def build_rag(device: str) -> tuple[Any, dict[str, Any]]:
    from medidiag.rag.runtime import RuntimeMedicalRAG, _hybrid_factory
    from medidiag.workflow.application import _rerank_factory, load_application_config

    config = load_application_config(ROOT / CONFIG)
    config["runtime"] = {"device": device, "batch_size": 64 if device == "cuda" else 8}
    started = time.time()
    rag = RuntimeMedicalRAG.from_config(config, root=ROOT, retriever_factory=_hybrid_factory,
                                        reranker_factory=_rerank_factory)
    print(f"[rag] 检索就绪（{device}，子块 {len(rag.chunks)}，{time.time() - started:.0f}s）", flush=True)
    return rag, config


def run_query(rag: Any, config: dict[str, Any], query: dict[str, Any], transport: Any) -> dict[str, Any]:
    counter = CountingLLM(OpenAICompatibleProvider(get_provider_profile("mimo_v25"), post=transport,
                                                   max_retries=0, max_structured_retries=0))
    generation_cfg = config["generation"]
    provider = MimoGroundedWorkflowProvider(rag, llm=counter, on_invalid=generation_cfg["on_invalid"],  # type: ignore[arg-type]
                                            answer_mode=generation_cfg["answer_mode"])
    started = time.perf_counter()
    row: dict[str, Any] = {"sample_id": query["sample_id"], "category": query["category"],
                           "answerable": query["answerable"], "question": query["question"], "error": None}
    try:
        question = provider.normalize(query["question"])["normalized_query"]
        retrieval = provider.retrieve(question)
        plan = provider.plan(question, retrieval)
        generation = provider.generate(question, retrieval, plan).payload
        arbitration = provider.arbitrate(generation, retrieval)
        review = provider.review(generation, arbitration, retrieval)
        report = provider.report("smoke_" + query["sample_id"], generation, review)
    except MediDiagError as exc:
        row["error"] = exc.code
        generation, review, report, retrieval = {"claims": [], "answer_mode": None}, {}, {}, {"chunks": []}
    trace = generation.get("generative_trace", {})
    shown_generative = generation.get("answer_mode") == "generative" and bool(generation.get("claims"))
    if row["error"]:
        outcome = "error"
    elif generation.get("degraded_from") == "generative":
        outcome = "degraded_extractive"
    elif shown_generative:
        outcome = "generative"
    else:
        outcome = "refused"
    row.update(
        outcome=outcome, evidence_ids=[c["chunk_id"] for c in retrieval["chunks"]], answer_mode=report.get("answer_mode"),
        sentences=[{"text": c["text"], "entailment": c.get("entailment"), "suspicious": c.get("suspicious"),
                    "citations": c.get("citations", [])} for c in generation["claims"]] if outcome != "degraded_extractive" else [],
        extractive_claims=[c["text"] for c in generation["claims"]] if outcome == "degraded_extractive" else [],
        dropped=trace.get("dropped", []), invalid_reasons=trace.get("invalid_reasons", []),
        entailment_error=trace.get("entailment_error"), degraded=trace.get("degraded"),
        review_verdict=review.get("verdict"), report_title=report.get("title"),
        model_calls=dict(counter.calls), call_errors=dict(counter.errors), tokens=dict(counter.tokens),
        latency_ms=round((time.perf_counter() - started) * 1000))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(r["outcome"] for r in rows)
    shown = [r for r in rows if r["outcome"] == "generative"]
    retries = sum(max(0, r["model_calls"].get("generation", 0) - 1) for r in rows)
    tokens = Counter()
    for r in rows:
        tokens.update(r["tokens"])
    return {
        "n": len(rows), "outcomes": dict(outcomes),
        "generative_shown": len(shown),
        "avg_sentences_shown": round(sum(len(r["sentences"]) for r in shown) / len(shown), 2) if shown else None,
        "sentences_dropped": sum(len(r["dropped"]) for r in rows),
        "partial_sentences": sum(1 for r in rows for s in r["sentences"] if s["entailment"] == "partial"),
        "degraded_extractive": outcomes["degraded_extractive"], "refused": outcomes["refused"],
        "validation_failures": sum(len(r["invalid_reasons"]) for r in rows), "generation_retries": retries,
        "entailment_errors": sum(1 for r in rows if r["entailment_error"]),
        "review_not_approved": sum(1 for r in rows if r["review_verdict"] not in (None, "APPROVED")),
        "model_calls": dict(sum((Counter(r["model_calls"]) for r in rows), Counter())), "tokens": dict(tokens)}


def comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows]


def ledger_usage() -> dict[str, Any]:
    import sqlite3

    with sqlite3.connect(LEDGER) as db:
        n, prompt, completion = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0) FROM calls").fetchone()
    return {"calls": n, "prompt_tokens": prompt, "completion_tokens": completion, "cap": MAX_CALLS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("record", "replay"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=None, help="报告路径；replay 用它读取 record 报告")
    args = parser.parse_args()
    report_path = args.report or REPORT_DIR / f"generative-smoke-v4-{date.today():%Y%m%d}.json"
    queries = select_queries()
    rag, config = build_rag(args.device)
    if args.mode == "record":
        transport: Any = RecordingTransport(CASSETTE, LEDGER, max_calls=MAX_CALLS)
    else:
        transport = ReplayTransport(Cassette(CASSETTE))
    rows = []
    for i, query in enumerate(queries, 1):
        rows.append(run_query(rag, config, query, transport))
        print(f"[{i}/{len(queries)}] {query['sample_id']} {rows[-1]['outcome']} 调用 {rows[-1]['model_calls']}", flush=True)
    summary = summarize(rows)
    if args.mode == "record":
        report = {"schema_version": SCHEMA_VERSION, "config": CONFIG, "model": ACTIVE_MIMO_MODEL, "date": f"{date.today():%Y-%m-%d}",
                  "prompt_versions": [PROMPT_VERSION, ENTAILMENT_VERSION], "summary": summary, "ledger": ledger_usage(),
                  "rows": rows}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    recorded = json.loads(report_path.read_text(encoding="utf-8"))
    consistent = comparable(rows) == comparable(recorded["rows"])
    recorded["replay"] = {"consistent": consistent, "n": len(rows),
                          "differences": [r["sample_id"] for r, o in zip(comparable(rows), comparable(recorded["rows"]), strict=True) if r != o]}
    report_path.write_text(json.dumps(recorded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("replay 一致" if consistent else f"replay 不一致：{recorded['replay']['differences']}", flush=True)


if __name__ == "__main__":
    main()
