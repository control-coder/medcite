"""生成侧评测（M13）：v4 dev 中 intent=popular_science 的全部查询，系统为 v4 混合检索 + 重排 + 检索 Agent v2 + 生成式回答。

指标按公开定义自行实现（不引入 ragas 包，模型调用全部经预算账本并写调用记录），定义见 docs/evaluation.md「v4 生成侧指标（M13）」：
引用召回率 / 引用精确率（ALCE）、忠实度 / 答案相关性（RAGAS）、上下文精确率、误答率与过度拒答率。

评审（NLI）由同一 MiMo 模型担任，提示词 ``eval-nli-v1`` 独立于生成与 ``entailment-v1``；每个回答一次批量调用
（超出输入上限时自动拆分），答案相关性另一次调用 ``eval-qgen-v1`` 反向生成 3 个问题，再用本地 bge-small-zh 算相似度。

检索 Agent 的请求与 M11 相同，只读复用 ``eval/cassettes/v4-agent-v2.jsonl``，新记录写入 ``v4-generation-quality.jsonl``。

用法::

    python -m eval.generation_quality --mode record --limit 3   # 冒烟，不写报告
    python -m eval.generation_quality --mode record             # 全量
    python -m eval.generation_quality --mode replay             # 回放并核对
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
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

from eval.agent_v2_eval import build_rag, classify, dev_queries, serialize_retrieval  # noqa: E402
from eval.hybrid_matrix import CHUNKS_PATH, JUDGMENTS_PATH, QUERIES_PATH  # noqa: E402
from eval.llm_pipeline_eval import call_with_retry, make_provider  # noqa: E402
from eval.router_eval import build_relevance  # noqa: E402
from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport  # noqa: E402
from medidiag.llm.contracts import LLMRequest  # noqa: E402
from medidiag.llm.models import ACTIVE_MIMO_MODEL  # noqa: E402
from medidiag.workflow.generative_answer import ENTAILMENT_VERSION, PROMPT_VERSION  # noqa: E402
from medidiag.workflow.retrieval_agent_v2 import RetrievalAgentV2WorkflowProvider  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "generation-quality-v1"
REPORT_DIR = ROOT / "artifacts/reports/generation"
M11_REPORT = ROOT / "artifacts/reports/retrieval/agent-v2-v4-20261028.json"
CONFIG = "configs/application_v4_agent.yaml"
NLI_VERSION = "eval-nli-v1"
QGEN_VERSION = "eval-qgen-v1"
EVIDENCE_REF = "@evidence"
LABELS = ("entailed", "partial", "not_entailed")
MAX_PROMPT_CHARS = 11800  # 传输层上限是 str(messages) <= 12000
N_QUESTIONS = 3
PIPELINE_KINDS = {"retrieval-agent-v2": "agent", "hyde-v1": "hyde", PROMPT_VERSION: "generation",
                  ENTAILMENT_VERSION: "entailment", "mimo-grounded-v1": "extractive"}
JUDGE_KINDS = {NLI_VERSION: "nli", QGEN_VERSION: "qgen"}
# 费用估算口径同 M3：每百万输入 Token 1 元、输出 2 元（未核实实际计费，仅供量级参考）
YUAN_PER_M_PROMPT = 1.0
YUAN_PER_M_COMPLETION = 2.0

NLI_PROMPT = f"""你是自然语言推理（NLI）评审。items 中每项有 id、hypothesis（假设）、premise（前提）。
premise 是一段文字；若为 "{EVIDENCE_REF}"，前提是输入里 evidence 中全部段落的合并。
只依据前提判断假设能否成立，不使用常识或其他资料，不考虑假设本身是否正确。
entailed：假设的全部信息都能由前提推出。
partial：只有一部分信息能由前提推出，其余信息前提没有提到。
not_entailed：前提推不出假设，或与假设矛盾（包括数值、否定、条件不一致）。
返回 JSON：{{"verdicts":[{{"id":整数,"label":"entailed 或 partial 或 not_entailed"}}]}}，每个 id 恰好一条，不输出其他内容。"""

QGEN_PROMPT = f"""你是问题生成器。给你一段回答（answer），请反推出用户最可能提出的 {N_QUESTIONS} 个不同的问题，
使这段回答恰好是它们的答案。问题要完整、自成一句，不照抄回答，不使用代词指代回答内容。
返回 JSON：{{"questions":["问题1","问题2","问题3"]}}，不输出其他内容。"""


# ----------------------------------------------------------------- 指标纯函数

def citation_relevance(all_verdict: str, alone: Sequence[str]) -> list[bool]:
    """ALCE 引用精确率的逐条判定。

    ``all_verdict`` 是全部引用拼接后对句子的判定，``alone`` 是每条引用单独的判定（至多 2 条）。
    句子未被全部引用蕴含时引用均不计为相关；单引用句的引用在句被蕴含时相关；
    多引用句中，一条引用「不相关」当且仅当它单独不蕴含句子，且去掉它后其余引用仍蕴含句子。
    """
    if all_verdict != "entailed":
        return [False] * len(alone)
    if len(alone) == 1:
        return [True]
    out = []
    for i, verdict in enumerate(alone):
        rest_entails = all(other == "entailed" for j, other in enumerate(alone) if j != i)
        out.append(not (verdict != "entailed" and rest_entails))
    return out


def citation_metrics(sentences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """引用召回率 / 精确率。每句含 ``all``（全部引用拼接判定）与 ``alone``（各引用单独判定）；缺失判定的句子不计入。"""
    complete = [s for s in sentences if s.get("all") is not None and None not in (s.get("alone") or [None])]
    recalled = sum(1 for s in complete if s["all"] == "entailed")
    relevant = total = 0
    for s in complete:
        flags = citation_relevance(s["all"], s["alone"])
        relevant += sum(flags)
        total += len(flags)
    return {"recall": rate(recalled, len(complete)), "precision": rate(relevant, total), "n_sentences": len(complete)}


def faithfulness(verdicts: Sequence[str | None]) -> dict[str, Any]:
    """RAGAS 忠实度：展示句中被整份证据蕴含的比例（每句即一条陈述）；另列部分蕴含占比。"""
    judged = [v for v in verdicts if v is not None]
    return {"faithfulness": rate(sum(v == "entailed" for v in judged), len(judged)),
            "partial_share": rate(sum(v == "partial" for v in judged), len(judged)), "n_sentences": len(judged)}


def context_precision(evidence_ids: Sequence[str], positives: set[str]) -> float:
    """RAGAS 上下文精确率：对相关位置 k 取 precision@k，按相关块数平均；没有相关块为 0。"""
    relevant = [cid in positives for cid in evidence_ids]
    if not any(relevant):
        return 0.0
    hits, total = 0, 0.0
    for k, flag in enumerate(relevant, 1):
        if flag:
            hits += 1
            total += hits / k
    return total / sum(relevant)


def mean_cosine(query_vec: Sequence[float], vectors: Sequence[Sequence[float]]) -> float:
    """向量已归一化时的平均余弦相似度（内积）。"""
    return sum(sum(a * b for a, b in zip(query_vec, v, strict=True)) for v in vectors) / len(vectors)


def rate(count: int, total: int) -> dict[str, Any]:
    return {"count": f"{count}/{total}", "rate": round(count / total, 4) if total else None}


def mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# ----------------------------------------------------------------- 评审调用

class _LLM:
    """计数的模型入口：可重试错误自动重试，按调用类型统计次数、错误码和 token（线程安全）。"""

    def __init__(self, provider: Any, kinds: Mapping[str, str]) -> None:
        self.provider, self.kinds = provider, kinds
        self.calls: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()
        self._lock = threading.Lock()

    def generate(self, request: Any, *, timeout_s: float, idempotency_key: str) -> Any:
        with self._lock:
            self.calls[self.kinds.get(request.prompt_version, request.prompt_version)] += 1
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


def nli_request(evidence: Mapping[str, str] | None, items: Sequence[Mapping[str, Any]], model: str) -> LLMRequest:
    payload: dict[str, Any] = {"items": [{k: i[k] for k in ("id", "hypothesis", "premise")} for i in items]}
    if evidence is not None:
        payload = {"evidence": dict(evidence), **payload}
    return LLMRequest(
        messages=[{"role": "system", "content": NLI_PROMPT},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        model=model, response_format={"type": "json_object"}, max_tokens=1024, reasoning_mode="disabled",
        prompt_version=NLI_VERSION)


def plan_batches(evidence: Mapping[str, str], items: Sequence[dict[str, Any]], model: str
                 ) -> list[tuple[LLMRequest, list[dict[str, Any]]]]:
    """一个回答的全部判定对优先放进一次请求；超出输入上限时先按是否引用整份证据拆开，再对半拆。"""

    def build(group: list[dict[str, Any]]) -> LLMRequest:
        uses_evidence = any(i["premise"] == EVIDENCE_REF for i in group)
        return nli_request(evidence if uses_evidence else None, group, model)

    def split(group: list[dict[str, Any]]) -> list[tuple[LLMRequest, list[dict[str, Any]]]]:
        request = build(group)
        if len(str(request.messages)) <= MAX_PROMPT_CHARS or len(group) == 1:
            return [(request, group)]
        with_evidence = [i for i in group if i["premise"] == EVIDENCE_REF]
        without = [i for i in group if i["premise"] != EVIDENCE_REF]
        if with_evidence and without:
            return split(with_evidence) + split(without)
        mid = len(group) // 2
        return split(group[:mid]) + split(group[mid:])

    return split(list(items))


def parse_nli(parsed: Any, ids: Sequence[int]) -> dict[int, str] | None:
    verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
    if not isinstance(verdicts, list):
        return None
    out: dict[int, str] = {}
    for verdict in verdicts:
        if (not isinstance(verdict, dict) or set(verdict) != {"id", "label"} or type(verdict["id"]) is not int
                or verdict["label"] not in LABELS or verdict["id"] in out):
            return None
        out[verdict["id"]] = verdict["label"]
    return out if set(out) == set(ids) else None


def build_items(sentences: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """每句的判定对：全部引用拼接（召回）、双引用句的每条引用单独（精确率）、整份证据（忠实度）。"""
    items: list[dict[str, Any]] = []

    def add(kind: str, sentence: int, cite: int | None, premise: str) -> None:
        items.append({"id": len(items), "kind": kind, "sentence": sentence, "cite": cite,
                      "hypothesis": sentences[sentence]["text"], "premise": premise})

    for i, sentence in enumerate(sentences):
        quotes = [c["quote"] for c in sentence["citations"]]
        add("all", i, None, "\n".join(quotes))
        if len(quotes) > 1:
            for j, quote in enumerate(quotes):
                add("alone", i, j, quote)
        add("faith", i, None, EVIDENCE_REF)
    return items


def judge_answer(llm: _LLM, model: str, evidence: Mapping[str, str], sentences: Sequence[Mapping[str, Any]]
                 ) -> dict[str, Any]:
    """对一个回答做 NLI 判定与反向生成问题；失败只记录错误码，对应判定为 None，不重试付费调用。"""
    items = build_items(sentences)
    verdicts: dict[int, str] = {}
    errors: list[str] = []
    for request, group in plan_batches(evidence, items, model):
        try:
            result = llm.generate(request, timeout_s=45, idempotency_key="m13-nli-" + uuid.uuid4().hex)
        except MediDiagError as exc:
            errors.append(exc.code)
            continue
        parsed = parse_nli(result.parsed_json, [i["id"] for i in group])
        if parsed is None:
            errors.append("NLI_SHAPE")
            continue
        verdicts.update(parsed)
    per_sentence: list[dict[str, Any]] = [{"all": None, "alone": [None] * len(s["citations"]), "faith": None}
                                          for s in sentences]
    for item in items:
        verdict = verdicts.get(item["id"])
        slot = per_sentence[item["sentence"]]
        if item["kind"] == "alone":
            slot["alone"][item["cite"]] = verdict
        else:
            slot[item["kind"]] = verdict
    for slot in per_sentence:
        if len(slot["alone"]) == 1:  # 单引用句：单独判定即全部引用判定
            slot["alone"] = [slot["all"]]
    return {"sentences": per_sentence, "questions": generate_questions(llm, model, sentences, errors), "errors": errors}


def generate_questions(llm: _LLM, model: str, sentences: Sequence[Mapping[str, Any]], errors: list[str]) -> list[str] | None:
    request = LLMRequest(
        messages=[{"role": "system", "content": QGEN_PROMPT},
                  {"role": "user", "content": json.dumps({"answer": "".join(s["text"] for s in sentences)}, ensure_ascii=False)}],
        model=model, response_format={"type": "json_object"}, max_tokens=512, reasoning_mode="disabled",
        prompt_version=QGEN_VERSION)
    try:
        result = llm.generate(request, timeout_s=45, idempotency_key="m13-qgen-" + uuid.uuid4().hex)
    except MediDiagError as exc:
        errors.append(exc.code)
        return None
    questions = result.parsed_json.get("questions") if isinstance(result.parsed_json, dict) else None
    if (not isinstance(questions, list) or len(questions) != N_QUESTIONS
            or not all(isinstance(q, str) and q.strip() for q in questions)):
        errors.append("QGEN_SHAPE")
        return None
    return [q.strip() for q in questions]


# ----------------------------------------------------------------- 单题流程

def make_base(rag: Any, model: str) -> Any:
    provider = RetrievalAgentV2WorkflowProvider(rag, llm=None, on_invalid="feedback", model=model,
                                                answer_mode="generative")
    provider._parent_map()  # 先加载父块，之后每题复制出的 provider 共用同一份
    return provider


def run_query(base: Any, query: dict[str, Any], positives: set[str], inner: Callable[..., Any], model: str) -> dict[str, Any]:
    provider_llm = make_provider(inner)
    pipeline, judge = _LLM(provider_llm, PIPELINE_KINDS), _LLM(provider_llm, JUDGE_KINDS)
    provider = replace(base, llm=pipeline)
    question = provider.normalize(query["question"])["normalized_query"]
    error: str | None = None
    retrieval: dict[str, Any] = {"chunks": []}
    payload: dict[str, Any] = {"claims": []}
    try:
        retrieval = provider.retrieve(question)
        payload = provider.generate(question, retrieval, provider.plan(question, retrieval)).payload
    except MediDiagError as exc:
        error = exc.code
    claims = list(payload.get("claims", []))
    trace = payload.get("generative_trace") or {}
    evidence = {c["chunk_id"]: c["text"] for c in retrieval["chunks"]}
    evidence_ids = list(evidence)
    cited = [cid for c in claims for cid in c.get("citation_chunk_ids", [])]
    degraded = payload.get("degraded_from") == "generative"
    generative = bool(claims) and payload.get("answer_mode") == "generative" and not degraded
    answered = bool(claims)
    outcome = ("error" if error else "degraded_extractive" if degraded else "generative" if generative
               else "refused")
    sentences = [{"text": c["text"], "entailment": c.get("entailment"), "suspicious": c.get("suspicious"),
                  "citations": c["citations"]} for c in claims] if generative else []
    judged: dict[str, Any] = {"sentences": [], "questions": None, "errors": []}
    if sentences:
        judged = judge_answer(judge, model, evidence, sentences)
    return {
        "sample_id": query["sample_id"], "answerable": query.get("answerable"), "question": question,
        "outcome": outcome, "classification": classify(query, answered, error is not None, any(c in positives for c in cited)),
        "generation_error": error, "evidence_ids": evidence_ids,
        "context_precision": context_precision(evidence_ids, positives) if query.get("answerable") is True else None,
        "sentences": sentences, "extractive_claims": [c["text"] for c in claims] if degraded else [],
        "dropped": trace.get("dropped", []), "invalid_reasons": trace.get("invalid_reasons", []),
        "entailment_error": trace.get("entailment_error"), "degraded": trace.get("degraded"),
        "judge": judged["sentences"], "generated_questions": judged["questions"], "judge_errors": judged["errors"],
        "pipeline_calls": dict(pipeline.calls), "judge_calls": dict(judge.calls),
        "call_errors": dict(pipeline.errors + judge.errors),
        "tokens": dict(pipeline.tokens), "judge_tokens": dict(judge.tokens)}


def run_all(base: Any, queries: Sequence[dict[str, Any]], relevance: dict[str, set[str]], inner: Callable[..., Any],
            model: str, workers: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = last_print = time.time()

    def one(query: dict[str, Any]) -> dict[str, Any]:
        return run_query(base, query, relevance[query["sample_id"]], inner, model)

    with ThreadPoolExecutor(workers) as pool:
        for done, row in enumerate(pool.map(one, queries), 1):
            rows.append(row)
            if done % 5 == 0 or done == len(queries) or time.time() - last_print > 30:
                last_print = time.time()
                print(f"[gen-quality] {done}/{len(queries)}（{last_print - started:.0f}s）", flush=True)
    return rows


def add_relevancy(rows: list[dict[str, Any]], encoder: Callable[[list[str]], Any]) -> None:
    """答案相关性：生成问题与原问题的平均余弦相似度（本地 bge-small-zh，两侧都不加检索前缀）。"""
    for row in rows:
        row["answer_relevancy"] = None
        if row["generated_questions"]:
            vectors = encoder([row["question"], *row["generated_questions"]])
            row["answer_relevancy"] = round(float(mean_cosine(vectors[0], vectors[1:])), 4)


# ----------------------------------------------------------------- 汇总

def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["answerable"] is True]
    unanswerable = [r for r in rows if r["answerable"] is not True]
    ok = [r for r in rows if r["outcome"] != "error"]
    outcomes = Counter(r["outcome"] for r in rows)
    answered_u = [r for r in unanswerable if r["outcome"] in ("generative", "degraded_extractive")]
    refused_a = [r for r in answerable if r["outcome"] == "refused"]
    shown = [r for r in rows if r["outcome"] == "generative"]
    flat = [{**j, "faith": j["faith"]} for r in shown for j in r["judge"]]
    cite = citation_metrics(flat)
    faith = faithfulness([j["faith"] for j in flat])
    relevancy = [r["answer_relevancy"] for r in shown if r.get("answer_relevancy") is not None]
    ctx = [r["context_precision"] for r in answerable if r["context_precision"] is not None]
    return {
        "n": len(rows), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable),
        "outcomes": dict(outcomes), "generation_errors": outcomes["error"],
        "generative_answers": len(shown), "sentences_shown": sum(len(r["sentences"]) for r in shown),
        "citation_recall": cite["recall"], "citation_precision": cite["precision"],
        "citation_sentences_judged": cite["n_sentences"],
        "faithfulness": faith["faithfulness"], "faithfulness_partial_share": faith["partial_share"],
        "faithfulness_sentences_judged": faith["n_sentences"],
        "answer_relevancy": {"mean": mean(relevancy), "n": len(relevancy)},
        "context_precision": {"mean": mean(ctx), "n": len(ctx), "n_zero": sum(1 for v in ctx if v == 0)},
        "wrong_answer_unanswerable_answered": rate(len(answered_u), len(unanswerable)),
        "over_refusal": rate(len(refused_a), len(answerable)),
        "wrong_answer_irrelevant_citation": rate(
            sum(1 for r in answerable if r["classification"] == "answered_irrelevant"), len(answerable)),
        "degraded_extractive": outcomes["degraded_extractive"],
        "sentences_dropped": sum(len(r["dropped"]) for r in ok),
        "sentences_partial": sum(1 for r in shown for s in r["sentences"] if s["entailment"] == "partial"),
        "validation_failures": sum(len(r["invalid_reasons"]) for r in ok),
        "generation_retries": sum(max(0, r["pipeline_calls"].get("generation", 0) - 1) for r in ok),
        "entailment_errors": sum(1 for r in ok if r["entailment_error"]),
        "judge_error_answers": sum(1 for r in rows if r["judge_errors"])}


def extractive_baseline() -> dict[str, Any] | None:
    """同检索下抽取式回答的拒答指标，直接引用 M11 报告的 agent_v2；抽取式内容即原文，忠实度不与生成式直接比较。"""
    if not M11_REPORT.exists():
        return None
    summary = json.loads(M11_REPORT.read_text(encoding="utf-8"))["summary"]["agent_v2"]
    return {"source": str(M11_REPORT.relative_to(ROOT)).replace("\\", "/"),
            "wrong_answer_unanswerable_answered": summary["wrong_answer_unanswerable_answered"],
            "over_refusal": summary["over_refusal"],
            "note": "抽取式回答的内容就是证据原文，忠实度不与生成式直接比较。"}


LIMITATIONS = [
    "评审与被评系统同源（均为 MiMo），有偏差；判定不等于医学正确，也不等于人工审核。",
    "忠实度、引用召回率与精确率依赖模型 NLI 判定；评审检查（entailment_check）单独报告判定在程序构造样本上的检出率。",
    "上下文精确率的相关性来自 v4 句子级 LLM 标注（relevant 与 partial 都算相关），未经人工复核。",
    "样本量小（dev 46 条，其中展示生成式回答的更少），只作工程对照，不作显著性结论；dev 内未调参。",
    "答案相关性只对展示了生成式回答的查询计算；降级为抽取式或拒答的查询不计入。",
]


# ----------------------------------------------------------------- 主流程

def run(args: argparse.Namespace) -> dict[str, Any]:
    from medidiag.schemas import read_jsonl

    for path in (CHUNKS_PATH, QUERIES_PATH, JUDGMENTS_PATH):
        if not path.exists():
            raise SystemExit(f"[generation_quality] 缺少 {path}")
    queries = dev_queries()
    if args.limit:
        queries = queries[: args.limit]
    n_a = sum(1 for q in queries if q.get("answerable") is True)
    print(f"[load] dev 科普查询 {len(queries)} 条（可回答 {n_a}，不可回答 {len(queries) - n_a}）", flush=True)
    relevance = build_relevance(queries, read_jsonl(CHUNKS_PATH))
    import torch

    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rag, config = build_rag(device)
    serialize_retrieval(rag)

    cassette = Cassette(args.cassette, base=Cassette(args.base_cassette))
    if args.mode == "record":
        inner: Any = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        inner = ReplayTransport(cassette)

    rows = run_all(make_base(rag, args.model), queries, relevance, inner, args.model, args.workers)

    from medidiag.rag.dense import load_local_encoder

    embedding = config["embedding"]
    encoder = load_local_encoder(embedding["model"], embedding["revision"], device=device,
                                 batch_size=64 if device == "cuda" else 8)
    add_relevancy(rows, encoder)

    tokens: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    for row in rows:
        tokens.update(row["tokens"])
        tokens.update(row["judge_tokens"])
        errors.update(row["call_errors"])
        calls.update(row["pipeline_calls"])
        calls.update(row["judge_calls"])
    cost: dict[str, Any] = {
        "model_calls": dict(calls), "model_calls_total": sum(calls.values()), "call_errors": dict(errors),
        "input_tokens": tokens["input_tokens"], "output_tokens": tokens["output_tokens"],
        "est_yuan": round((tokens["input_tokens"] * YUAN_PER_M_PROMPT
                           + tokens["output_tokens"] * YUAN_PER_M_COMPLETION) / 1e6, 4),
        "note": "model_calls 含命中旧调用记录（检索 Agent 与回放）的请求；真实新增调用见 spend（账本累计，含评审检查）。"}
    if args.mode == "record":
        cost["spend"] = inner.spent()
    return {
        "schema_version": SCHEMA_VERSION, "date": str(date.today()), "mode": args.mode, "model": args.model,
        "split": "dev", "intent": "popular_science", "queries": len(queries), "cost": cost,
        "params": {"config": CONFIG, "answer_mode": "generative", "on_invalid": "feedback",
                   "retrieval": "hybrid + rerank + MMR，parent_child 子块，top3，检索 Agent v2（同 M11）",
                   "judge": {"nli_prompt": NLI_VERSION, "question_prompt": QGEN_VERSION,
                             "nli_prompt_sha256": hashlib.sha256(NLI_PROMPT.encode()).hexdigest()[:12],
                             "reasoning": "disabled", "n_generated_questions": N_QUESTIONS,
                             "relevancy_encoder": embedding["model"]}},
        "summary": summarize(rows), "extractive_baseline": extractive_baseline(), "limitations": LIMITATIONS,
        "rows": rows}


def replay_check(report: dict[str, Any], path: Path) -> dict[str, Any]:
    """回放结果与 record 报告逐题对照（含判定、生成的问题与相关性）。"""
    if not path.exists():
        return {"checked": False, "reason": "没有找到 record 报告"}
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(report["rows"], ensure_ascii=False))
    differing = [a["sample_id"] for a, b in zip(recorded["rows"], current, strict=False) if a != b]
    consistent = not differing and len(recorded["rows"]) == len(current)
    errors = sum(report["cost"]["call_errors"].values())
    return {"checked": True, "consistent": consistent, "differing": differing[:10], "call_errors": errors,
            "summary_equal": recorded["summary"] == report["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/v4-generation-quality.jsonl")
    parser.add_argument("--base-cassette", type=Path, default=ROOT / "eval/cassettes/v4-agent-v2.jsonl")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/v4-m13-ledger.db")
    parser.add_argument("--max-calls", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只评前 N 条查询，不写报告")
    args = parser.parse_args()

    report = run(args)
    path = REPORT_DIR / f"generation-quality-v4-{date.today():%Y%m%d}.json"
    brief = {k: report[k] for k in ("queries", "cost", "summary", "extractive_baseline")}
    if args.limit:
        print(json.dumps(brief, ensure_ascii=False, indent=2))
        return
    if args.mode == "replay":
        check = replay_check(report, path)
        print(json.dumps(check, ensure_ascii=False, indent=2))
        if check.get("checked") and check["consistent"]:
            recorded = json.loads(path.read_text(encoding="utf-8"))
            recorded["replay_check"] = check
            path.write_text(json.dumps(recorded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"[done] 报告 {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
