"""v4 评测集句子级相关性标注与冻结（M7 阶段 B）。

池化标注（pooling，TREC 式）：对每条查询，取 4 种现有检索方法（recursive×BM25 cjk_bigram、
recursive×向量、parent_child×BM25、parent_child×向量）各自的前 5 名子块，按
``chunk_id`` 去重；v3 查询另把 ``v3_gold_mapping.json`` 中两种策略的 gold 子块
并入池（用于与 v3 标注的一致率计算）。

标注：标注池中每个段落按 ``chunking._split_sentences`` 切成句子，同一句子（去空白
sha256 相同）在标注池内只提交一次、只标一次；按请求内顺序编号 ``p{段序}s{句序}``
（如 p3s2），超出输入预算则拆成多次请求。由模型返回 relevant / partial 的句子
编号，其余默认无关。输出校验失败时最多重试 1 次：在原消息后追加确定性的纠错说明
（列出非法编号），仍无效的请求记 judge_failed，其句子按 irrelevant 处理。提示词
版本号 ``pooled-judge-v4-sentence``。标注与分块策略无关：评估时用
``chunk_relevance`` 把句子标签投影到任意策略的子块上（与其有重叠的已标注句子
中的最高标签，去空白比较）。

产物（均写入 ``examples/public_health_v4/``）：

- ``judgments.jsonl``：句子级标注（query_id、doc_id、sentence_hash、label），
  sentence_hash = 去空白后句子文本的 sha256；不提交句子原文（WHO 页面未获全文
  再分发许可）。
- ``pools.jsonl``：每条查询的池（doc_id、chunk_id、method、rank、via），供审计；
  含被模型引用的句子的 sha256，用于校验模型未改写原文，仍不含原文。
- ``v3_no_evidence_recheck.json``：v3 的 56 条无证据查询在 v4 语料上的重新判定
  （v3 文件不改，变化单列）。
- ``queries.jsonl``：就地写回 answerable、gold_evidence_ids（parent_child 策略下
  相关子块，兼容旧评测脚本；权威标注是 judgments.jsonl）后冻结。

模式：``record`` 需要真实凭据并占用持久预算；``replay`` 只读调用记录，绝不联网。

用法::

    python -m eval.v4_judgments --mode record --limit 4   # 小规模试跑
    python -m eval.v4_judgments --mode record             # 全量标注 + 冻结
    python -m eval.v4_judgments --mode replay --skip-freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace

# 向量模型只读本地缓存，不联网探测（须在 transformers/sentence_transformers 导入前设置）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.llm_pipeline_eval import CONTRACT_VIOLATIONS, call_with_retry, make_provider  # noqa: E402
from eval.retrieval_benchmark import (  # noqa: E402
    BGE_ZH_QUERY_PREFIX,
    DEFAULT_EMBEDDING_MODEL,
    Bm25Scorer,
    DenseScorer,
    EmbeddingCache,
    rank_chunks,
)
from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport  # noqa: E402
from medidiag.llm.contracts import LLMRequest  # noqa: E402
from medidiag.llm.models import ACTIVE_MIMO_MODEL, ALLOWED_MIMO_MODELS  # noqa: E402
from medidiag.rag.ingest.chunking import _split_sentences, _squash  # noqa: E402
from medidiag.schemas import read_jsonl, write_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "v4-judgments-v1"
PROMPT_VERSION = "pooled-judge-v4-sentence"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
POOL_DEPTH = 5  # 每种检索方法的前 5 名子块
# BudgetedTransport 要求 len(str(messages)) <= 12000；每次请求的句子预算按
# 「字符数 + 每句 30 字符结构开销」计 7500，超出则拆成追加请求。
SENTENCE_CHAR_BUDGET = 7500
LABELS = ("relevant", "partial")
LABEL_RANK = {"irrelevant": 0, "partial": 1, "relevant": 2}

JUDGE_PROMPT = (
    "你是检索评测的相关性标注员。给你一个用户问题和若干段公共卫生科普候选原文，"
    "每段已切成带编号的句子（p段序s句序，如 p1s2）。逐句判断该句是否回答了问题所问的内容："
    "句子本身包含问题要的答案（事实、数字、做法或原因），标 relevant；"
    "只覆盖问题的一部分、或不完整但确有直接帮助，标 partial；"
    "只是话题相同、只出现了相关词语、或需要额外知识才能推出答案的，不标。"
    "只依据给出的原文判断，不使用你自己的医学知识。"
    '返回 JSON：{"relevant": ["p1s2", ...], "partial": ["p2s1", ...]}；'
    "编号必须逐字照抄给出的句子编号，不要自己编造；"
    "两个列表不重复；没有就返回空列表。"
)



# ----------------------------------------------------------------- 数据集加载

class V4Corpus:
    """v4 查询、两种策略的子块与 v3 gold 映射。"""

    def __init__(self, data_dir: Path, chunks_dir: Path, mapping_path: Path | None = None,
                 v3_queries_path: Path | None = None) -> None:
        self.queries = read_jsonl(data_dir / "queries.jsonl")
        self.chunks = {
            strategy: read_jsonl(chunks_dir / strategy / "chunks.jsonl")
            for strategy in ("recursive", "parent_child")
        }
        for strategy, chunks in self.chunks.items():
            ids = [c["chunk_id"] for c in chunks]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{strategy} 的 chunk_id 不唯一")
        self.chunk_by_id = {
            strategy: {c["chunk_id"]: c for c in chunks} for strategy, chunks in self.chunks.items()
        }
        mapping_path = mapping_path or data_dir / "v3_gold_mapping.json"
        self.mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        # v3 gold 摘录只从 v3 原查询读取：v4 的 queries.jsonl 冻结时会被改写 gold_evidence_ids
        v3_path = v3_queries_path or data_dir.parent / "public_health_v3" / "queries.jsonl"
        self.v3_excerpts = {q["sample_id"]: list(q.get("gold_evidence_ids") or [])
                            for q in read_jsonl(v3_path)} if v3_path.exists() else {}

    def v3_gold(self, sample_id: str) -> dict[str, list[str]]:
        """v3 查询在两种策略下的 gold 子块（仅映射成功的）；非 v3 查询为空。"""
        out: dict[str, list[str]] = {}
        for strategy in ("recursive", "parent_child"):
            entry = self.mapping["strategies"][strategy]["queries"].get(sample_id)
            out[strategy] = list(entry["gold"]) if entry else []
        return out

    def query_excerpt_ids(self, sample_id: str) -> list[str]:
        """v3 查询的 gold 摘录 ID（来自 v3 原查询文件）；非 v3 查询为空。"""
        return list(self.v3_excerpts.get(sample_id, []))


# ----------------------------------------------------------------- 检索与池化

def build_scorers(corpus: V4Corpus, cache: EmbeddingCache | None) -> dict[str, Any]:
    """4 种检索方法的打分器：两种策略 × BM25 cjk_bigram / 向量。"""
    scorers: dict[str, Any] = {}
    for strategy in ("recursive", "parent_child"):
        chunks = corpus.chunks[strategy]
        bm25 = Bm25Scorer(f"{strategy}_bm25", "cjk_bigram")
        bm25.fit(chunks)
        scorers[f"{strategy}_bm25"] = (bm25, chunks)
        if cache is not None:
            dense = DenseScorer(f"{strategy}_dense", cache, BGE_ZH_QUERY_PREFIX)
            dense.fit(chunks)
            scorers[f"{strategy}_dense"] = (dense, chunks)
    return scorers


def sentence_id(segment_no: int, sentence_no: int) -> str:
    """请求内按顺序编号：p{段序}s{句序}，池确定则编号确定。"""
    return f"p{segment_no}s{sentence_no}"


class PassageState:
    """标注池中的一段候选（一个子块）：句子切分、本段独占的句子与请求归属。"""

    def __init__(self, doc_id: str, chunk_id: str, title: str, text: str) -> None:
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.title = title
        self.text = text
        self.sentences = _split_sentences(text)
        self.hashes = [sentence_hash(s) for s in self.sentences]
        self.owned: list[int] = []  # 标注池内首次出现的句子下标（重复句子只提交一次）
        self.request_no = 0  # 第几次请求（1 起）；0 表示本段没有需要提交标注的新句子
        self.ids: dict[int, str] = {}  # owned 句子下标 -> 请求内编号


class QueryPool:
    """一条查询的标注池：按 chunk_id 去重的段落序列与检索方法来源。"""

    def __init__(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.passages: list[PassageState] = []
        self.sources: dict[str, list[dict[str, Any]]] = {}  # chunk_id -> [{method, rank, via}]
        self.sentence_texts: dict[str, str] = {}  # sentence_hash -> 去空白句子文本（会话内）
        self.request_ids: dict[int, dict[str, str]] = {}  # 请求号 -> {编号: sentence_hash}
        self.hash_sid: dict[str, str] = {}  # sentence_hash -> 编号

    def add(self, passage: PassageState, method: str, rank: int, via: str) -> None:
        self.sources.setdefault(passage.chunk_id, []).append({"method": method, "rank": rank, "via": via})
        if all(p.chunk_id != passage.chunk_id for p in self.passages):
            self.passages.append(passage)
        for sentence, digest in zip(passage.sentences, passage.hashes, strict=True):
            self.sentence_texts.setdefault(digest, _squash(sentence))


def build_pool(corpus: V4Corpus, scorers: dict[str, Any], query: dict[str, Any],
               depth: int = POOL_DEPTH) -> QueryPool:
    """对一条查询跑 4 种检索方法取前 depth 名，按 chunk_id 去重并入 v3 gold 子块。"""
    pool = QueryPool(query["sample_id"])

    def passage_of(strategy: str, chunk_id: str) -> PassageState:
        chunk = corpus.chunk_by_id[strategy][chunk_id]
        meta = chunk.get("metadata", {})
        return PassageState(doc_id=str(meta.get("doc_id") or chunk["source_id"]),
                            chunk_id=chunk_id, title=chunk.get("source", chunk_id),
                            text=chunk["text"])

    for method, (scorer, chunks) in scorers.items():
        strategy = method.rsplit("_", 1)[0]
        chunk_ids = [c["chunk_id"] for c in chunks]
        ranked = rank_chunks(scorer.score(query["question"]), chunk_ids,
                             scorer.positive_only, top_n=depth)
        for rank, (chunk_id, _score) in enumerate(ranked, 1):
            pool.add(passage_of(strategy, chunk_id), method, rank, "pool")

    for strategy, via in (("parent_child", "gold_pc"), ("recursive", "gold_rc")):
        for chunk_id in corpus.v3_gold(query["sample_id"])[strategy]:
            if chunk_id in corpus.chunk_by_id[strategy]:
                pool.add(passage_of(strategy, chunk_id), f"gold_{strategy}", 0, via)
    return pool


# ----------------------------------------------------------------- 提示词组装

def assign_requests(pool: QueryPool, budget: int = SENTENCE_CHAR_BUDGET) -> int:
    """句子按哈希去重后给段落分配请求号并编号；同一段不拆到两次请求。

    返回请求数。预算按「句子字符数 + 每句 30 字符结构开销」估算。
    """
    seen: set[str] = set()
    request_no = 0
    used = budget  # 强制从第 1 次请求开始
    pool.request_ids = {}
    pool.hash_sid = {}
    segments: Counter[int] = Counter()
    for passage in pool.passages:
        passage.owned = []
        for index, digest in enumerate(passage.hashes):
            if digest not in seen:
                seen.add(digest)
                passage.owned.append(index)
        passage.request_no = 0
        passage.ids = {}
        if not passage.owned:
            continue
        size = sum(len(passage.sentences[i]) + 30 for i in passage.owned)
        if used + size > budget:
            request_no += 1
            used = 0
        used += size
        passage.request_no = request_no
        segments[request_no] += 1
        ids = pool.request_ids.setdefault(request_no, {})
        for number, index in enumerate(passage.owned, 1):
            sid = sentence_id(segments[request_no], number)
            passage.ids[index] = sid
            ids[sid] = passage.hashes[index]
            pool.hash_sid[passage.hashes[index]] = sid
    return request_no


def build_judge_request(pool: QueryPool, question: str, request_no: int,
                        model: str = ACTIVE_MIMO_MODEL) -> LLMRequest:
    """把该次请求覆盖的段落切成带编号句子，组装一次请求。"""
    passages = []
    for segment_no, passage in enumerate((p for p in pool.passages if p.request_no == request_no), 1):
        sentences = [{"id": passage.ids[i], "text": passage.sentences[i]} for i in passage.owned]
        passages.append({"passage": segment_no, "title": passage.title, "sentences": sentences})
    # BudgetedTransport 的输出上限 2048；reasoning_mode="enabled" 的思维链也占输出 Token
    return LLMRequest(
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps(
                {"question": question, "passages": passages}, ensure_ascii=False)},
        ],
        model=model, response_format={"type": "json_object"}, max_tokens=2048,
        reasoning_mode="enabled", prompt_version=PROMPT_VERSION)


def valid_sentence_ids(pool: QueryPool, request_no: int) -> set[str]:
    """该次请求内全部合法句子编号（供输出校验）。"""
    return set(pool.request_ids.get(request_no, {}))


def label_issues(parsed: Any, valid_ids: set[str]) -> list[str]:
    """列出模型输出的违规项（确定性顺序，供纠错说明使用）；空列表表示合法。"""
    if not isinstance(parsed, dict):
        return ["输出不是 JSON 对象"]
    issues: list[str] = []
    unknown_keys = sorted(set(parsed) - set(LABELS))
    if unknown_keys:
        issues.append(f"出现未定义的键：{unknown_keys}")
    seen: set[str] = set()
    illegal: list[str] = []
    duplicated: list[str] = []
    for label in LABELS:
        values = parsed.get(label, [])
        if not isinstance(values, list):
            issues.append(f"{label} 不是列表")
            continue
        for value in values:
            if not isinstance(value, str) or value not in valid_ids:
                illegal.append(str(value)[:24])
            elif value in seen:
                duplicated.append(value)
            else:
                seen.add(value)
    if illegal:
        issues.append(f"非法编号：{illegal[:10]}")
    if duplicated:
        issues.append(f"重复编号：{duplicated[:10]}")
    return issues


def build_correction_request(request: LLMRequest, issues: Sequence[str]) -> LLMRequest:
    """在原消息后追加确定性的纠错说明；请求体不同，cassette 可区分、回放确定。"""
    note = ("你上一次的输出不合法：" + "；".join(issues) +
            "。请重新输出，只使用题面 passages 中给出的句子编号，逐字照抄，"
            '两个列表不重复，格式仍为 {"relevant": [...], "partial": [...]}。')
    return replace(request, messages=[*request.messages, {"role": "user", "content": note}])


# ----------------------------------------------------------------- 标注解析与存储

def sentence_hash(text: str) -> str:
    """去空白后句子文本的 sha256。"""
    return hashlib.sha256(_squash(text).encode("utf-8")).hexdigest()


def parse_labels(parsed: Any, valid_ids: set[str]) -> dict[str, str] | None:
    """校验并展开模型输出；任何违规（非法编号、重复、未知键、非法标签）返回 None。"""
    if not isinstance(parsed, dict) or not set(parsed) <= set(LABELS):
        return None
    labels: dict[str, str] = {}
    for label in LABELS:
        values = parsed.get(label, [])
        if not isinstance(values, list):
            return None
        for value in values:
            if not isinstance(value, str) or value not in valid_ids or value in labels:
                return None
            labels[value] = label
    return labels


def query_requests(corpus: V4Corpus, scorers: dict[str, Any],
                   query: dict[str, Any], model: str) -> tuple[QueryPool, list[LLMRequest]]:
    """构建一条查询的池与全部请求（供 record/replay 共用，保证请求体一致）。"""
    pool = build_pool(corpus, scorers, query)
    n_requests = assign_requests(pool)
    return pool, [build_judge_request(pool, query["question"], no, model)
                  for no in range(1, n_requests + 1)]


def judge_requests(provider: Any, pool: QueryPool, sample_id: str,
                   requests: Sequence[LLMRequest], workers: int) -> dict[str, Any]:
    """对已建好的池发请求并解析；输出无效时最多重试 1 次（追加确定性纠错说明）。"""

    def generate(request: LLMRequest, key: str) -> Any:
        return call_with_retry(lambda: provider.generate(request, timeout_s=45, idempotency_key=key))

    def call(request_no: int, request: LLMRequest) -> dict[str, Any]:
        valid = valid_sentence_ids(pool, request_no)
        key = f"v4j-{sample_id}-{request_no}"
        usage: Counter[str] = Counter()
        issues: list[str] = []
        for attempt in (0, 1):
            current = request if attempt == 0 else build_correction_request(request, issues)
            try:
                result = generate(current, key if attempt == 0 else key + "-retry")
            except MediDiagError as exc:
                if exc.code not in CONTRACT_VIOLATIONS:
                    return {"request_no": request_no, "status": "error", "code": exc.code,
                            "labels": {}, "retried": attempt, "usage": dict(usage)}
                issues = ["输出不是合法的 JSON 对象"]
                continue
            for name, value in (result.usage or {}).items():
                if isinstance(value, int):
                    usage[name] += value
            issues = label_issues(result.parsed_json, valid)
            if not issues:
                parsed = result.parsed_json
                return {"request_no": request_no, "status": "ok", "retried": attempt,
                        "labels": {sid: label for label in LABELS for sid in parsed.get(label, [])},
                        "usage": dict(usage)}
        return {"request_no": request_no, "status": "judge_failed", "retried": 1,
                "detail": "；".join(issues)[:200], "labels": {}, "usage": dict(usage)}

    outcomes = list(ThreadPoolExecutor(min(workers, max(1, len(requests)))).map(
        lambda item: call(*item), enumerate(requests, 1)))
    labels: dict[str, str] = {}  # sentence_hash -> label
    for outcome in outcomes:
        ids = pool.request_ids.get(outcome["request_no"], {})
        labels.update({ids[sid]: label for sid, label in outcome["labels"].items()})
    return {"pool": pool, "labels": labels, "responses": outcomes}


def judge_query(provider: Any, corpus: V4Corpus, scorers: dict[str, Any],
                query: dict[str, Any], model: str, workers: int) -> dict[str, Any]:
    """标注一条查询：池化 → （可能多次）请求 → 解析 → 句子标签。"""
    pool, requests = query_requests(corpus, scorers, query, model)
    return judge_requests(provider, pool, query["sample_id"], requests, workers)


# ----------------------------------------------------------------- 标注落盘

def judgment_rows(corpus: V4Corpus, result: dict[str, Any]) -> list[dict[str, Any]]:
    """一条查询的句子级标注行（每个 doc_id × 句子哈希一行）：正标签按模型输出，其余默认 irrelevant。"""
    pool: QueryPool = result["pool"]
    labels: dict[str, str] = result["labels"]
    rows = []
    seen: set[tuple[str, str]] = set()
    for passage in pool.passages:
        for digest in passage.hashes:
            if (passage.doc_id, digest) in seen:
                continue
            seen.add((passage.doc_id, digest))
            rows.append({"record": "judgment", "query_id": pool.sample_id,
                         "doc_id": passage.doc_id, "chunk_id": passage.chunk_id,
                         "sentence_id": pool.hash_sid[digest], "sentence_hash": digest,
                         "label": labels.get(digest, "irrelevant")})
    return rows


def pool_rows(corpus: V4Corpus, result: dict[str, Any]) -> dict[str, Any]:
    """一条查询的池审计记录（不含原文；含被引用句子的 sha256 供校验）。"""
    pool: QueryPool = result["pool"]
    labels: dict[str, str] = result["labels"]
    cited: dict[str, dict[str, str]] = {}
    for passage in pool.passages:
        for digest in passage.hashes:
            if digest in labels:
                cited[digest] = {"doc_id": passage.doc_id, "label": labels[digest]}
    responses = result["responses"]
    return {"record": "pool", "query_id": pool.sample_id,
            "candidates": [{"doc_id": p.doc_id, "chunk_id": p.chunk_id, "title": p.title,
                            "n_sentences": len(p.sentences), "n_sent": len(p.owned),
                            "sources": pool.sources[p.chunk_id]}
                           for p in pool.passages],
            "n_unique_sentences": len(pool.hash_sid),
            "n_requests": len(responses),
            "statuses": [r["status"] for r in responses],
            "retried": sum(r.get("retried", 0) for r in responses),
            "judge_failed": any(r["status"] == "judge_failed" for r in responses),
            "cited": cited}


# ----------------------------------------------------------------- 句子 → 子块投影

def load_judgments(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """读 judgments.jsonl：query_id -> doc_id -> sentence_hash -> label。"""
    table: dict[str, dict[str, dict[str, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record", "judgment") != "judgment":
            continue
        table.setdefault(row["query_id"], {}).setdefault(row["doc_id"], {})[
            row["sentence_hash"]] = row["label"]
    return table


def chunk_relevance(chunk_text: str, doc_labels: dict[str, str]) -> str:
    """子块的相关性 = 与其有重叠的已标注句子中的最高标签（去空白比较）。

    ``doc_labels`` 的键是去空白句子或其 sha256（judgments.jsonl 不存原文，按
    sha256 匹配；测试与审计可直接给去空白句子）。重叠按去空白后的包含关系
    判定：句子 ⊆ 块，或块 ⊆ 句子（子块可能只覆盖长句的一部分）。没有重叠时
    返回 irrelevant。
    """
    squashed_chunk = _squash(chunk_text)
    chunk_digest = hashlib.sha256(squashed_chunk.encode("utf-8")).hexdigest()
    best = "irrelevant"
    for sentence, label in doc_labels.items():
        if sentence == chunk_digest:
            overlap = True  # 整句哈希命中：该块本身就是这条句子
        elif len(sentence) == 64 and all(c in "0123456789abcdef" for c in sentence):
            overlap = False  # 其他句子的哈希无法做子串比较
        else:
            overlap = bool(sentence) and (sentence in squashed_chunk or squashed_chunk in sentence)
        if overlap and LABEL_RANK[label] > LABEL_RANK[best]:
            best = label
    return best


def chunk_labels_for_query(query_id: str, chunks: Sequence[dict[str, Any]],
                           judgments: dict[str, dict[str, dict[str, str]]],
                           sentence_texts: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """把一条查询的句子标注投影到给定策略的全部子块：chunk_id -> label。

    ``sentence_texts`` 为 query_id -> sentence_hash -> 去空白句子文本；不提供时
    只匹配整句哈希命中的子块（父子子块多为 1–2 句的整句），提供时按包含关系
    做完整重叠匹配。标注方在投影时总会提供它（来自 pools.jsonl 之外的内部
    状态），下游评测若只有 judgments.jsonl，可用 ``recover_sentence_texts``
    从本机分块产物重建。
    """
    table = judgments.get(query_id, {})
    texts = (sentence_texts or {}).get(query_id, {})
    out: dict[str, str] = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        doc_id = str(meta.get("doc_id") or chunk["source_id"])
        doc_labels = dict(table.get(doc_id, {}))
        if texts:  # 用原文替换哈希键，做包含关系匹配
            doc_labels = {texts.get(key, key): label for key, label in doc_labels.items()}
        out[chunk["chunk_id"]] = chunk_relevance(chunk["text"], doc_labels)
    return out


def recover_sentence_texts(chunks_by_strategy: dict[str, Sequence[dict[str, Any]]],
                           judgments: dict[str, dict[str, dict[str, str]]]) -> dict[str, dict[str, str]]:
    """从本机分块产物重建 query_id -> sentence_hash -> 去空白句子文本。

    对 judgments 里出现过的 (query_id, doc_id)，把该文档在任一策略下的子块切成
    句子并计算哈希，命中的保留原文。与分块策略无关：同一句子在不同策略下哈希相同。
    """
    wanted: dict[str, set[str]] = {}  # doc_id -> sentence_hash
    for table in judgments.values():
        for doc_id, labels in table.items():
            wanted.setdefault(doc_id, set()).update(labels)
    doc_sentences: dict[str, dict[str, str]] = {}
    for chunks in chunks_by_strategy.values():
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            doc_id = str(meta.get("doc_id") or chunk["source_id"])
            if doc_id not in wanted:
                continue
            for sentence in _split_sentences(chunk["text"]):
                squashed = _squash(sentence)
                digest = hashlib.sha256(squashed.encode("utf-8")).hexdigest()
                if digest in wanted[doc_id]:
                    doc_sentences.setdefault(doc_id, {})[digest] = squashed
    out: dict[str, dict[str, str]] = {}
    for query_id, table in judgments.items():
        merged: dict[str, str] = {}
        for doc_id in table:
            merged.update(doc_sentences.get(doc_id, {}))
        out[query_id] = merged
    return out


def gold_chunks(query_id: str, chunks: Sequence[dict[str, Any]],
                judgments: dict[str, dict[str, dict[str, str]]],
                sentence_texts: dict[str, dict[str, str]] | None = None) -> list[str]:
    """parent_child 策略下被投影为 relevant 的子块 id（兼容旧评测脚本）。"""
    labels = chunk_labels_for_query(query_id, chunks, judgments, sentence_texts)
    return sorted(cid for cid, label in labels.items() if label == "relevant")


# ----------------------------------------------------------------- 汇总：answerable、一致率、v3 无证据复查


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa：pairs 为 (v3 判正, LLM 判正) 序列。期望一致率 1 时无定义。"""
    n = len(pairs)
    if not n:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    pos_a = sum(1 for a, _ in pairs if a)
    pos_b = sum(1 for _, b in pairs if b)
    expected = (pos_a * pos_b + (n - pos_a) * (n - pos_b)) / n
    if expected >= n:
        return None
    return (agree - expected) / (n - expected)


def query_answerable(judgments: dict[str, dict[str, dict[str, str]]], query_id: str) -> tuple[bool, bool]:
    """按句子级标注判定：至少一句 relevant 为 true；只有 partial 记 partial_only。"""
    labels = {label for table in judgments.get(query_id, {}).values() for label in table.values()}
    return "relevant" in labels, "relevant" not in labels and "partial" in labels


def v3_agreement(corpus: V4Corpus, judgments: dict[str, dict[str, dict[str, str]]],
                 queries: Sequence[dict[str, Any]],
                 sentence_texts: dict[str, dict[str, str]]) -> dict[str, Any]:
    """v3 查询上 LLM 标注与 v3 标注的一致率（按摘录，relevant/partial 视为正）。

    只比较该查询标注池内实际提交标注的摘录（摘录映射到的 parent_child 子块中至少一句
    被提交标注）：属于该查询 v3 gold 的为 v3 正例，其余为 v3 负例；标注池外摘录不参与。
    LLM 侧取该摘录映射到的子块的投影最高标签。
    """
    pc_chunks = corpus.chunks["parent_child"]
    by_id = corpus.chunk_by_id["parent_child"]
    pairs: list[tuple[bool, bool]] = []
    missed_gold: list[dict[str, Any]] = []
    out_of_pool_gold = 0
    excerpts = corpus.mapping["strategies"]["parent_child"]["excerpts"]
    skipped = sum(1 for entry in excerpts.values() if entry["mapping"] == "failed")
    chunk_hashes: dict[str, tuple[str, set[str]]] = {}

    def in_pool(query_id: str, chunk_ids: Sequence[str]) -> bool:
        table = judgments.get(query_id, {})
        for chunk_id in chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            if chunk_id not in chunk_hashes:
                doc_id = str(chunk.get("metadata", {}).get("doc_id") or chunk["source_id"])
                chunk_hashes[chunk_id] = (doc_id, {sentence_hash(s) for s in _split_sentences(chunk["text"])})
            doc_id, hashes = chunk_hashes[chunk_id]
            if hashes & set(table.get(doc_id, {})):
                return True
        return False

    for query in queries:
        gold_set = set(corpus.query_excerpt_ids(query["sample_id"]))
        labels = chunk_labels_for_query(query["sample_id"], pc_chunks, judgments, sentence_texts)
        for excerpt_id, entry in excerpts.items():
            if entry["mapping"] == "failed":
                continue
            if not in_pool(query["sample_id"], entry["gold"]):
                out_of_pool_gold += excerpt_id in gold_set
                continue
            llm_positive = any(labels.get(cid, "irrelevant") != "irrelevant" for cid in entry["gold"])
            v3_positive = excerpt_id in gold_set
            pairs.append((v3_positive, llm_positive))
            if v3_positive and not llm_positive:
                missed_gold.append({"query_id": query["sample_id"], "excerpt_id": excerpt_id,
                                    "chunk_ids": entry["gold"],
                                    "titles": sorted({by_id[c]["source"] for c in entry["gold"] if c in by_id})})
    agree = sum(1 for a, b in pairs if a == b)
    confusion = {"v3_pos_llm_pos": sum(1 for a, b in pairs if a and b),
                 "v3_pos_llm_neg": sum(1 for a, b in pairs if a and not b),
                 "v3_neg_llm_pos": sum(1 for a, b in pairs if not a and b),
                 "v3_neg_llm_neg": sum(1 for a, b in pairs if not a and not b)}
    return {"n_pairs": len(pairs), "failed_mapping_excerpts": skipped,
            "v3_gold_out_of_pool": out_of_pool_gold,
            "agreement": round(agree / len(pairs), 4) if pairs else None,
            "kappa": (lambda k: round(k, 4) if k is not None else None)(cohen_kappa(pairs)),
            "confusion": confusion, "missed_gold_count": len(missed_gold),
            "missed_gold_examples": missed_gold[:5]}


def v3_no_evidence_recheck(corpus: V4Corpus, judgments: dict[str, dict[str, dict[str, str]]],
                           queries: Sequence[dict[str, Any]],
                           sentence_texts: dict[str, dict[str, str]]) -> dict[str, Any]:
    """v3 的 56 条无证据查询在 v4 语料上的重新判定（v3 文件不改）。

    判定按句子级标注（至少一句 relevant），命中文档取投影为 relevant 的
    parent_child 子块所属文档。
    """
    rows = []
    for query in queries:
        if query["origin"] != "v3" or corpus.query_excerpt_ids(query["sample_id"]):
            continue
        answerable, _ = query_answerable(judgments, query["sample_id"])
        labels = chunk_labels_for_query(query["sample_id"], corpus.chunks["parent_child"],
                                        judgments, sentence_texts)
        hits = [cid for cid, label in labels.items() if label == "relevant"]
        doc_ids = sorted({str(corpus.chunk_by_id["parent_child"][c].get("metadata", {}).get("doc_id"))
                          for c in hits})
        source_types = sorted({str(corpus.chunk_by_id["parent_child"][c].get("metadata", {}).get("source_type"))
                               for c in hits})
        rows.append({"query_id": query["sample_id"], "bucket": query["bucket"],
                     "split": query["split"], "now_answerable": answerable,
                     "hit_docs": doc_ids, "hit_source_types": source_types})
    return {"schema_version": "v3-no-evidence-recheck-v1",
            "note": "v3 的 56 条无证据查询在 v4 语料上的重新判定（句子级标注投影到 parent_child 子块）；v3 文件不改。",
            "n_queries": len(rows), "now_answerable": sum(1 for r in rows if r["now_answerable"]),
            "queries": rows}


# ----------------------------------------------------------------- 冻结

def freeze_queries(data_dir: Path, corpus: V4Corpus,
                   judgments: dict[str, dict[str, dict[str, str]]],
                   sentence_texts: dict[str, dict[str, str]],
                   query_subset: Sequence[dict[str, Any]] | None = None) -> dict[str, str]:
    """把 answerable、gold_evidence_ids 写回 queries.jsonl（冻结）；重跑幂等，不依赖已冻结字段。

    answerable 按句子级标注（至少一句 relevant）；gold_evidence_ids 写 parent_child
    策略下投影为 relevant 的子块 id（兼容旧评测脚本；权威标注是 judgments.jsonl）。
    ``query_subset`` 非空时只更新这些查询（其余保留文件原值），便于补跑后合并；
    此时 judgments/sentence_texts 也必须已包含全部查询（由调用方保证）。
    """
    subset_ids = {q["sample_id"] for q in query_subset} if query_subset else None
    queries = corpus.queries
    for query in queries:
        if subset_ids is not None and query["sample_id"] not in subset_ids:
            continue
        answerable, partial_only = query_answerable(judgments, query["sample_id"])
        query["answerable"] = answerable
        query["partial_only"] = partial_only
        query["gold_evidence_ids"] = gold_chunks(query["sample_id"], corpus.chunks["parent_child"],
                                                 judgments, sentence_texts)
    write_jsonl(queries, data_dir / "queries.jsonl")
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("queries.jsonl", "judgments.jsonl")}


# ----------------------------------------------------------------- 主流程

# 费用估算口径同 M3：每百万输入 Token 1 元、输出 2 元（未核实实际计费，仅供量级参考）
YUAN_PER_M_PROMPT = 1.0
YUAN_PER_M_COMPLETION = 2.0
ARTIFACTS = ("queries.jsonl", "judgments.jsonl", "pools.jsonl", "v3_no_evidence_recheck.json")


def artifact_hashes(data_dir: Path) -> dict[str, str | None]:
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            if (data_dir / name).exists() else None for name in ARTIFACTS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir: Path = args.data_dir
    corpus = V4Corpus(data_dir, args.chunks)
    print(f"[load] 查询 {len(corpus.queries)} 条，"
          f"recursive 块 {len(corpus.chunks['recursive'])}，"
          f"parent_child 块 {len(corpus.chunks['parent_child'])}", flush=True)
    queries = list(corpus.queries)
    if args.limit:
        dev = [q for q in queries if q["split"] == "dev"][: args.limit // 2 + args.limit % 2]
        test = [q for q in queries if q["split"] == "test"][: args.limit // 2]
        queries = dev + test

    # 向量编码器：GPU 可用时用 GPU（与 chunking_matrix 同一加载方式，local_files_only）
    from medidiag.rag.dense import load_local_encoder

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    t0 = time.time()
    cache = EmbeddingCache(load_local_encoder(DEFAULT_EMBEDDING_MODEL, EMBEDDING_REVISION,
                                              device=device, batch_size=32 if device == "cuda" else 8))
    print(f"[pool] 向量编码器已加载（{device}，{time.time() - t0:.1f}s）", flush=True)
    scorers = build_scorers(corpus, cache)
    print(f"[pool] 4 种检索方法索引就绪（{time.time() - t0:.1f}s）", flush=True)

    # 先建全部池（确定性、离线），再统一发请求；池总数与请求数在发起付费调用前可见
    t0 = time.time()
    pools: dict[str, QueryPool] = {}
    total_requests = 0
    for index, query in enumerate(queries, 1):
        pool = build_pool(corpus, scorers, query)
        total_requests += assign_requests(pool)
        pools[query["sample_id"]] = pool
        if index % 20 == 0 or index == len(queries):
            print(f"[pool] 已建池 {index}/{len(queries)}（{time.time() - t0:.1f}s，"
                  f"累计请求 {total_requests}）", flush=True)
    print(f"[pool] 池构建完成：{len(pools)} 条查询，共 {total_requests} 次请求，"
          f"{time.time() - t0:.1f}s", flush=True)

    cassette = Cassette(args.cassette)
    if args.mode == "record":
        transport: Any = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        transport = ReplayTransport(cassette)
    provider = make_provider(transport)

    def judge_with_pool(query: dict[str, Any], pool: QueryPool) -> dict[str, Any]:
        requests = [build_judge_request(pool, query["question"], no, args.model)
                    for no in range(1, max(pool.request_ids, default=0) + 1)]
        return judge_requests(provider, pool, query["sample_id"], requests, args.workers)

    progress: Counter[str] = Counter()  # 请求状态累计（跨查询）
    tokens: Counter[str] = Counter()
    calls = 0
    t0 = time.time()
    last_print = t0
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(args.workers) as executor:
        futures = {q["sample_id"]: executor.submit(judge_with_pool, q, pools[q["sample_id"]])
                   for q in queries}
        for done, (sample_id, future) in enumerate(futures.items(), 1):
            results[sample_id] = future.result()
            for response in results[sample_id]["responses"]:
                progress[response["status"]] += 1
                calls += 1 + response.get("retried", 0)
                usage = response.get("usage") or {}
                tokens["prompt"] += int(usage.get("input_tokens") or 0)
                tokens["completion"] += int(usage.get("output_tokens") or 0)
            if done % 10 == 0 or done == len(queries) or time.time() - last_print > 30:
                last_print = time.time()
                print(f"[judge] 已完成 {done}/{len(queries)}（{last_print - t0:.0f}s，"
                      f"请求 {sum(progress.values())}/{total_requests}，调用 {calls}，"
                      f"Token prompt={tokens['prompt']} completion={tokens['completion']}，"
                      f"状态 {dict(progress)}）", flush=True)

    errors = progress["error"]
    if errors:
        # 回放时 error 即调用记录未命中；记录时 error 是网络/预算问题，重跑会复用已记录的调用
        hit = (calls - errors) / calls if calls else 0.0
        raise SystemExit(f"[{args.mode}] {errors} 次调用失败（命中率 {hit:.2%}），不写任何产物；"
                         f"失败码：{Counter(r.get('code') for q in results.values() for r in q['responses'] if r['status'] == 'error')}")

    all_judgments: list[dict[str, Any]] = []
    all_pools: list[dict[str, Any]] = []
    sentence_texts: dict[str, dict[str, str]] = {}
    for query in queries:
        result = results[query["sample_id"]]
        all_judgments.extend(judgment_rows(corpus, result))
        all_pools.append(pool_rows(corpus, result))
        sentence_texts[query["sample_id"]] = result["pool"].sentence_texts
    # 汇总所用的标注表直接来自内存结果，--limit 冒烟不落盘
    judgments: dict[str, dict[str, dict[str, str]]] = {}
    for row in all_judgments:
        judgments.setdefault(row["query_id"], {}).setdefault(row["doc_id"], {})[
            row["sentence_hash"]] = row["label"]

    n_passages = [len(row["candidates"]) for row in all_pools]
    n_sentences = [row["n_unique_sentences"] for row in all_pools]
    spent = {"prompt_tokens": tokens["prompt"], "completion_tokens": tokens["completion"]}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "mode": args.mode, "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "queries": len(queries), "requests": sum(row["n_requests"] for row in all_pools),
        "calls": calls, "retried": sum(row["retried"] for row in all_pools),
        "request_status": dict(progress),
        "judge_failed_queries": [row["query_id"] for row in all_pools if row["judge_failed"]],
        "pool_passages_mean": round(statistics.mean(n_passages), 2) if n_passages else 0,
        "pool_unique_sentences_mean": round(statistics.mean(n_sentences), 2) if n_sentences else 0,
        "judgment_rows": len(all_judgments),
        "label_counts": dict(Counter(r["label"] for r in all_judgments)),
        "tokens": {**spent, "est_yuan": round(
            (tokens["prompt"] * YUAN_PER_M_PROMPT + tokens["completion"] * YUAN_PER_M_COMPLETION) / 1e6, 3)},
    }

    # answerable 分布（按 origin × intent × split，按句子级标注）
    answerable_dist: Counter[tuple[str, str, str, str]] = Counter()
    for query in queries:
        answerable, partial_only = query_answerable(judgments, query["sample_id"])
        kind = "answerable" if answerable else ("partial_only" if partial_only else "unanswerable")
        answerable_dist[(query["origin"], query["intent"], query["split"], kind)] += 1
    summary["answerable_dist"] = {
        f"{o}/{i}/{s}/{k}": n for (o, i, s, k), n in sorted(answerable_dist.items())}

    if any(q["origin"] == "v3" for q in queries):
        summary["v3_agreement"] = v3_agreement(corpus, judgments, [q for q in queries if q["origin"] == "v3"],
                                               sentence_texts)
        recheck = v3_no_evidence_recheck(corpus, judgments, queries, sentence_texts)
        summary["v3_no_evidence_now_answerable"] = f"{recheck['now_answerable']}/{recheck['n_queries']}"
    else:
        recheck = None

    if args.mode == "record":
        summary["spend"] = transport.spent()
    if args.limit:
        return summary  # 冒烟：不写任何产物

    before = artifact_hashes(data_dir)
    write_jsonl(all_judgments, data_dir / "judgments.jsonl")
    write_jsonl(all_pools, data_dir / "pools.jsonl")
    if recheck is not None:
        (data_dir / "v3_no_evidence_recheck.json").write_text(
            json.dumps(recheck, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.skip_freeze:
        freeze_queries(data_dir, corpus, judgments, sentence_texts)
    after = artifact_hashes(data_dir)
    summary["artifact_sha256"] = after
    summary["artifacts_unchanged"] = before == after
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", choices=ALLOWED_MIMO_MODELS, default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "examples/public_health_v4")
    parser.add_argument("--chunks", type=Path, default=ROOT / ".cache/corpus/chunks")
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/v4-judgments.jsonl")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/v4-m7-ledger-r2.db")
    parser.add_argument("--max-calls", type=int, default=700)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只标注前 N 条（dev/test 各半），不写任何产物")
    parser.add_argument("--skip-freeze", action="store_true", help="只标注与汇总，不写回 queries.jsonl")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
