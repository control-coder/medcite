"""蕴含判定的评审检查（M13）：程序构造约 100 个（句子, quote）对，检查生产路径的 entailment-v1 判定器。

不做人工标注。quote 取自 v4 父子子块里含数字或否定词的中文句子（dev 可回答查询的 relevant 块，固定随机种子）：

- 正例约 40：quote 原句；只改标点、空白的句子；
- 负例约 60，各约 15：改动数字、插入否定词、删除否定词（仅原句含否定的）、换成另一篇文档的 quote（句子不变）。

判定使用生产路径的同一请求构造（``generative_answer.entailment_messages``，参数同 ``_entail``），每批 5 对模拟生产批量。
报告各负例类别的检出率（判 not_entailed 的比例，另列 partial）和正例误判率；确定性校验（``validate_sentences``）
是否也会拦下这些样本单独列出，因为生产中数值改动多半先被它拦住，评审检查针对的是模型判定本身。

用法::

    python -m eval.entailment_check --mode record --limit 1   # 冒烟（1 批），不写报告
    python -m eval.entailment_check --mode record
    python -m eval.entailment_check --mode replay

默认 r2：判定容错与生产路径一致（``entailed`` 字段同义，非法 JSON 或结构不合规时附纠错说明重试一次）。
旧调用记录作为只读 base，命中即不再请求；需要重试的新请求才真实调用，写入 ``v4-entailment-check-r2.jsonl``。
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval.agent_v2_eval import dev_queries  # noqa: E402
from eval.hybrid_matrix import CHUNKS_PATH, JUDGMENTS_PATH, _load_judgments_module  # noqa: E402
from eval.llm_pipeline_eval import call_with_retry, make_provider  # noqa: E402
from medidiag.errors import MediDiagError  # noqa: E402
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport  # noqa: E402
from medidiag.llm.contracts import LLMRequest  # noqa: E402
from medidiag.llm.models import ACTIVE_MIMO_MODEL  # noqa: E402
from medidiag.rag.ingest.chunking import _split_sentences  # noqa: E402
from medidiag.workflow.generative_answer import (  # noqa: E402
    _NEGATION,
    ENTAILMENT_MAX_TOKENS,
    ENTAILMENT_VERSION,
    MAX_ENTAILMENT_CALLS,
    InvalidAnswer,
    _norm,
    entailment_messages,
    entailment_retry_messages,
    parse_verdicts,
    validate_sentences,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "entailment-check-v1"
REPORT_DIR = ROOT / "artifacts/reports/generation"
SEED = 20261028
BATCH = 5
TARGETS = {"pos_exact": 20, "pos_punct": 20, "neg_number": 15, "neg_insert_negation": 15,
           "neg_delete_negation": 15, "neg_swap_quote": 15}
NEGATIVES = tuple(c for c in TARGETS if c.startswith("neg_"))
LOW_DETECTION = 0.8  # 检出率低于它的负例类别单列
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# 插入否定：词 -> 否定形式（只对不含否定词的句子）；删除否定：否定形式 -> 肯定形式（顺序即优先级）
_INSERT = (("可以", "不可以"), ("能够", "不能够"), ("需要", "不需要"), ("应该", "不应该"), ("应当", "不应当"),
           ("建议", "不建议"), ("容易", "不容易"), ("必须", "不必"), ("属于", "不属于"), ("具有", "不具有"),
           ("适合", "不适合"), ("可能", "不可能"))
_DELETE = (("不可以", "可以"), ("不能够", "能够"), ("不需要", "需要"), ("不应该", "应该"), ("不建议", "建议"),
           ("不宜", "宜"), ("不能", "能"), ("不会", "会"), ("不要", "要"), ("无需", "需"), ("没有", "有"),
           ("不是", "是"), ("不应", "应"))
_PUNCT = str.maketrans({"，": ", ", "；": "; ", "：": ": ", "（": " (", "）": ") "})


# ----------------------------------------------------------------- 构造变换（纯函数）

def change_number(sentence: str) -> str | None:
    """把第一个数字换成不同的数值（2n+1，小数保留位数）；没有数字返回 None。"""
    match = _NUMBER.search(sentence)
    if not match:
        return None
    text = match.group()
    if "." in text:
        decimals = len(text.split(".")[1])
        new = f"{float(text) * 2 + 1:.{decimals}f}"
    else:
        new = str(int(text) * 2 + 1)
    return sentence[:match.start()] + new + sentence[match.end():]


def insert_negation(sentence: str) -> str | None:
    """在第一个可否定的词前加否定；原句已含否定词或没有可改的词时返回 None。"""
    if _NEGATION.search(sentence):
        return None
    for word, negated in _INSERT:
        if word in sentence:
            return sentence.replace(word, negated, 1)
    return None


def delete_negation(sentence: str) -> str | None:
    """去掉第一个否定表达；原句没有可去的否定时返回 None。"""
    for negated, positive in _DELETE:
        if negated in sentence:
            return sentence.replace(negated, positive, 1)
    return None


def punct_variant(sentence: str) -> str | None:
    """只改标点与空白：全角标点换半角（带空格）并去掉句末句号；无可改处返回 None。"""
    changed = sentence.translate(_PUNCT).rstrip("。")
    changed = changed.replace("。", ". ")
    return changed if _norm(changed) != _norm(sentence) or changed != sentence else None


# ----------------------------------------------------------------- 样本构造

def load_candidates() -> list[dict[str, Any]]:
    """dev 可回答查询 relevant 子块里含数字或否定词、长度 10–90 字的句子，按句子文本去重并排序（可复现）。"""
    from medidiag.schemas import read_jsonl

    load_judgments, chunk_labels_for_query, recover_sentence_texts = _load_judgments_module()
    chunks = read_jsonl(CHUNKS_PATH)
    judgments = load_judgments(JUDGMENTS_PATH)
    texts = recover_sentence_texts({"parent_child": chunks}, judgments)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_doc.setdefault(str(chunk.get("metadata", {}).get("doc_id") or chunk["source_id"]), []).append(chunk)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for query in sorted((q for q in dev_queries() if q.get("answerable") is True), key=lambda q: q["sample_id"]):
        qid = query["sample_id"]
        doc_chunks = [c for doc_id in judgments.get(qid, {}) for c in by_doc.get(doc_id, [])]
        labels = chunk_labels_for_query(qid, doc_chunks, judgments, texts)
        for chunk in sorted(doc_chunks, key=lambda c: c["chunk_id"]):
            if labels.get(chunk["chunk_id"]) not in ("relevant", "partial"):
                continue
            doc_id = str(chunk.get("metadata", {}).get("doc_id") or chunk["source_id"])
            for sentence in _split_sentences(chunk["text"]):
                sentence = sentence.strip()
                if (10 <= len(sentence) <= 90 and re.search(r"[\u4e00-\u9fff]", sentence) and sentence not in seen):
                    seen.add(sentence)
                    out.append({"sentence": sentence, "chunk_id": chunk["chunk_id"], "doc_id": doc_id, "query_id": qid,
                                "label": labels[chunk["chunk_id"]],
                                "narrow": labels[chunk["chunk_id"]] == "relevant"
                                and bool(_NUMBER.search(sentence) or _NEGATION.search(sentence))})
    return out


def build_pairs(candidates: Sequence[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    """按类别从候选里抽样并变换；某类别候选不足时按实际数量构造（报告里写 n）。"""
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []

    narrow = [c for c in candidates if c["narrow"]]

    def pick(category: str, transform: Any, broad: bool = False) -> None:
        """优先用窄池（relevant 块里含数字或否定词的句子）；数量不足时用宽池（relevant 与 partial 块的全部句子）补足。"""
        pool = list(narrow)
        rng.shuffle(pool)
        if broad:
            rest = [c for c in candidates if not c["narrow"]]
            rng.shuffle(rest)
            pool += rest
        count = 0
        for cand in pool:
            new = transform(cand["sentence"]) if transform else cand["sentence"]
            if new is None:
                continue
            pairs.append({"category": category, "sentence": new, "quote": cand["sentence"], "quote_chunk_id": cand["chunk_id"],
                          "source_sentence": cand["sentence"]})
            count += 1
            if count >= TARGETS[category]:
                return

    pick("pos_exact", None)
    pick("pos_punct", punct_variant)
    pick("neg_number", change_number)
    pick("neg_insert_negation", insert_negation, broad=True)
    pick("neg_delete_negation", delete_negation, broad=True)
    # 换 quote：句子不变，quote 取自另一篇文档的候选句
    pool = list(narrow)
    rng.shuffle(pool)
    for cand in pool[: TARGETS["neg_swap_quote"] * 4]:
        others = [o for o in narrow if o["doc_id"] != cand["doc_id"] and o["query_id"] != cand["query_id"]]
        if not others:
            continue
        other = rng.choice(others)
        pairs.append({"category": "neg_swap_quote", "sentence": cand["sentence"], "quote": other["sentence"],
                      "quote_chunk_id": other["chunk_id"], "source_sentence": cand["sentence"]})
        if sum(1 for p in pairs if p["category"] == "neg_swap_quote") >= TARGETS["neg_swap_quote"]:
            break
    rng.shuffle(pairs)
    for i, pair in enumerate(pairs):
        pair["id"] = i
    return pairs


# ----------------------------------------------------------------- 判定与汇总

def production_flag(sentence: str, quote: str) -> bool:
    """同 ``validate_sentences`` 的规则 5：句有否定词而引用没有。"""
    return bool(_NEGATION.search(sentence)) and not _NEGATION.search(_norm(quote))


def rules_catch(pair: dict[str, Any], chunk_text: str) -> bool:
    """确定性校验（规则 1–4）是否会拦下该样本。"""
    try:
        validate_sentences([(pair["sentence"], [(pair["quote_chunk_id"], pair["quote"])])],
                           {pair["quote_chunk_id"]: chunk_text})
    except InvalidAnswer:
        return True
    return False


def entailment_request(items: Sequence[dict[str, Any]], model: str) -> LLMRequest:
    """与生产 ``_chat`` 相同的请求参数。"""
    return LLMRequest(messages=entailment_messages(items), model=model, response_format={"type": "json_object"},
                      max_tokens=ENTAILMENT_MAX_TOKENS, reasoning_mode="disabled", prompt_version=ENTAILMENT_VERSION)


def judge_with_retry(provider: Any, items: Sequence[dict[str, Any]], model: str, tokens: Counter[str],
                     errors: Counter[str], retries: Counter[str]) -> list[str] | None:
    """与生产 ``_entail`` 同一套容错：非法 JSON 或结构不合规时附纠错说明重试一次，仍失败返回 None（该批视为未核实）。

    ``entailed`` 字段名同义已在 ``parse_verdicts`` 内处理；超时等其他错误不重试，只计入 errors。
    """
    base = entailment_request(items, model)
    for attempt in range(MAX_ENTAILMENT_CALLS):
        request = base if attempt == 0 else replace(base, messages=entailment_retry_messages(base.messages, len(items)))
        if attempt:
            retries["retry_calls"] += 1
        try:
            result = call_with_retry(functools.partial(
                provider.generate, request, timeout_s=45, idempotency_key="m13-ent-" + uuid.uuid4().hex))
        except MediDiagError as exc:
            errors[exc.code] += 1
            if exc.code != "STRUCTURED_OUTPUT_INVALID":
                return None
            continue
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            tokens[name] += int((result.usage or {}).get(name) or 0)
        verdicts: list[str] | None = parse_verdicts(result.parsed_json, len(items))
        if verdicts is not None:
            if attempt:
                retries["retry_recovered"] += 1
            return verdicts
        errors["ENTAILMENT_SHAPE"] += 1
    return None


def summarize(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, dict[str, Any]] = {}
    for category in TARGETS:
        rows = [p for p in pairs if p["category"] == category]
        judged = [p for p in rows if p.get("verdict") in ("entailed", "partial", "not_entailed")]
        counts = Counter(p["verdict"] for p in judged)
        n = len(judged)
        entry: dict[str, Any] = {"n": len(rows), "judged": n, "verdicts": dict(counts)}
        if category.startswith("pos_"):
            entry["false_reject_rate"] = round(counts["not_entailed"] / n, 4) if n else None
            entry["partial_rate"] = round(counts["partial"] / n, 4) if n else None
        else:
            entry["detection_rate"] = round(counts["not_entailed"] / n, 4) if n else None
            entry["detection_or_partial_rate"] = round((counts["not_entailed"] + counts["partial"]) / n, 4) if n else None
            entry["missed_as_entailed"] = counts["entailed"]
        entry["rules_catch_rate"] = round(sum(1 for p in rows if p["rules_catch"]) / len(rows), 4) if rows else None
        by_cat[category] = entry
    neg = [p for p in pairs if p["category"] in NEGATIVES and p.get("verdict")]
    pos = [p for p in pairs if p["category"].startswith("pos_") and p.get("verdict")]
    return {
        "n_pairs": len(pairs), "n_positive": sum(1 for p in pairs if p["category"].startswith("pos_")),
        "n_negative": sum(1 for p in pairs if p["category"] in NEGATIVES), "by_category": by_cat,
        "overall_detection_rate": round(sum(p["verdict"] == "not_entailed" for p in neg) / len(neg), 4) if neg else None,
        "overall_false_reject_rate": round(sum(p["verdict"] == "not_entailed" for p in pos) / len(pos), 4) if pos else None,
        "low_detection_categories": [c for c in NEGATIVES if (by_cat[c].get("detection_rate") or 0) < LOW_DETECTION
                                     and by_cat[c]["judged"]],
        "unjudged": sum(1 for p in pairs if not p.get("verdict")),
        "shape_error_pairs": sum(1 for p in pairs if p.get("shape_error"))}


LIMITATIONS = [
    "样本由程序按固定规则构造，覆盖面有限：数字改动是 2n+1 替换，否定改动是词表替换，不代表真实生成错误的分布。",
    "判定器与被评对象同源（均为 MiMo），有偏差；判定不等于医学正确。",
    "「换成无关 quote」取自另一篇文档中含数字或否定词的句子，多半话题明显不同，比真实的错引容易检出。",
]


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_candidates()
    pairs = build_pairs(candidates)
    chunk_texts = {}
    from medidiag.schemas import read_jsonl

    wanted = {p["quote_chunk_id"] for p in pairs}
    for chunk in read_jsonl(CHUNKS_PATH):
        if chunk["chunk_id"] in wanted:
            chunk_texts[chunk["chunk_id"]] = chunk["text"]
    for pair in pairs:
        pair["flag"] = production_flag(pair["sentence"], pair["quote"])
        pair["rules_catch"] = rules_catch(pair, chunk_texts[pair["quote_chunk_id"]])
    print(f"[build] 候选句 {len(candidates)}，构造 {len(pairs)} 对：{dict(Counter(p['category'] for p in pairs))}", flush=True)

    cassette = Cassette(args.cassette, base=Cassette(args.base_cassette) if args.base_cassette else None)
    if args.mode == "record":
        inner: Any = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        inner = ReplayTransport(cassette)
    provider = make_provider(inner)
    batches = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]
    if args.limit:
        batches = batches[: args.limit]
    tokens: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    retries: Counter[str] = Counter()
    for n, batch in enumerate(batches, 1):
        items = [{"id": i, "sentence": p["sentence"], "evidence": p["quote"], "check_negation": p["flag"]}
                 for i, p in enumerate(batch)]
        verdicts = judge_with_retry(provider, items, args.model, tokens, errors, retries)
        for i, pair in enumerate(batch):
            pair["verdict"] = verdicts[i] if verdicts else None
            pair["shape_error"] = verdicts is None
        print(f"[judge] {n}/{len(batches)}", flush=True)
    scored = [p for b in batches for p in b] if args.limit else pairs
    cost: dict[str, Any] = {"batches": len(batches), "batch_size": BATCH, "call_errors": dict(errors),
                            "input_tokens": tokens["input_tokens"], "output_tokens": tokens["output_tokens"],
                            "entailment_retry": {"retry_calls": retries["retry_calls"],
                                                 "retry_recovered": retries["retry_recovered"]}}
    if args.mode == "record":
        cost["spend"] = inner.spent()
    return {"schema_version": SCHEMA_VERSION, "date": str(date.today()), "mode": args.mode, "model": args.model,
            "seed": SEED, "prompt_version": ENTAILMENT_VERSION, "cost": cost,
            "summary": summarize(scored), "limitations": LIMITATIONS, "pairs": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--revision", default="r2", help="报告后缀；r2 是加入字段同义与纠错重试后的检查，旧报告不动")
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/v4-entailment-check-r2.jsonl")
    parser.add_argument("--base-cassette", type=Path, default=ROOT / "eval/cassettes/v4-entailment-check.jsonl",
                        help="只读的旧调用记录；命中则不再请求，新记录只写入 --cassette")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/v4-router-v2-ledger.db")
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="冒烟：只判前 N 批，不写报告")
    args = parser.parse_args()

    report = run(args)
    path = REPORT_DIR / f"entailment-check-v4-{date.today():%Y%m%d}-{args.revision}.json"
    brief = {k: report[k] for k in ("cost", "summary")}
    if args.limit:
        print(json.dumps(brief, ensure_ascii=False, indent=2))
        return
    if args.mode == "replay":
        if not path.exists():
            print("没有找到 record 报告")
            return
        recorded = json.loads(path.read_text(encoding="utf-8"))
        current = json.loads(json.dumps(report["pairs"], ensure_ascii=False))
        differing = [a["id"] for a, b in zip(recorded["pairs"], current, strict=False) if a != b]
        consistent = not differing and len(recorded["pairs"]) == len(current)
        print("replay 一致" if consistent else f"replay 不一致：{differing[:10]}", flush=True)
        if consistent:
            recorded["replay_check"] = {"consistent": True, "n": len(current)}
            path.write_text(json.dumps(recorded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"[done] 报告 {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
