"""补全扩充测试集的正确片段：池化各检索方案的前 3 段，由模型判断是否回答了问题，判为是的再逐条人工复核。

扩充测试集每条可回答问题只锚定了一段正确片段，语料里可能还有别的片段同样回答了它；
不补全会把“检索到另一段同样正确的片段”算成未命中，低估所有方案，且偏向检索方式与锚定片段相近的方案。

做法（TREC 式池化）：

1. 池：给定端到端评测报告里所有条件交给生成的前 3 段，去掉已标为正确的片段；没有对应证据的问题也进池，
   若有片段被确认回答了它，说明这条问题构造有误，由 ``scripts/build_paraphrase_extension.py`` 报错。
2. 模型判断：每个（问题，片段）一次请求，只看这一段原文；调用记录可零费用回放。
3. 人工复核：只复核模型判为“回答了”的，在 ``human_label`` 填 ``answers`` 或 ``does_not_answer``；
   模型判为否的不复核，``human_label`` 留空（池化标注的常见做法，漏标只会让指标偏低，不会偏向某个方案）。

再次运行时保留已有的人工复核结果，只补新进池的条目。

用法::

    python -m eval.pooled_judgments --mode record --dataset-dir examples/public_health_v3_ext \\
        --report artifacts/reports/llm/llm-pipeline-v3-ext-20261017.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from eval.llm_pipeline_eval import CONTRACT_VIOLATIONS, call_with_retry, make_provider
from eval.retrieval_benchmark import load_dataset
from medidiag.errors import MediDiagError
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport
from medidiag.llm.contracts import LLMRequest
from medidiag.llm.models import ACTIVE_MIMO_MODEL, ALLOWED_MIMO_MODELS

ROOT = Path(__file__).resolve().parent.parent
JUDGE_PROMPT = (
    "你是检索评测的相关性标注员。给你一个用户问题和一段公共卫生科普原文，判断这段原文是否回答了问题所问的内容。"
    "原文本身包含问题要的答案（事实、数字、做法或原因），才算回答；只是话题相同、只出现了相关词语、"
    "或者需要额外知识才能推出答案的，都不算。只依据这段原文判断，不使用你自己的医学知识。"
    '返回 JSON：{"answers": true 或 false, "reason": "一句话理由，不超过 60 字"}。'
)


def build_judge_request(question: str, text: str, model: str = ACTIVE_MIMO_MODEL) -> LLMRequest:
    return LLMRequest(
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": json.dumps({"question": question, "passage": text}, ensure_ascii=False)}],
        model=model, response_format={"type": "json_object"}, max_tokens=1024,
        reasoning_mode="enabled", prompt_version="pooled-judge-v1")


def pool(queries: Sequence[dict[str, Any]], reports: Sequence[dict[str, Any]],
         keep: frozenset[tuple[str, str]] = frozenset()) -> list[dict[str, Any]]:
    """（问题，片段）对，附上哪些条件检索到了它；顺序固定，便于复核与回放。

    ``keep`` 是已经人工确认、因而已并入正确片段的对：它们仍留在池中，否则重跑时会丢掉复核记录。
    """
    by_id = {q["sample_id"]: q for q in queries}
    found: dict[tuple[str, str], set[str]] = {}
    for report in reports:
        for row in report["per_query"]:
            gold = set(by_id[row["sample_id"]]["gold_evidence_ids"])
            for condition, outcome in row["conditions"].items():
                for chunk_id in outcome["retrieved"]:
                    if chunk_id not in gold or (row["sample_id"], chunk_id) in keep:
                        found.setdefault((row["sample_id"], chunk_id), set()).add(condition)
    return [{"sample_id": sid, "chunk_id": cid, "conditions": sorted(found[(sid, cid)])} for sid, cid in sorted(found)]


def judge(provider: Any, question: str, text: str, model: str) -> dict[str, Any]:
    request = build_judge_request(question, text, model)
    key = "judge-" + hashlib.sha256(f"{question}\n{text}".encode()).hexdigest()[:24]
    try:
        result = call_with_retry(lambda: provider.generate(request, timeout_s=45, idempotency_key=key))
    except MediDiagError as exc:
        return {"model_label": "invalid" if exc.code in CONTRACT_VIOLATIONS else "error", "model_reason": exc.code}
    answers = (result.parsed_json or {}).get("answers")
    if not isinstance(answers, bool):
        return {"model_label": "invalid", "model_reason": "JUDGE_SHAPE"}
    reason = (result.parsed_json or {}).get("reason")
    return {"model_label": "answers" if answers else "does_not_answer",
            "model_reason": reason.strip()[:120] if isinstance(reason, str) else ""}


def run(dataset: Any, reports: Sequence[dict[str, Any]], provider: Any, model: str, workers: int,
        existing: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    queries = {q["sample_id"]: q for q in dataset.queries}
    text = {c["chunk_id"]: c["text"] for c in dataset.chunks}
    confirmed = frozenset(key for key, row in existing.items() if row.get("human_label") == "answers")
    items = pool(dataset.queries, reports, confirmed)

    def one(item: dict[str, Any]) -> dict[str, Any]:
        verdict = judge(provider, queries[item["sample_id"]]["question"], text[item["chunk_id"]], model)
        old = existing.get((item["sample_id"], item["chunk_id"]), {})
        return {**item, **verdict, "human_label": old.get("human_label"), "human_note": old.get("human_note", "")}

    with ThreadPoolExecutor(workers) as executor:
        return list(executor.map(one, items))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--model", choices=ALLOWED_MIMO_MODELS, default=ACTIVE_MIMO_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "examples/public_health_v3_ext")
    parser.add_argument("--report", type=Path, nargs="+", required=True, help="端到端评测报告，取其中各条件的前 3 段")
    parser.add_argument("--cassette", type=Path, default=ROOT / "eval/cassettes/pooled-judgments-v3-ext.jsonl")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".cache/pooled-judgments-ledger.db")
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None, help="默认写到数据集目录下的 pooled_judgments.jsonl")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_dir)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    out = args.out or args.dataset_dir / "pooled_judgments.jsonl"
    existing = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(row["sample_id"], row["chunk_id"])] = row
    cassette = Cassette(args.cassette)
    if args.mode == "record":
        transport: Any = RecordingTransport(cassette, args.ledger, max_calls=args.max_calls)
    else:
        os.environ["MIMO_API_KEY"] = "replay-no-network"  # 回放不联网，也不使用真实密钥
        os.environ["MIMO_BASE_URL"] = "https://api.xiaomimimo.com"
        transport = ReplayTransport(cassette)
    rows = run(dataset, reports, make_provider(transport), args.model, args.workers, existing)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n")
    labels: dict[str, int] = {}
    for row in rows:
        labels[row["model_label"]] = labels.get(row["model_label"], 0) + 1
    print(f"池中 {len(rows)} 对；模型判断 {labels}；待人工复核 "
          f"{sum(1 for r in rows if r['model_label'] == 'answers' and r['human_label'] is None)} 对：{out}")
    if args.mode == "record":
        print("花费：", transport.spent())


if __name__ == "__main__":
    main()
