"""多轮追问评测集与评测报告：数据集可复现、继承关系正确，入库报告可以零网络重算与回放。

CI 可运行的部分不需要本地模型：数据集重建、只靠逐条结果重算指标、上限参照与原有端到端报告一致、
问题补全与生成步骤从调用记录回放（生成用报告里记录的 3 段证据）。检索 Agent 依赖向量检索与重排模型，
完整回放只在本机进行：设置 ``MEDIDIAG_MODEL_REPLAY=1`` 且本地有两个模型缓存与 GPU 时运行，CI 中跳过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from eval import multiturn_eval as mt
from eval.llm_pipeline_eval import FEEDBACK_BASE, grounded_decision, make_provider
from medidiag.llm.cassette import Cassette, ReplayTransport
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.workflow.context_rewrite import contextualize
from scripts import build_multiturn_set as bm

ROOT = Path(__file__).resolve().parents[1]
DIALOGUES = ROOT / "examples/public_health_v3_multiturn/dialogues.jsonl"
REPORT = ROOT / "artifacts/reports/llm/multiturn-v3-20261022.json"
CASSETTE = ROOT / "eval/cassettes/multiturn-v3.jsonl"
FEEDBACK = ROOT / "eval/cassettes/multiturn-v3-feedback.jsonl"
SOURCE_REPORTS = {"v3": ROOT / "artifacts/reports/llm/llm-pipeline-v3-rerank-20261017.json",
                  "v3_ext": ROOT / "artifacts/reports/llm/llm-pipeline-v3-ext-20261017.json"}
BASES = [ROOT / "eval/cassettes/llm-pipeline-v3-rerank.jsonl", ROOT / "eval/cassettes/llm-pipeline-v3-flash.jsonl",
         ROOT / "eval/cassettes/llm-pipeline-v3-ext.jsonl"]
FEEDBACK_BASES = [ROOT / "eval/cassettes/llm-pipeline-v3-rerank-feedback.jsonl",
                  ROOT / "eval/cassettes/llm-pipeline-v3-flash-feedback.jsonl",
                  ROOT / "eval/cassettes/llm-pipeline-v3-ext-feedback.jsonl"]


def _chain(path: Path, bases: list[Path]) -> Cassette:
    base = None
    for item in reversed(bases):
        base = Cassette(item, base=base)
    return Cassette(path, base=base)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dialogues() -> list[dict[str, Any]]:
    return mt.load_dialogues(DIALOGUES)


@pytest.fixture(scope="module")
def replay(request: pytest.FixtureRequest) -> dict[int, ReplayTransport]:
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setenv("MIMO_API_KEY", "replay-no-network")
    mp.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    return {0: ReplayTransport(_chain(CASSETTE, BASES), trial=0),
            FEEDBACK_BASE: ReplayTransport(_chain(FEEDBACK, FEEDBACK_BASES), trial=FEEDBACK_BASE)}


# ----------------------------------------------------------------- 数据集

def test_dataset_is_reproducible_from_the_builder() -> None:
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bm.build())
    assert content == DIALOGUES.read_text(encoding="utf-8")


def test_follow_ups_inherit_labels_and_split_from_their_source(dialogues) -> None:
    sources = {name: {q["sample_id"]: q for q in map(json.loads, (path / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
               for name, path in bm.SOURCES.items()}
    assert len(dialogues) == 96 and len({d["sample_id"] for d in dialogues}) == 96
    for d in dialogues:
        source = sources[d["source_dataset"]][d["source_sample_id"]]
        assert (d["split"], d["bucket"], d["gold_evidence_ids"], d["source_question"]) == (
            source["split"], source["bucket"], source["gold_evidence_ids"], source["question"])
        assert d["sample_id"].startswith("pmt_T_" if d["split"] == "test" else "pmt_D_")
        assert not any(t in d["follow_up"] for t in d["topic_terms"])
        assert not set(d["history"][0]["answer_chunk_ids"]) & set(d["gold_evidence_ids"])  # 历史不含本轮答案
    assert {d["source_dataset"] for d in dialogues if d["split"] == "dev"} == {"v3"}


def test_concat_and_condition_questions() -> None:
    dialogue = {"sample_id": "x", "history": [{"question": "麻疹是什么？", "outcome": "abstained", "answer": []}],
                "follow_up": "它有哪些症状？", "source_question": "麻疹有哪些症状？"}
    rewrites = {"x": ("麻疹有哪些症状？", {"status": "ok"})}
    assert mt.concat_question(dialogue) == "麻疹是什么？ 它有哪些症状？"
    assert [mt.questions_for(dialogue, c, rewrites) for c in mt.CONDITIONS] == [
        "它有哪些症状？", "麻疹是什么？ 它有哪些症状？", "麻疹有哪些症状？", "麻疹有哪些症状？"]
    assert mt.turns_of({"history": [{"question": "q", "outcome": "answered", "answer": ["a"], "answer_chunk_ids": ["c"]}]}) == [
        {"question": "q", "outcome": "answered", "answer": ["a"]}]


# ----------------------------------------------------------------- 入库报告（不需要模型）

def test_report_fingerprints_match_the_committed_files(report) -> None:
    config = report["config"]
    assert config["dialogues_sha256"] == hashlib.sha256(DIALOGUES.read_bytes()).hexdigest()
    assert config["cassette_sha256"] == hashlib.sha256(CASSETTE.read_bytes()).hexdigest()
    assert config["feedback_cassette_sha256"] == hashlib.sha256(FEEDBACK.read_bytes()).hexdigest()
    for path in [*BASES, *FEEDBACK_BASES]:
        assert config["base_cassette_sha256"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_metrics_recompute_from_the_per_query_rows(report) -> None:
    rebuilt = mt.summarize_rows(report["per_query"], report["config"]["conditions"])
    for key, value in rebuilt.items():
        assert report[key] == value, key


def test_oracle_matches_the_original_end_to_end_runs(report, dialogues) -> None:
    """上限参照与 v3 / v3-ext 端到端评测的向量 Agent 加重排条件请求相同，证据和最终结果必须逐条一致。"""
    sources = {name: {r["sample_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["per_query"]}
               for name, path in SOURCE_REPORTS.items()}
    by_id = {d["sample_id"]: d for d in dialogues}
    for row in report["per_query"]:
        d = by_id[row["sample_id"]]
        original = sources[d["source_dataset"]][d["source_sample_id"]]["conditions"]["dense_agent_rerank"]
        oracle = row["conditions"]["oracle_standalone"]
        assert oracle["retrieved"] == original["retrieved"], row["sample_id"]
        first, recorded = oracle["trials"][0], original["trials"][0]
        assert (first.get("feedback_retry", first)["decision"], first.get("feedback_retry", first)["selected"]) == (
            recorded.get("feedback_retry", recorded)["decision"], recorded.get("feedback_retry", recorded)["selected"])


def test_context_rewrites_replay_without_network(report, dialogues, replay) -> None:
    provider = make_provider(replay[0])
    test = [d for d in dialogues if d["split"] == "test"]
    for d in test:
        question, info = contextualize(provider, d["follow_up"], mt.turns_of(d), ACTIVE_MIMO_MODEL)
        assert {"standalone_question": question, **info} == report["rewrites"][d["sample_id"]]
    assert report["rewrite"]["status"] == {"ok": len(test)}


def test_generation_replays_from_the_recorded_evidence(report, replay) -> None:
    providers = {t: make_provider(r) for t, r in replay.items()}
    chunks = {c["chunk_id"]: c["text"] for c in map(json.loads, (ROOT / "examples/public_health_v3/chunks.jsonl")
                                                 .read_text(encoding="utf-8").splitlines())}
    for row in report["per_query"]:
        for entry in row["conditions"].values():
            ranked = [(cid, chunks[cid]) for cid in entry["retrieved"]]
            recorded = entry["trials"][0]
            first = grounded_decision(providers[0], entry["question"], ranked, ACTIVE_MIMO_MODEL)
            assert (first["decision"], first["selected"]) == (recorded["decision"], recorded["selected"])
            if "feedback_retry" in recorded:
                again = grounded_decision(providers[FEEDBACK_BASE], entry["question"], ranked, ACTIVE_MIMO_MODEL,
                                          feedback=first.get("detail"))
                assert (again["decision"], again["selected"]) == (
                    recorded["feedback_retry"]["decision"], recorded["feedback_retry"]["selected"])


def test_headline_numbers_quoted_in_the_docs(report) -> None:
    hit = {c: report["retrieval"][c]["hit@3"]["k"] for c in mt.CONDITIONS}
    useful = {c: report["useful_answers"][c]["k"] for c in mt.CONDITIONS}
    false = {c: report["false_answers"][c]["k"] for c in mt.CONDITIONS}
    assert hit == {"last_turn_only": 31, "concat_history": 43, "context_rewrite": 46, "oracle_standalone": 44}
    assert useful == {"last_turn_only": 9, "concat_history": 35, "context_rewrite": 42, "oracle_standalone": 41}
    assert false == {"last_turn_only": 0, "concat_history": 3, "context_rewrite": 0, "oracle_standalone": 0}


# ----------------------------------------------------------------- 完整回放（本机，需要模型）

@pytest.mark.skipif(os.environ.get("MEDIDIAG_MODEL_REPLAY") != "1", reason="需要本地向量与重排模型缓存，设置 MEDIDIAG_MODEL_REPLAY=1 时运行")
def test_full_replay_reproduces_the_report(report, dialogues, replay) -> None:
    import torch

    from eval.rerank_matrix import CachedScorer
    from eval.retrieval_benchmark import load_dataset
    from medidiag.rag.rerank import load_local_reranker

    selected = report["config"]["rerank"]
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 录制在 GPU 上进行；CPU 与 GPU 打分差在 1e-6 量级
    scorer = CachedScorer(load_local_reranker(selected["model"], selected["revision"], device=device))
    args = argparse.Namespace(conditions=list(mt.CONDITIONS), splits=["test"], limit=0, workers=4, rewrite_only=False,
                              model=ACTIVE_MIMO_MODEL, embedding_revision="7999e1d3359715c523056ef9478215996d62a620",
                              rerank_selected=selected)
    result = mt.run(args, replay, dialogues, load_dataset(ROOT / "examples/public_health_v3"), scorer)
    for key in ("per_query", "retrieval", "generation_with_feedback_retry", "paired_comparisons", "rewrites", "agent"):
        assert result[key] == report[key], key
