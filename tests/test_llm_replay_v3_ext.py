"""扩充测试集（v3 语料 + 163 条新问题）的调用记录离线回放，指标必须与入库报告逐项一致。

只回放不需要本地模型的 BM25 三个条件：固定检索、查询改写、检索 Agent。向量检索与重排条件依赖本地嵌入模型和
交叉编码器，CI 不下载模型，这些条件只能对照入库报告核对。同时核对 v3 重排报告里的五个原有条件与 2026-10-10 的
v3 报告逐项一致：新增条件只读旧记录，没有改变原有条件的结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from eval import llm_pipeline_eval as ev
from eval.retrieval_benchmark import load_dataset
from medidiag.llm.cassette import Cassette, ReplayTransport
from medidiag.llm.models import ACTIVE_MIMO_MODEL

ROOT = Path(__file__).resolve().parent.parent
CASSETTES = {
    "cassette_sha256": ROOT / "eval/cassettes/llm-pipeline-v3-ext.jsonl",
    "retry_cassette_sha256": ROOT / "eval/cassettes/llm-pipeline-v3-ext-retry.jsonl",
    "feedback_cassette_sha256": ROOT / "eval/cassettes/llm-pipeline-v3-ext-feedback.jsonl",
}
REPORT = ROOT / "artifacts/reports/llm/llm-pipeline-v3-ext-20261017.json"
V3_OLD = ROOT / "artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json"
V3_RERANK = ROOT / "artifacts/reports/llm/llm-pipeline-v3-rerank-20261017.json"
CONDITIONS = ["bm25", "bm25_rewrite", "bm25_agent"]


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def replayed(request: pytest.FixtureRequest) -> dict:
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setenv("MIMO_API_KEY", "replay-no-network")
    mp.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    main, retry, feedback = (Cassette(path) for path in CASSETTES.values())
    transports = {t: ReplayTransport(main, trial=t) for t in range(3)}
    transports.update({ev.RETRY_BASE + t: ReplayTransport(retry, trial=ev.RETRY_BASE + t) for t in range(3)})
    transports.update({ev.FEEDBACK_BASE + t: ReplayTransport(feedback, trial=ev.FEEDBACK_BASE + t) for t in range(3)})
    args = argparse.Namespace(conditions=CONDITIONS, repeat_trials=3, workers=4, limit=0, embedding_revision=None,
                              retry_on_violation=True, feedback_retry=True, model=ACTIVE_MIMO_MODEL)
    return ev.run(args, transports, load_dataset(ROOT / "examples/public_health_v3_ext"))


def test_recordings_and_dataset_match_report_fingerprints(committed: dict) -> None:
    for key, path in CASSETTES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == committed["config"][key], path.name
    assert load_dataset(ROOT / "examples/public_health_v3_ext").fingerprints == committed["config"]["dataset_sha256"]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_generation_metrics_reproduce(replayed: dict, committed: dict, condition: str) -> None:
    assert replayed["generation"][condition] == committed["generation"][condition]
    assert (replayed["generation_with_feedback_retry"][condition]
            == committed["generation_with_feedback_retry"][condition])


def test_agent_traces_and_retrieval_reproduce(replayed: dict, committed: dict) -> None:
    assert replayed["agent"]["bm25_agent"] == committed["agent"]["bm25_agent"]
    for condition in CONDITIONS:
        assert replayed["retrieval"]["matrix"][condition] == committed["retrieval"]["matrix"][condition]


def test_no_false_answers_on_near_miss_questions(committed: dict) -> None:
    """文档引用的结论：36 条没有对应证据的问题上，十个条件的误答都为 0。"""
    for key in ("generation", "generation_with_feedback_retry"):
        for condition, trials in committed[key].items():
            for metrics in trials.values():
                assert metrics["false_answer_rate"]["k"] == 0, (key, condition)


def test_rerank_report_keeps_the_original_v3_conditions() -> None:
    old, new = (json.loads(path.read_text(encoding="utf-8")) for path in (V3_OLD, V3_RERANK))
    for condition in old["config"]["conditions"]:
        assert new["generation"][condition] == old["generation"][condition], condition
        assert new["generation_with_feedback_retry"][condition] == old["generation_with_feedback_retry"][condition]
    for condition in ("bm25_agent", "dense_agent"):
        assert new["agent"][condition] == old["agent"][condition]
