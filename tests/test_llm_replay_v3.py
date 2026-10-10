"""用仓库内 mimo-v2.6-flash 在 v3 语料（131 篇、498 条摘录）上的调用记录零网络回放，指标必须与入库报告逐项一致。

涵盖固定流程（BM25、BM25 加改写）和检索智能体（BM25）。向量检索的两个条件依赖本地嵌入模型，不在此回放。
改动提示词、检索分词、证据排序、智能体的提示词或工具定义，都会使请求哈希变化而导致调用记录未命中，
这时需要重新调用并记录，而不是悄悄改测试。
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
    "cassette_sha256": ROOT / "eval/cassettes/llm-pipeline-v3-flash.jsonl",
    "retry_cassette_sha256": ROOT / "eval/cassettes/llm-pipeline-v3-flash-retry.jsonl",
    "feedback_cassette_sha256": ROOT / "eval/cassettes/llm-pipeline-v3-flash-feedback.jsonl",
}
REPORT = ROOT / "artifacts/reports/llm/llm-pipeline-v3-flash-20261010.json"
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
    return ev.run(args, transports, load_dataset(ROOT / "examples/public_health_v3"))


def test_recordings_match_report_fingerprints(committed: dict) -> None:
    for key, path in CASSETTES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == committed["config"][key], path.name


@pytest.mark.parametrize("condition", CONDITIONS)
def test_generation_metrics_reproduce(replayed: dict, committed: dict, condition: str) -> None:
    assert replayed["generation"][condition] == committed["generation"][condition]
    assert (replayed["generation_with_feedback_retry"][condition]
            == committed["generation_with_feedback_retry"][condition])


def test_agent_traces_and_retrieval_effect_reproduce(replayed: dict, committed: dict) -> None:
    assert replayed["agent"]["bm25_agent"] == committed["agent"]["bm25_agent"]
    assert replayed["rewrite"]["effect_on_retrieval"] == committed["rewrite"]["effect_on_retrieval"]


def test_no_false_answers_in_any_condition(committed: dict) -> None:
    """文档里引用的结论：不可回答问题上的误答为 0。"""
    for key in ("generation", "generation_with_feedback_retry"):
        for condition, trials in committed[key].items():
            for metrics in trials.values():
                assert metrics["false_answer_rate"]["k"] == 0, (key, condition)
