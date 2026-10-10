"""用仓库内的调用记录零网络重放真实模型实验，指标必须与入库报告逐项一致。

改动提示词、检索分词、证据排序或应用的生成路径，都会改变请求哈希而导致调用记录未命中；
这时需要重新调用并记录（付费），而不是悄悄改测试。dense 条件依赖本地嵌入模型，不在此回放。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from eval import llm_pipeline_eval as ev
from eval.retrieval_benchmark import load_dataset
from medidiag.llm.cassette import Cassette, ReplayTransport
from medidiag.llm.models import PREVIOUS_MIMO_MODEL

ROOT = Path(__file__).resolve().parent.parent
CASSETTE = ROOT / "eval/cassettes/llm-pipeline-v2.jsonl"
REPORT = ROOT / "artifacts/reports/llm/llm-pipeline-v2-20261008.json"


@pytest.fixture(scope="module")
def replayed(request: pytest.FixtureRequest) -> dict:
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setenv("MIMO_API_KEY", "replay-no-network")
    mp.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    cassette = Cassette(CASSETTE)
    args = argparse.Namespace(conditions=["bm25", "bm25_rewrite"], repeat_trials=3, workers=4, limit=0,
                              embedding_revision=None, model=PREVIOUS_MIMO_MODEL)
    transports = {t: ReplayTransport(cassette, trial=t) for t in range(3)}
    return ev.run(args, transports, load_dataset(ROOT / "examples/public_health_v2"))


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_cassette_matches_report_fingerprint(committed: dict):
    import hashlib

    assert hashlib.sha256(CASSETTE.read_bytes()).hexdigest() == committed["config"]["cassette_sha256"]


@pytest.mark.parametrize("condition", ["bm25", "bm25_rewrite"])
def test_generation_metrics_reproduce(replayed: dict, committed: dict, condition: str):
    assert replayed["generation"][condition] == committed["generation"][condition]


def test_repeatability_and_rewrite_effect_reproduce(replayed: dict, committed: dict):
    assert replayed["repeatability"] == committed["repeatability"]
    assert replayed["rewrite"]["effect_on_retrieval"] == committed["rewrite"]["effect_on_retrieval"]
    assert replayed["rewrite"]["rewrites"] == committed["rewrite"]["rewrites"]


def test_headline_claims_hold_in_the_committed_run(committed: dict):
    """文档里引用的结论：不可回答问题上的误答为 0，且改写在口语改写桶上显著提高命中。"""
    for condition, trials in committed["generation"].items():
        for metrics in trials.values():
            assert metrics["false_answer_rate"]["k"] == 0, condition
            assert metrics["contract_violations_by_kind"]["unanswerable"] == 0, condition
    paraphrase = committed["rewrite"]["effect_on_retrieval"]["paraphrase/test"]
    assert paraphrase["gained"] > paraphrase["lost"] and paraphrase["sign_test_p"] < 0.01


RETRY_CASSETTE = ROOT / "eval/cassettes/llm-pipeline-v2-retry.jsonl"
RETRY_REPORT = ROOT / "artifacts/reports/llm/llm-pipeline-v2-retry-20261008.json"


def test_retry_experiment_reproduces_from_committed_cassettes(request: pytest.FixtureRequest):
    import hashlib

    committed = json.loads(RETRY_REPORT.read_text(encoding="utf-8"))
    assert hashlib.sha256(RETRY_CASSETTE.read_bytes()).hexdigest() == committed["config"]["retry_cassette_sha256"]
    assert hashlib.sha256(CASSETTE.read_bytes()).hexdigest() == committed["config"]["cassette_sha256"]

    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setenv("MIMO_API_KEY", "replay-no-network")
    mp.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    base, retry = Cassette(CASSETTE), Cassette(RETRY_CASSETTE)
    transports = {t: ReplayTransport(base, trial=t) for t in range(3)}
    transports.update({ev.RETRY_BASE + t: ReplayTransport(retry, trial=ev.RETRY_BASE + t) for t in range(3)})
    args = argparse.Namespace(conditions=["bm25", "bm25_rewrite"], repeat_trials=3, workers=4, limit=0,
                              embedding_revision=None, retry_on_violation=True,
                              model=PREVIOUS_MIMO_MODEL)
    replayed = ev.run(args, transports, load_dataset(ROOT / "examples/public_health_v2"))
    for condition in ("bm25", "bm25_rewrite"):
        assert replayed["generation_with_retry"][condition] == committed["generation_with_retry"][condition]
        assert replayed["retry_summary"][condition] == committed["retry_summary"][condition]
    # 重采样不能引入误答：不论是否重采样，不可回答问题上的误答都是 0
    for trials in committed["generation_with_retry"].values():
        assert all(m["false_answer_rate"]["k"] == 0 for m in trials.values())


FEEDBACK_CASSETTE = ROOT / "eval/cassettes/llm-pipeline-v2-feedback.jsonl"
FEEDBACK_REPORT = ROOT / "artifacts/reports/llm/llm-pipeline-v2-feedback-20261008.json"


def test_feedback_retry_experiment_reproduces_from_committed_cassettes(request: pytest.FixtureRequest):
    import hashlib

    committed = json.loads(FEEDBACK_REPORT.read_text(encoding="utf-8"))
    assert hashlib.sha256(FEEDBACK_CASSETTE.read_bytes()).hexdigest() == committed["config"]["feedback_cassette_sha256"]
    assert hashlib.sha256(RETRY_CASSETTE.read_bytes()).hexdigest() == committed["config"]["retry_cassette_sha256"]

    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setenv("MIMO_API_KEY", "replay-no-network")
    mp.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    base, retry, feedback = Cassette(CASSETTE), Cassette(RETRY_CASSETTE), Cassette(FEEDBACK_CASSETTE)
    transports = {t: ReplayTransport(base, trial=t) for t in range(3)}
    transports.update({ev.RETRY_BASE + t: ReplayTransport(retry, trial=ev.RETRY_BASE + t) for t in range(3)})
    transports.update({ev.FEEDBACK_BASE + t: ReplayTransport(feedback, trial=ev.FEEDBACK_BASE + t) for t in range(3)})
    args = argparse.Namespace(conditions=["bm25", "bm25_rewrite"], repeat_trials=3, workers=4, limit=0,
                              embedding_revision=None, retry_on_violation=True, feedback_retry=True,
                              model=PREVIOUS_MIMO_MODEL)
    replayed = ev.run(args, transports, load_dataset(ROOT / "examples/public_health_v2"))
    for condition in ("bm25", "bm25_rewrite"):
        assert (replayed["generation_with_feedback_retry"][condition]
                == committed["generation_with_feedback_retry"][condition])
        assert replayed["feedback_retry_summary"][condition] == committed["feedback_retry_summary"][condition]
    for trials in committed["generation_with_feedback_retry"].values():
        assert all(m["false_answer_rate"]["k"] == 0 for m in trials.values())  # 重试不能引入误答

