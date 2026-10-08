"""LLM 评测编排：用假服务器录制，再零网络回放，结果逐字一致；指标口径可手算核对。"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from eval import llm_pipeline_eval as ev
from eval.retrieval_benchmark import Dataset
from medidiag.errors import MediDiagError
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport

CHUNKS = [
    {"chunk_id": "c1", "source_id": "d1", "text": "流感是一种急性呼吸道传染病，由流感病毒引起。"},
    {"chunk_id": "c2", "source_id": "d2", "text": "勤洗手可以减少病原体传播，保持良好卫生习惯。"},
    {"chunk_id": "c3", "source_id": "d3", "text": "高温天气应及时补充水分，避免长时间户外活动导致中暑。"},
]


def query(sample_id: str, split: str, bucket: str, question: str, gold: list[str]) -> dict[str, Any]:
    return {"sample_id": sample_id, "split": split, "bucket": bucket, "question": question, "gold_evidence_ids": gold}


QUERIES = [
    query("t1", "test", "direct", "流感病毒引起什么疾病", ["c1"]),
    query("t2", "test", "paraphrase", "天气很热怎样避免中暑", ["c3"]),
    query("t3", "test", "out_of_scope", "量子计算机的工作原理", []),
    query("t4", "test", "near_miss", "流感疫苗的价格是多少", []),
    query("d1", "dev", "direct", "洗手有什么作用", ["c2"]),
]


class FakeModel:
    """改写：原问题 + “流感”；生成：只要证据里有 c1 且问题提到流感就选 c1，否则弃答。"""

    def __init__(self) -> None:
        self.model_calls = 0
        self.lock = threading.Lock()

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float,
                 follow_redirects: bool) -> httpx.Response:
        with self.lock:
            self.model_calls += 1
        system, user = (m["content"] for m in json["messages"])
        data = __import__("json").loads(user)
        if system.startswith("你是公共卫生科普检索的查询改写助手"):
            content = __import__("json").dumps({"query": data["question"] + "中暑"}, ensure_ascii=False)
        else:
            evidence = data["evidence"]
            pick = [cid for cid in evidence if cid == "c1" and "流感" in data["question"]]
            content = __import__("json").dumps({
                "status": "sufficient" if pick else "insufficient",
                "claims": [{"text": evidence[c], "citation_chunk_ids": [c]} for c in pick]}, ensure_ascii=False)
        body = {"id": "r", "model": "mimo-v2.5", "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))


def args(**kw: Any) -> argparse.Namespace:
    base = dict(conditions=["bm25", "bm25_rewrite"], repeat_trials=2, workers=4, limit=0, embedding_revision=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")


def test_record_then_replay_reproduces_the_report_without_network(tmp_path: Path):
    dataset = Dataset(chunks=CHUNKS, queries=QUERIES)
    cassette = Cassette(tmp_path / "c.jsonl")
    server = FakeModel()
    recording = {t: RecordingTransport(cassette, tmp_path / "l.db", max_calls=100, trial=t, post=server) for t in range(2)}
    recorded = ev.run(args(), recording, dataset)
    calls_after_record = server.model_calls
    assert calls_after_record > 0

    replayed = ev.run(args(), {t: ReplayTransport(Cassette(tmp_path / "c.jsonl"), trial=t) for t in range(2)}, dataset)
    for report in (recorded, replayed):
        report.pop("wall_seconds")
    assert recorded == replayed
    assert server.model_calls == calls_after_record  # 回放没有再请求


def test_metrics_match_hand_computation(tmp_path: Path):
    dataset = Dataset(chunks=CHUNKS, queries=QUERIES)
    transports = {t: RecordingTransport(tmp_path / "c.jsonl", tmp_path / "l.db", max_calls=100, trial=t, post=FakeModel())
                  for t in range(2)}
    report = ev.run(args(), transports, dataset)
    metrics = report["generation"]["bm25"]["trial_0"]
    # 测试集：t1 可答且选中 c1；t2 可答但假模型弃答；t3 空检索弃答；t4 不可答却被作答（提到“流感”）
    assert metrics["useful_answer_rate"]["k"] == 1 and metrics["useful_answer_rate"]["n"] == 2
    assert metrics["over_abstention_rate"]["k"] == 1
    assert metrics["false_answer_rate"]["k"] == 1 and metrics["false_answer_rate"]["n"] == 2
    assert metrics["false_answer_by_bucket"]["near_miss"]["k"] == 1
    assert metrics["false_answer_by_bucket"]["out_of_scope"]["k"] == 0
    assert metrics["precision_when_answered"]["k"] == 1 and metrics["precision_when_answered"]["n"] == 2
    # 空检索（t3）不应调用模型
    t3 = next(r for r in report["per_query"] if r["sample_id"] == "t3")
    assert t3["conditions"]["bm25"]["trials"][0]["model_called"] is False
    # 两个 trial 的假模型确定性相同 => 完全一致
    assert report["repeatability"]["bm25"]["decision_all_agree"]["rate"] == 1.0


def test_rewrite_effect_counts_gains_and_losses(tmp_path: Path):
    dataset = Dataset(chunks=CHUNKS, queries=QUERIES)
    transports = {t: RecordingTransport(tmp_path / "c.jsonl", tmp_path / "l.db", max_calls=100, trial=t, post=FakeModel())
                  for t in range(2)}
    effect = ev.run(args(), transports, dataset)["rewrite"]["effect_on_retrieval"]
    both = effect["all/both"]
    # 改写给每个问题加了“中暑”，使 t2 之外的查询也能命中 c3，但不会让 c1/c2 丢失
    assert both["n"] == 3 and both["lost"] == 0 and both["bm25_hit@3"]["k"] <= both["rewrite_hit@3"]["k"]
    assert effect["paraphrase/test"]["n"] == 1


def test_contract_violation_is_counted_not_hidden(tmp_path: Path):
    class Rewriter(FakeModel):
        def __call__(self, url: str, **kw: Any) -> httpx.Response:
            response = super().__call__(url, **kw)
            data = response.json()
            if "evidence" in kw["json"]["messages"][1]["content"]:  # 生成：改写原文 => 违反契约
                content = json.loads(data["choices"][0]["message"]["content"])
                for claim in content["claims"]:
                    claim["text"] += "（改写）"
                data["choices"][0]["message"]["content"] = json.dumps(content, ensure_ascii=False)
            return httpx.Response(200, json=data, request=httpx.Request("POST", url))

    transports = {0: RecordingTransport(tmp_path / "c.jsonl", tmp_path / "l.db", max_calls=100, post=Rewriter())}
    report = ev.run(args(conditions=["bm25"], repeat_trials=1), transports, Dataset(chunks=CHUNKS, queries=QUERIES))
    metrics = report["generation"]["bm25"]["trial_0"]
    assert metrics["contract_violations_or_errors"] == 2  # t1 与 t4 的改写都被拒绝
    assert metrics["useful_answer_rate"]["k"] == 0 and metrics["false_answer_rate"]["k"] == 0


def test_sign_test_and_retry_helpers():
    assert ev.binomial_two_sided(0, 0) == 1.0
    assert ev.binomial_two_sided(0, 5) == pytest.approx(0.0625)
    assert ev.binomial_two_sided(5, 10) == 1.0
    delays: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise MediDiagError("PROVIDER_RATE_LIMITED")
        return "ok"

    assert ev.call_with_retry(flaky, sleep=delays.append) == "ok" and delays == [2.0, 4.0]
    with pytest.raises(MediDiagError):
        ev.call_with_retry(lambda: (_ for _ in ()).throw(MediDiagError("PROVIDER_AUTH_FAILED")), sleep=delays.append)
