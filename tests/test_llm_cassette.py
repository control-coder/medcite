"""调用记录：只有未命中时才计费，回放零网络，预算与边界在发包前强制。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from medidiag.errors import MediDiagError
from medidiag.llm.cassette import Cassette, RecordingTransport, ReplayTransport, request_key

URL = "https://api.xiaomimimo.com/v1/chat/completions"
HEADERS = {"Authorization": "Bearer sk-secret-value", "Idempotency-Key": "k"}


def body(text: str = "你好", **extra: Any) -> dict[str, Any]:
    return {"model": "mimo-v2.5", "messages": [{"role": "user", "content": text}], "max_tokens": 64, **extra}


class FakeServer:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float,
                 follow_redirects: bool) -> httpx.Response:
        with self.lock:
            self.calls += 1
            number = self.calls
        payload = {"id": f"r{number}", "model": "mimo-v2.5", "system_fingerprint": "x",
                   "choices": [{"message": {"content": f"答{number}", "reasoning_content": "不应被保存"},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "extra": 1}}
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))


def make(tmp_path: Path, **kw: Any) -> tuple[RecordingTransport, FakeServer]:
    server = FakeServer()
    kw.setdefault("max_calls", 5)
    return RecordingTransport(tmp_path / "c.jsonl", tmp_path / "ledger.db", post=server, **kw), server


def test_record_then_replay_is_identical_and_offline(tmp_path: Path):
    recorder, server = make(tmp_path)
    recorder(URL, headers=HEADERS, json=body(), timeout=30)
    again = recorder(URL, headers=HEADERS, json=body(), timeout=30).json()  # 命中，不再请求
    assert server.calls == 1
    replay = ReplayTransport(tmp_path / "c.jsonl")
    assert replay(URL, headers={}, json=body(), timeout=30).json() == again
    assert again["choices"][0]["message"]["content"] == "答1" and again["usage"]["total_tokens"] == 12
    assert recorder.spent() == {"calls": 1, "http_200": 1, "prompt_tokens": 10, "completion_tokens": 2}


def test_cassette_never_stores_secrets_reasoning_or_extra_fields(tmp_path: Path):
    recorder, _ = make(tmp_path)
    recorder(URL, headers=HEADERS, json=body(), timeout=30)
    text = (tmp_path / "c.jsonl").read_text(encoding="utf-8")
    assert "sk-secret-value" not in text and "不应被保存" not in text and "system_fingerprint" not in text
    assert '"extra"' not in text
    assert json.loads(text)["body"]["choices"][0]["message"]["content"] == "答1"


def test_replay_miss_fails_closed(tmp_path: Path):
    make(tmp_path)[0](URL, headers=HEADERS, json=body("甲"), timeout=30)
    with pytest.raises(MediDiagError) as excinfo:
        ReplayTransport(tmp_path / "c.jsonl")(URL, headers={}, json=body("乙"), timeout=30)
    assert excinfo.value.code == "PROVIDER_REQUEST_REJECTED"


def test_trials_are_independent_samples_of_the_same_request(tmp_path: Path):
    cassette = Cassette(tmp_path / "c.jsonl")
    server = FakeServer()
    answers = []
    for trial in range(3):
        transport = RecordingTransport(cassette, tmp_path / "l.db", max_calls=5, trial=trial, post=server)
        answers.append(transport(URL, headers=HEADERS, json=body(), timeout=30).json()["choices"][0]["message"]["content"])
    assert answers == ["答1", "答2", "答3"] and len(cassette) == 3
    assert request_key(body(), 0) != request_key(body(), 1)
    for trial in range(3):  # 回放时每个 trial 取回自己的样本
        out = ReplayTransport(cassette, trial=trial)(URL, headers={}, json=body(), timeout=30)
        assert out.json()["choices"][0]["message"]["content"] == answers[trial]


def test_budget_is_hard_cap_and_persistent(tmp_path: Path):
    recorder, server = make(tmp_path, max_calls=2)
    recorder(URL, headers=HEADERS, json=body("1"), timeout=30)
    recorder(URL, headers=HEADERS, json=body("2"), timeout=30)
    with pytest.raises(MediDiagError, match="额度"):
        recorder(URL, headers=HEADERS, json=body("3"), timeout=30)
    # 重启后额度不重置，也不能改上限
    again = RecordingTransport(tmp_path / "c.jsonl", tmp_path / "ledger.db", max_calls=2, post=server)
    with pytest.raises(MediDiagError, match="额度"):
        again(URL, headers=HEADERS, json=body("4"), timeout=30)
    with pytest.raises(ValueError):
        RecordingTransport(tmp_path / "c.jsonl", tmp_path / "ledger.db", max_calls=9, post=server)
    assert server.calls == 2
    # 已记录过的请求仍可免费读取
    assert again(URL, headers=HEADERS, json=body("1"), timeout=30).status_code == 200


def test_parallel_recording_never_exceeds_budget(tmp_path: Path):
    recorder, server = make(tmp_path, max_calls=10)
    outcomes: list[str] = []

    def work(i: int) -> None:
        try:
            recorder(URL, headers=HEADERS, json=body(f"q{i}"), timeout=30)
            outcomes.append("ok")
        except MediDiagError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert server.calls == 10 and outcomes.count("ok") == 10 and outcomes.count("rejected") == 30
    assert len(Cassette(tmp_path / "c.jsonl")) == 10


@pytest.mark.parametrize("url,payload,timeout", [
    ("https://example.com/v1/chat/completions", body(), 30),
    ("http://api.xiaomimimo.com/v1/chat/completions", body(), 30),
    (URL + "?x=1", body(), 30),
    (URL, body(model="other"), 30),
    (URL, {**body(), "max_tokens": 4096}, 30),
    (URL, body("长" * 13000), 30),
    (URL, body(), 120),
])
def test_requests_outside_boundary_are_rejected_before_any_call(tmp_path: Path, url, payload, timeout):
    recorder, server = make(tmp_path)
    if payload.get("model") == "other":
        payload = {**payload, "model": "other"}
    with pytest.raises(MediDiagError):
        recorder(url, headers=HEADERS, json=payload, timeout=timeout)
    assert server.calls == 0 and recorder.spent()["calls"] == 0


def test_tool_calls_are_kept_so_agent_runs_can_be_replayed(tmp_path: Path):
    payload = {"id": "r1", "model": "mimo-v2.6-flash", "usage": {"total_tokens": 5},
               "choices": [{"finish_reason": "tool_calls", "message": {
                   "content": None, "reasoning_content": "不应被保存", "tool_calls": [
                       {"id": "call_1", "type": "function", "index": 0,
                        "function": {"name": "search_kb", "arguments": "{\"query\": \"通风\"}"}}]}}]}

    def post(url: str, **kw: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    recorder = RecordingTransport(tmp_path / "c.jsonl", tmp_path / "ledger.db", post=post, max_calls=2)
    recorder(URL, headers=HEADERS, json=body(model="mimo-v2.6-flash"), timeout=30)
    replayed = ReplayTransport(tmp_path / "c.jsonl")(URL, headers={}, json=body(model="mimo-v2.6-flash"), timeout=30)
    message = replayed.json()["choices"][0]["message"]
    assert message["tool_calls"] == [{"id": "call_1", "type": "function",
                                      "function": {"name": "search_kb", "arguments": "{\"query\": \"通风\"}"}}]
    assert "reasoning_content" not in message and "index" not in json.dumps(message)


def test_both_mimo_model_names_are_accepted(tmp_path: Path):
    """旧模型名用于回放旧调用记录，新模型名用于真实调用；其他模型名仍被拒绝。"""
    recorder, server = make(tmp_path)
    for model in ("mimo-v2.5", "mimo-v2.6-flash"):
        recorder(URL, headers=HEADERS, json=body(model=model), timeout=30)
    assert server.calls == 2


def test_failed_responses_cost_budget_but_are_not_cached(tmp_path: Path):
    def rate_limited(url: str, **_: Any) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"}, request=httpx.Request("POST", url))

    recorder = RecordingTransport(tmp_path / "c.jsonl", tmp_path / "l.db", max_calls=3, post=rate_limited)
    assert recorder(URL, headers=HEADERS, json=body(), timeout=30).status_code == 429
    assert len(Cassette(tmp_path / "c.jsonl")) == 0 and recorder.spent()["calls"] == 1


def test_works_through_the_real_provider_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """经 OpenAICompatibleProvider 走完整适配路径：回放结果与记录时的归一化结果一致。"""
    from medidiag.llm.contracts import LLMRequest
    from medidiag.llm.openai_compatible import OpenAICompatibleProvider
    from medidiag.llm.profiles import get_provider_profile

    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    profile = get_provider_profile("mimo_v25")
    recorder, _ = make(tmp_path)

    def run(post: Any) -> Any:
        provider = OpenAICompatibleProvider(profile, post=post, max_retries=0, max_structured_retries=0)
        request = LLMRequest(messages=[{"role": "user", "content": "测试"}], model="mimo-v2.5", max_tokens=64,
                             response_format=None, reasoning_mode="disabled")
        return provider.generate(request, timeout_s=30, idempotency_key="k")

    live, replayed = run(recorder), run(ReplayTransport(tmp_path / "c.jsonl"))
    assert (live.content, live.usage, live.finish_reason) == (replayed.content, replayed.usage, replayed.finish_reason)
    assert live.reasoning_content == "不应被保存" and replayed.reasoning_content is None
