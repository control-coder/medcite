"""真实应用的离线注入回归；测试不联网，不向传输发送真实凭据。"""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from medidiag.api.app import create_app
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.llm.budget import BudgetedTransport
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.llm.openai_compatible import OpenAICompatibleProvider
from medidiag.llm.profiles import get_provider_profile
from medidiag.workflow.application import build_application_provider
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.xiaomimimo.com/v1/chat/completions"


def response(answer, **changes):
    return httpx.Response(200, json={"id": "injected-response-id", "model": ACTIVE_MIMO_MODEL,
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(answer, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, **changes})


def make_provider(tmp_path, monkeypatch, post):
    monkeypatch.setenv("MIMO_API_KEY", "test-placeholder-not-a-real-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", ACTIVE_MIMO_MODEL)
    profile = get_provider_profile("mimo_v25")
    transport = BudgetedTransport(tmp_path / "calls.db", post=post)
    llm = OpenAICompatibleProvider(profile, post=transport, max_retries=0, max_structured_retries=0)
    rag = build_application_provider("retrieval_mock", root=ROOT).rag_stage
    return MimoGroundedWorkflowProvider(rag, llm=llm), transport


def exact_answer(body):
    evidence = json.loads(body["messages"][1]["content"])["evidence"]
    cid, text = next(iter(evidence.items()))
    return {"status": "sufficient", "claims": [{"text": text, "citation_chunk_ids": [cid]}]}


@pytest.mark.parametrize("query,mode,expected,calls", [
    ("通风如何改善室内空气质量？", "exact", "ready", 1),
    ("疫苗抗体滴度的具体阈值", "abstain", "insufficient_evidence", 1),
    ("模拟提问：量子纠缠计算芯片", "exact", "insufficient_evidence", 0),
    ("通风如何改善室内空气质量？", "timeout", "failed", 1),
    ("通风如何改善室内空气质量？", "429", "failed", 1),
    ("通风如何改善室内空气质量？", "malformed", "failed", 1),
    ("通风如何改善室内空气质量？", "invented", "failed", 1),
])
def test_worker_and_api_without_live_calls(tmp_path, monkeypatch, query, mode, expected, calls):
    def post(url, **kwargs):
        assert kwargs["follow_redirects"] is False
        if mode == "timeout":
            raise httpx.ReadTimeout("注入超时")
        if mode == "429":
            return httpx.Response(429, json={"error": "注入限流"})
        if mode == "malformed":
            return response({"extra": "注入错误格式"})
        if mode == "abstain":
            return response({"status": "insufficient", "claims": []})
        answer = exact_answer(kwargs["json"])
        if mode == "invented":
            answer["claims"][0]["text"] = "来源中不存在的断言"
        return response(answer)
    provider, ledger = make_provider(tmp_path, monkeypatch, post)
    engine = create_db_engine("sqlite:///" + (tmp_path / "app.db").as_posix())
    init_db(engine)
    factory = get_session_factory(engine)
    try:
        with TestClient(create_app(session_factory=factory)) as client:
            created = client.post("/api/v1/consultations", headers={"Idempotency-Key": "create"}, json={
                "symptoms": query, "duration": "模拟", "input_kind": "deidentified_simulation",
                "non_sensitive_confirmed": True})
            assert created.status_code == 201
            prefix = "/api/v1/cases/" + created.json()["case_id"]
            assert client.post(prefix + "/workflow", headers={"Idempotency-Key": "run"}).status_code == 200
            worker = SingleMachineWorker(factory, provider)
            assert worker.call_runner.max_attempts == 1
            worker.run_once()
            analysis = client.get(prefix + "/analysis").json()
            assert analysis["execution_mode"] == "mimo_grounded"
            assert analysis["outcome"] == expected, analysis
            assert len(ledger.records()) == calls
            if expected == "ready":
                assert analysis["claims"]
                assert analysis["observation"]["recorded_input_tokens"] == 10
                assert analysis["observation"]["cost_usd"] is None
                assert "模拟生成" not in str(analysis["limitations"])
            else:
                assert not analysis["claims"]
    finally:
        engine.dispose()


def test_budget_persists_and_rejects_unsafe_requests(tmp_path):
    seen = []
    def post(*args, **kwargs):
        seen.append(kwargs)
        raise httpx.ConnectError("注入网络失败")
    ledger = BudgetedTransport(tmp_path / "limit.db", max_calls=1, post=post)
    args = {"headers": {}, "json": {"model": "mimo-v2.5", "max_tokens": 128, "messages": []}, "timeout": 45}
    with pytest.raises(MediDiagError):
        ledger("https://other.example/v1/chat/completions", **args)
    assert not ledger.records()
    with pytest.raises(httpx.ConnectError):
        ledger(URL, **args)
    with pytest.raises(MediDiagError):
        BudgetedTransport(ledger.path, max_calls=1, post=post)(URL, **args)
    with pytest.raises(ValueError):
        BudgetedTransport(ledger.path, max_calls=8, post=post)
    assert len(seen) == 1
    assert "注入网络失败" not in str(ledger.records())


def test_live_factory_requires_explicit_ledger():
    with pytest.raises(ValueError, match="live-budget"):
        build_application_provider("mimo_grounded", root=ROOT)


@pytest.mark.parametrize("kind", ["unknown", "rewrite", "status", "model", "length", "json", "content"])
def test_reject_unbound_truncated_and_bad_json(tmp_path, monkeypatch, kind):
    def post(url, **kwargs):
        answer = exact_answer(kwargs["json"])
        changes = {}
        if kind == "unknown":
            answer["claims"][0]["citation_chunk_ids"] = ["unknown"]
        elif kind == "rewrite":
            answer["claims"][0]["text"] = "任意改写"
        elif kind == "status":
            answer["status"] = "insufficient"
        elif kind == "model":
            changes["model"] = "wrong-model"
        elif kind == "length":
            changes["choices"] = [{"finish_reason": "length", "message": {"content": json.dumps(answer)}}]
        elif kind == "content":
            changes["choices"] = [{"finish_reason": "stop", "message": {"content": "非JSON输出"}}]
        elif kind == "json":
            return httpx.Response(200, text="not json")
        return response(answer, **changes)
    provider, ledger = make_provider(tmp_path, monkeypatch, post)
    with pytest.raises(MediDiagError):
        provider.generate("通风如何改善室内空气质量？", provider.retrieve("通风空气"), {})
    assert len(ledger.records()) == 1


def test_budget_atomic_reservation_and_redirect_rejection(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    ledger = BudgetedTransport(tmp_path / "concurrent.db", max_calls=1,
        post=lambda *args, **kwargs: httpx.Response(302, headers={"Location": "https://other.example"}))
    def call():
        try:
            ledger(URL, headers={}, json={"model": "mimo-v2.5", "max_tokens": 16, "messages": []}, timeout=45)
        except MediDiagError:
            return "rejected"
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: call(), range(4))) == ["rejected"] * 4
    assert len(ledger.records()) == 1
    assert ledger.records()[0]["http_status"] == 302


def test_review_checks_full_excerpt_again(tmp_path, monkeypatch):
    provider, _ = make_provider(tmp_path, monkeypatch, lambda url, **kwargs: response(exact_answer(kwargs["json"])))
    evidence = provider.retrieve("通风空气")
    generation = provider.generate("通风空气", evidence, {}).payload
    review = provider.review(generation, {}, evidence)
    assert review["verdict"] == "APPROVED"
    assert all(c["verdict"] == "PARTIAL" and c["confidence"] is None for c in review["citation_verdicts"])
    generation["claims"][0]["text"] = "只改写了一个字也必须拒绝"
    assert provider.review(generation, {}, evidence)["verdict"] == "ESCALATED"
