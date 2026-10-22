"""多轮追问：问题补全、历史截断与回退、追问的归属校验，以及单轮问题行为不变。全部离线，用脚本化的假模型。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import select

from medidiag.api.app import create_app
from medidiag.db.models import Case, StageArtifact
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.llm.contracts import LLMRequest, ProviderResult
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.workflow.application import build_application_provider, load_application_config
from medidiag.workflow.context_rewrite import (
    MAX_HISTORY_TURNS,
    MAX_STANDALONE_CHARS,
    MAX_TURN_ANSWER_CHARS,
    PROMPT_VERSION,
    Turn,
    build_context_rewrite_request,
    contextualize,
    parse_standalone,
)
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.retrieval_agent import RetrievalAgentWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker

ROOT = Path(__file__).resolve().parents[1]
HISTORY: list[Turn] = [{"question": "高血压要注意什么？", "outcome": "answered", "answer": ["高血压患者应减少盐的摄入。"]}]


def _result(payload: dict | None, *, model: str = ACTIVE_MIMO_MODEL) -> ProviderResult:
    return ProviderResult(
        provider_id="fake", profile_id="fake", model=model, response_id="rid", system_fingerprint=None,
        content=json.dumps(payload, ensure_ascii=False) if payload is not None else "", parsed_json=payload,
        tool_calls=[], reasoning_content=None, usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        finish_reason="stop", raw_error_code=None, latency_ms=1, retry_count=0, provenance_mode="provider_response_id")


class ScriptedLLM:
    """按顺序返回预设的回复；预设项是异常时抛出。"""

    provider_id = "fake"
    profile_id = "fake"

    def __init__(self, *script: ProviderResult | Exception) -> None:
        self.script = list(script)
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest, *, timeout_s: float, idempotency_key: str) -> ProviderResult:
        self.requests.append(request)
        item = self.script[len(self.requests) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    def capabilities(self):  # pragma: no cover - 协议要求，测试不使用
        raise NotImplementedError


# ----------------------------------------------------------------- 补全本身

def test_parse_standalone_accepts_one_line_and_rejects_bad_shapes() -> None:
    assert parse_standalone({"standalone_question": " 高血压需要一直吃药吗？ "}, "那它需要一直吃药吗？", HISTORY) == (
        "高血压需要一直吃药吗？", "ok")
    for bad in (None, {}, {"standalone_question": 3}, {"standalone_question": "  "},
                {"standalone_question": "两行\n问题"}, {"standalone_question": "长" * (MAX_STANDALONE_CHARS + 1)}):
        assert parse_standalone(bad, "追问", HISTORY) == (None, "CONTEXT_REWRITE_SHAPE")


def test_numbers_not_in_history_or_question_are_treated_as_added_facts() -> None:
    assert parse_standalone({"standalone_question": "高血压每天吃 5 克盐可以吗？"}, "那它吃盐呢？", HISTORY) == (
        None, "CONTEXT_REWRITE_NEW_NUMBER")
    history: list[Turn] = [{"question": "每天 5 克盐够吗？", "outcome": "abstained", "answer": []}]
    assert parse_standalone({"standalone_question": "每天 5 克盐对高血压够吗？"}, "高血压呢？", history)[1] == "ok"


def test_request_keeps_only_recent_turns_and_truncates_answers() -> None:
    history: list[Turn] = [{"question": f"第{i}轮问题", "outcome": "answered", "answer": ["长" * 500]} for i in range(5)]
    history.append({"question": "被拒答的一轮", "outcome": "abstained", "answer": []})
    request = build_context_rewrite_request("那它呢？", history)
    body = json.loads(request.messages[1]["content"])
    assert [t["question"] for t in body["history"]] == ["第3轮问题", "第4轮问题", "被拒答的一轮"]
    assert len(body["history"]) == MAX_HISTORY_TURNS and body["question"] == "那它呢？"
    assert len(body["history"][0]["answer"]) == MAX_TURN_ANSWER_CHARS + 2  # 截断后加省略号
    assert body["history"][2]["answer"] == "（资料不足，系统未回答）"
    assert request.prompt_version == PROMPT_VERSION and request.response_format == {"type": "json_object"}
    assert request.reasoning_mode == "disabled"


def test_contextualize_without_history_makes_no_call() -> None:
    llm = ScriptedLLM()
    assert contextualize(llm, "单轮问题", []) == ("单轮问题", {"status": "skipped", "reason": "no_history"})  # type: ignore[arg-type]
    assert llm.requests == []


@pytest.mark.parametrize("reply,code", [
    (MediDiagError("LLM_TIMEOUT"), "LLM_TIMEOUT"),
    (_result({"standalone_question": ""}), "CONTEXT_REWRITE_SHAPE"),
    (_result({"query": "错的键"}), "CONTEXT_REWRITE_SHAPE"),
    (_result({"standalone_question": "高血压需要吃药吗？"}, model="other-model"), "CONTEXT_REWRITE_MODEL_MISMATCH"),
])
def test_failures_fall_back_to_the_raw_question(reply, code) -> None:
    llm = ScriptedLLM(reply)
    question, info = contextualize(llm, "那它需要一直吃药吗？", HISTORY)  # type: ignore[arg-type]
    assert question == "那它需要一直吃药吗？" and info["status"] == "fallback" and info["code"] == code
    assert len(llm.requests) == 1  # 失败不重试


# ----------------------------------------------------------------- provider

class _Rag:
    chunks: list = []

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> dict:
        self.queries.append(query)
        return {"query": query, "chunks": [], "top_k": 0, "config_hash": "h", "corpus_hash": "c"}


def test_provider_retrieves_with_the_standalone_question() -> None:
    llm = ScriptedLLM(_result({"standalone_question": "高血压需要一直吃药吗？"}))
    rag = _Rag()
    provider = MimoGroundedWorkflowProvider(rag, llm=llm, context_rewrite=True)  # type: ignore[arg-type]
    result = provider.retrieve_in_context("那它需要一直吃药吗？", HISTORY)
    assert rag.queries == ["高血压需要一直吃药吗？"]
    info = result["context_rewrite"]
    assert info["status"] == "ok" and info["standalone_question"] == "高血压需要一直吃药吗？"
    assert info["original_question"] == "那它需要一直吃药吗？" and info["history_turns"] == 1


def test_provider_fallback_and_disabled_paths() -> None:
    rag = _Rag()
    failing = MimoGroundedWorkflowProvider(rag, llm=ScriptedLLM(MediDiagError("LLM_TIMEOUT")),  # type: ignore[arg-type]
                                           context_rewrite=True)
    assert failing.retrieve_in_context("那它呢？", HISTORY)["context_rewrite"]["status"] == "fallback"
    llm = ScriptedLLM()
    disabled = MimoGroundedWorkflowProvider(rag, llm=llm)  # type: ignore[arg-type]
    assert disabled.retrieve_in_context("那它呢？", HISTORY)["context_rewrite"] == {"status": "disabled", "history_turns": 1}
    assert rag.queries == ["那它呢？", "那它呢？"] and llm.requests == []


def test_agent_searches_and_reranks_by_the_standalone_question() -> None:
    config = load_application_config(ROOT / "configs/application.yaml")
    from medidiag.rag.runtime import RuntimeMedicalRAG

    rag = RuntimeMedicalRAG.from_config(config, root=ROOT)
    end = ProviderResult(
        provider_id="fake", profile_id="fake", model=ACTIVE_MIMO_MODEL, response_id="rid", system_fingerprint=None,
        content="结束", parsed_json=None, tool_calls=[], reasoning_content=None, usage={}, finish_reason="stop",
        raw_error_code=None, latency_ms=1, retry_count=0, provenance_mode="provider_response_id")
    llm = ScriptedLLM(_result({"standalone_question": "室内通风有什么好处？"}), end)
    provider = RetrievalAgentWorkflowProvider(rag, llm=llm, context_rewrite=True)  # type: ignore[arg-type]
    result = provider.retrieve_in_context("那它有什么好处？", [
        {"question": "屋里怎么通风？", "outcome": "abstained", "answer": []}])
    assert result["search_agent"]["steps"][0]["query"] == "室内通风有什么好处？"
    agent_user = json.loads(llm.requests[1].messages[1]["content"])
    assert agent_user["question"] == "室内通风有什么好处？"


# ----------------------------------------------------------------- worker

class _Recording(DeterministicWorkflowProvider):
    """记录 worker 调用了哪个检索入口、生成时用的是哪个问题。"""

    def __init__(self, standalone: str | None = "补全后的问题") -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []
        self.standalone = standalone

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        self.calls.append(("retrieve", normalized_query))
        return super().retrieve(normalized_query)

    def retrieve_in_context(self, normalized_query: str, history: list[Turn]) -> dict[str, Any]:
        self.calls.append(("retrieve_in_context", (normalized_query, history)))
        info = ({"status": "ok", "standalone_question": self.standalone} if self.standalone
                else {"status": "fallback", "code": "LLM_TIMEOUT"})
        return {**super().retrieve(self.standalone or normalized_query), "context_rewrite": info}

    def generate(self, question: str, retrieval: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("generate", question))
        return super().generate(question, retrieval, plan)


@pytest.fixture
def factory(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'multiturn.db').as_posix()}")
    init_db(engine)
    yield get_session_factory(engine)
    engine.dispose()


def _run_case(factory, provider, question: str, *, parent: str | None = None, owner: str = "alice") -> str:
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(session, question, f"k-{question}-{parent}", owner, owner_id=owner,
                                    parent_case_id=parent)
        task = executor.start_workflow(session, case.case_id, "case_workflow", f"w-{case.case_id}", "h")
        case_id, task_id = case.case_id, task.task_id
    result = SingleMachineWorker(factory, provider).run_task(task_id)
    assert result.final_state == "CLOSED_SUCCESS"
    return case_id


def test_single_turn_case_uses_the_original_retrieve_and_question(factory) -> None:
    provider = _Recording()
    _run_case(factory, provider, "高血压要注意什么？")
    assert [name for name, _ in provider.calls] == ["retrieve", "generate"]
    assert provider.calls[1] == ("generate", "高血压要注意什么？")
    with factory() as session:
        retrieval = session.scalar(select(StageArtifact).where(StageArtifact.stage == "retrieval"))
        assert retrieval is not None and "context_rewrite" not in retrieval.payload


def test_follow_up_passes_history_and_generates_with_the_standalone_question(factory) -> None:
    provider = _Recording()
    first = _run_case(factory, provider, "高血压要注意什么？")
    provider.calls.clear()
    _run_case(factory, provider, "那它需要一直吃药吗？", parent=first)
    (name, (query, history)), generate = provider.calls
    assert name == "retrieve_in_context" and query == "那它需要一直吃药吗？"
    assert history == [{"question": "高血压要注意什么？", "outcome": "answered",
                        "answer": ["Available evidence is insufficient for a definitive diagnosis."]}]
    assert generate == ("generate", "补全后的问题")


def test_fallback_generates_with_the_raw_follow_up(factory) -> None:
    provider = _Recording(standalone=None)
    first = _run_case(factory, provider, "高血压要注意什么？")
    _run_case(factory, provider, "那它需要一直吃药吗？", parent=first)
    assert provider.calls[-1] == ("generate", "那它需要一直吃药吗？")


def test_history_is_capped_to_recent_turns_in_order(factory) -> None:
    provider = _Recording()
    parent = None
    for i in range(5):
        parent = _run_case(factory, provider, f"第{i}轮问题", parent=parent)
    _, (_, history) = next(call for call in reversed(provider.calls) if call[0] == "retrieve_in_context")
    assert [t["question"] for t in history] == ["第1轮问题", "第2轮问题", "第3轮问题"]  # 第 4 轮自己的历史


def test_worker_ignores_a_parent_owned_by_someone_else(factory) -> None:
    provider = _Recording()
    other = _run_case(factory, provider, "别人的问题", owner="bob")
    provider.calls.clear()
    _run_case(factory, provider, "那它呢？", parent=other)  # 绕过 API 直接写库，worker 仍不读取别人的历史
    assert [name for name, _ in provider.calls] == ["retrieve", "generate"]


def test_unsupported_provider_runs_the_follow_up_as_a_single_turn(factory) -> None:
    class Plain(DeterministicWorkflowProvider):
        seen: list[str] = []

        def retrieve(self, normalized_query: str) -> dict[str, Any]:
            self.seen.append(normalized_query)
            return super().retrieve(normalized_query)

    first = _run_case(factory, Plain(), "高血压要注意什么？")
    _run_case(factory, Plain(), "那它需要一直吃药吗？", parent=first)
    assert Plain.seen[-1] == "那它需要一直吃药吗？"


# ----------------------------------------------------------------- API

@pytest.fixture
def clients(factory):
    app = create_app(session_factory=factory)
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.get("/api/v1/session")
        bob.get("/api/v1/session")
        yield alice, bob


def _post(client, key: str, **body):
    return client.post("/api/v1/cases", headers={"Idempotency-Key": key},
                       json={"input_kind": "deidentified_simulation", **body})


def _close(factory, case_id: str, status: str = "CLOSED_SUCCESS") -> None:
    with factory() as session:
        case = session.scalar(select(Case).where(Case.case_id == case_id))
        assert case is not None
        case.status = status
        session.commit()


def test_api_follow_up_requires_an_owned_closed_parent(clients, factory) -> None:
    alice, bob = clients
    first = _post(alice, "a1", question="高血压要注意什么？请说明。").json()["case_id"]
    assert _post(alice, "a2", question="那要吃药吗？", parent_case_id=first).status_code == 400  # 上一轮未结束
    _close(factory, first)
    response = _post(alice, "a3", question="那要吃药吗？", parent_case_id=first)
    assert response.status_code == 201 and response.json()["parent_case_id"] == first
    assert _post(bob, "b1", question="那要吃药吗？", parent_case_id=first).status_code == 404  # 越权与不存在一样
    assert _post(alice, "a4", question="那要吃药吗？", parent_case_id="case_missing").status_code == 404


def test_api_single_turn_is_backward_compatible(clients) -> None:
    alice, _ = clients
    response = _post(alice, "s1", question="高血压要注意什么？请说明。")
    assert response.status_code == 201 and response.json()["parent_case_id"] is None
    assert _post(alice, "s2", question="要吃药吗？").status_code == 422  # 单轮问题仍要求至少 10 个字


def test_api_error_status_for_unfinished_parent(clients) -> None:
    alice, _ = clients
    first = _post(alice, "u1", question="高血压要注意什么？请说明。").json()["case_id"]
    body = _post(alice, "u2", question="那要吃药吗？", parent_case_id=first).json()
    assert body["code"] == "CASE_INVALID_INPUT"


# ----------------------------------------------------------------- 配置

def test_multiturn_config_only_adds_context_rewrite(tmp_path) -> None:
    base = load_application_config(ROOT / "configs/application_dense_agent_rerank.yaml")
    multi = load_application_config(ROOT / "configs/application_dense_agent_rerank_multiturn.yaml")
    assert multi["generation"].pop("context_rewrite") is True and multi == base
    bad = copy.deepcopy(base)
    bad["generation"]["context_rewrite"] = "yes"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_application_config(path)


def test_shipped_multiturn_config_builds_an_agent_with_context_rewrite(monkeypatch, tmp_path) -> None:
    import numpy as np

    from medidiag.rag.dense import DenseRetriever
    from medidiag.rag.rerank import Reranker
    from medidiag.workflow import application

    def fake_dense(chunks, config):
        return DenseRetriever(chunks, encoder=lambda texts: np.array([[float(len(t)), 1.0] for t in texts]),
                              model_name=config["embedding"]["model"], model_revision=config["embedding"]["revision"])

    def fake_rerank(config):
        rerank = config["retrieval"]["rerank"]
        return Reranker(lambda pairs: [0.0] * len(pairs), model_name=rerank["model"], model_revision=rerank["revision"],
                        candidates=rerank["candidates"])

    monkeypatch.setattr(application, "_dense_factory", fake_dense)
    monkeypatch.setattr(application, "_rerank_factory", fake_rerank)
    monkeypatch.setenv("MIMO_API_KEY", "test-not-a-secret")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    provider = build_application_provider("mimo_grounded", app_config="configs/application_dense_agent_rerank_multiturn.yaml",
                                          root=ROOT, live_budget=tmp_path / "ledger.db")
    assert isinstance(provider, RetrievalAgentWorkflowProvider) and provider.context_rewrite is True
    single = build_application_provider("mimo_grounded", app_config="configs/application_dense_agent_rerank.yaml",
                                        root=ROOT, live_budget=tmp_path / "ledger2.db")
    assert isinstance(single, RetrievalAgentWorkflowProvider) and single.context_rewrite is False
