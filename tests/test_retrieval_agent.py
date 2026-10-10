"""检索智能体的离线回归：用脚本化的假模型，不联网。"""
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from medidiag.errors import MediDiagError
from medidiag.llm.contracts import LLMRequest, ProviderResult
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.application import load_application_config
from medidiag.workflow.retrieval_agent import (
    RetrievalAgentWorkflowProvider,
    merge_hits,
    run_search_agent,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str


def _hits(*ids: str) -> list[Hit]:
    return [Hit(i, f"原文 {i}") for i in ids]


def _reply(query: str | None = None, *, model: str = ACTIVE_MIMO_MODEL, name: str = "search_kb",
           arguments: str | None = None) -> ProviderResult:
    calls = []
    if query is not None or arguments is not None:
        calls = [{"id": "call_1", "type": "function",
                  "function": {"name": name, "arguments": arguments or json.dumps({"query": query}, ensure_ascii=False)}}]
    return ProviderResult(
        provider_id="fake", profile_id="fake", model=model, response_id="rid", system_fingerprint=None,
        content="" if calls else "结束", parsed_json=None, tool_calls=calls, reasoning_content=None,
        usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, finish_reason="stop",
        raw_error_code=None, latency_ms=1, retry_count=0, provenance_mode="provider_response_id")


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


def _run(llm: ScriptedLLM, results: dict[str, list[Hit]], first: list[Hit], **kw):
    searched: list[str] = []

    def search(query: str) -> list[Hit]:
        searched.append(query)
        return results.get(query, [])

    return run_search_agent(llm, search, "屋里太闷", first, ACTIVE_MIMO_MODEL, **kw), searched


def test_merge_prefers_chunks_ranked_high_in_several_searches() -> None:
    merged = merge_hits([_hits("a", "b", "c"), _hits("d", "a", "e")])
    assert [h.chunk_id for h in merged] == ["a", "d", "b"]  # a 两次都靠前；d 在第二次排第一，名次高于 b
    assert merge_hits([_hits("a", "b", "c", "d")], pool_size=2) == _hits("a", "b")


def test_model_done_immediately_keeps_first_search_only() -> None:
    llm = ScriptedLLM(_reply())
    outcome, searched = _run(llm, {}, _hits("a", "b", "c"))
    assert searched == [] and outcome.model_calls == 1 and outcome.status == "ok"
    assert [h.chunk_id for h in outcome.chunks] == ["a", "b", "c"]
    assert [s["action"] for s in outcome.steps] == ["search", "stop"]
    assert llm.requests[0].tools and llm.requests[0].tool_choice == "auto"


def test_model_reformulates_and_sees_the_tool_result() -> None:
    llm = ScriptedLLM(_reply("室内通风换气"), _reply())
    outcome, searched = _run(llm, {"室内通风换气": _hits("x", "a")}, _hits("a", "b", "c"))
    assert searched == ["室内通风换气"] and outcome.model_calls == 2
    assert [h.chunk_id for h in outcome.chunks] == ["a", "x", "b"]
    second = llm.requests[1].messages
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_1",
                          "content": json.dumps([{"chunk_id": "x", "text": "原文 x"}, {"chunk_id": "a", "text": "原文 a"}],
                                                ensure_ascii=False)}
    assert outcome.usage["total_tokens"] == 24


def test_repeated_query_stops_without_searching_again() -> None:
    outcome, searched = _run(ScriptedLLM(_reply("屋里太闷")), {}, _hits("a"))
    assert searched == [] and outcome.steps[-1]["reason"] == "duplicate_query" and outcome.status == "ok"


def test_search_count_is_capped_at_two_extra_searches() -> None:
    llm = ScriptedLLM(_reply("q1"), _reply("q2"), _reply("q3"))
    outcome, searched = _run(llm, {}, _hits("a"))
    assert searched == ["q1", "q2"] and outcome.model_calls == 3
    assert outcome.steps[-1] == {"action": "stop", "reason": "search_limit"}


@pytest.mark.parametrize("script,code", [
    ([MediDiagError("LLM_TIMEOUT")], "LLM_TIMEOUT"),
    ([_reply("q", name="other_tool")], "AGENT_BAD_TOOL_CALL"),
    ([_reply(arguments="不是 JSON")], "AGENT_BAD_TOOL_CALL"),
    ([_reply("")], "AGENT_BAD_TOOL_CALL"),
    ([_reply(model="other-model")], "AGENT_MODEL_MISMATCH"),
])
def test_model_problems_fall_back_to_the_evidence_already_found(script, code) -> None:
    outcome, searched = _run(ScriptedLLM(*script), {}, _hits("a", "b"))
    assert searched == [] and outcome.status == "fallback" and outcome.code == code
    assert [h.chunk_id for h in outcome.chunks] == ["a", "b"]


def test_provider_freezes_the_merged_evidence_into_one_bundle(monkeypatch) -> None:
    config = load_application_config(ROOT / "configs/application.yaml")
    rag = RuntimeMedicalRAG.from_config(config, root=ROOT)
    question = "室内通风"
    llm = ScriptedLLM(_reply("新冠 通风 室内空气"), _reply())
    provider = RetrievalAgentWorkflowProvider(rag, llm=llm)  # type: ignore[arg-type]
    result = provider.retrieve(question)
    ids = [c["chunk_id"] for c in result["chunks"]]
    assert result["execution_mode"] == "mimo_grounded" and len(ids) <= 3
    assert [e["chunk_id"] for e in result["evidence_bundle"]["evidence"]] == ids  # 冻结的证据与交给生成的完全一致
    agent = result["search_agent"]
    assert agent["status"] == "ok" and agent["model_calls"] == 2
    assert [s["action"] for s in agent["steps"]] == ["search", "search", "stop"]


def test_provider_without_any_evidence_does_not_reach_generation() -> None:
    config = load_application_config(ROOT / "configs/application.yaml")
    rag = RuntimeMedicalRAG.from_config(config, root=ROOT)
    provider = RetrievalAgentWorkflowProvider(rag, llm=ScriptedLLM(_reply()))  # type: ignore[arg-type]
    result = provider.retrieve("zzzz qqqq")
    assert result["chunks"] == [] and result["top_k"] == 0 and result["search_agent"]["status"] == "ok"


def test_agent_and_rewrite_cannot_be_combined(tmp_path) -> None:
    base = load_application_config(ROOT / "configs/application_rewrite.yaml")
    both = copy.deepcopy(base)
    both["generation"]["agent"] = True  # base 已开启 rewrite
    wrong_type = copy.deepcopy(base)
    wrong_type["generation"].update(rewrite=False, agent="yes")
    for name, content in (("both", both), ("type", wrong_type)):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
        with pytest.raises(ValueError):
            load_application_config(path)


def test_shipped_agent_config_builds_the_agent_provider(monkeypatch, tmp_path) -> None:
    from medidiag.workflow.application import build_application_provider

    monkeypatch.setenv("MIMO_API_KEY", "test-not-a-secret")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com")
    config = load_application_config(ROOT / "configs/application_agent.yaml")
    assert config["generation"]["agent"] is True and not config["generation"].get("rewrite")
    provider = build_application_provider("mimo_grounded", app_config="configs/application_agent.yaml", root=ROOT,
                                          live_budget=tmp_path / "ledger.db")
    assert isinstance(provider, RetrievalAgentWorkflowProvider)
