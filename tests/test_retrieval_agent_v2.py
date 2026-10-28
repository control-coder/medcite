"""检索 Agent v2（M11）的离线回归：脚本化的假模型与假检索器，不联网、不读 .cache。"""
from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from medidiag.errors import MediDiagError
from medidiag.llm.contracts import LLMRequest, ProviderResult
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.rag.retrieval import SearchResult
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.application import load_application_config
from medidiag.workflow.retrieval_agent import merge_hits
from medidiag.workflow.retrieval_agent_v2 import (
    HYDE_PROMPT_VERSION,
    AgentTools,
    AgentV2Result,
    RetrievalAgentV2WorkflowProvider,
    run_search_agent_v2,
)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _hits(*ids: str) -> list[Hit]:
    return [Hit(i, f"原文 {i}") for i in ids]


def _call(name: str, call_id: str, assessments: Mapping[str, str] | None = None, **args: Any) -> dict[str, Any]:
    args["assessments"] = dict(assessments or {})
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def _reply(*calls: dict[str, Any], content: str = "", tokens: int = 100, model: str = ACTIVE_MIMO_MODEL) -> ProviderResult:
    return ProviderResult(
        provider_id="fake", profile_id="fake", model=model, response_id="rid", system_fingerprint=None,
        content=content, parsed_json=None, tool_calls=list(calls), reasoning_content=None,
        usage={"input_tokens": tokens - 10, "output_tokens": 10, "total_tokens": tokens}, finish_reason="stop",
        raw_error_code=None, latency_ms=1, retry_count=0, provenance_mode="provider_response_id")


class FakeLLM:
    """决策调用按脚本依次返回（异常项会抛出）；HyDE 调用（prompt_version=hyde-v1）固定返回一段假设回答。"""

    provider_id = "fake"
    profile_id = "fake"

    def __init__(self, *script: ProviderResult | Exception) -> None:
        self.script = list(script)
        self.decisions: list[LLMRequest] = []
        self.hyde: list[LLMRequest] = []
        self._lock = threading.Lock()

    def generate(self, request: LLMRequest, *, timeout_s: float, idempotency_key: str) -> ProviderResult:
        with self._lock:
            if request.prompt_version == HYDE_PROMPT_VERSION:
                self.hyde.append(request)
                return _reply(content="假设回答：通风可以降低室内病原体浓度", tokens=50)
            self.decisions.append(request)
            item = self.script[len(self.decisions) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    def capabilities(self) -> Any:  # pragma: no cover - 协议要求，测试不使用
        raise NotImplementedError


def _tools(results: Mapping[str, Sequence[Hit]] | None = None, *, hyde: bool = False,
           delays: Mapping[str, float] | None = None) -> tuple[AgentTools, list[tuple[str, dict[str, str]]]]:
    searched: list[tuple[str, dict[str, str]]] = []
    table = results or {}

    def search(query: str, filters: Mapping[str, str]) -> Sequence[Hit]:
        time.sleep((delays or {}).get(query, 0))
        searched.append((query, dict(filters)))
        return table.get(query, [])

    return AgentTools(
        search=search, expand=lambda hit: f"父块：{hit.text}",
        merge=lambda lists: merge_hits(lists, 3),
        hyde=(lambda hypothetical, question: _hits("h1")) if hyde else None), searched


def _run(llm: FakeLLM, tools: AgentTools, first: list[Hit] | None = None, **kw: Any) -> AgentV2Result:
    return run_search_agent_v2(llm, tools, "屋里太闷", _hits("a", "b", "c") if first is None else first,
                               ACTIVE_MIMO_MODEL, **kw)


def _tool_messages(request: LLMRequest) -> list[dict[str, Any]]:
    return [m for m in request.messages if m["role"] == "tool"]


# ---- 并行回填 ----

def test_parallel_calls_are_backfilled_by_tool_call_id_in_order() -> None:
    llm = FakeLLM(
        _reply(_call("search_kb", "c1", query="子问题一"), _call("search_kb", "c2", query="子问题二"),
               _call("expand_context", "c3", chunk_id="a")),
        _reply(_call("finish", "c4", selected_chunk_ids=["x", "y"], verdict="sufficient")))
    tools, searched = _tools({"子问题一": _hits("x"), "子问题二": _hits("y")}, delays={"子问题一": 0.15})
    outcome = _run(llm, tools)
    assert {q for q, _ in searched} == {"子问题一", "子问题二"}
    assert searched[0][0] == "子问题二"  # 慢的调用后完成，说明确实并行
    messages = _tool_messages(llm.decisions[1])
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2", "c3"]  # 回填顺序与调用顺序一致，与完成顺序无关
    assert json.loads(messages[0]["content"])["results"][0]["chunk_id"] == "x"
    assert json.loads(messages[1]["content"])["results"][0]["chunk_id"] == "y"
    assert json.loads(messages[2]["content"])["parent_text"] == "父块：原文 a"
    assert outcome.parallel_rounds == 1 and outcome.tool_calls == 3 and outcome.status == "ok"
    assert {h.chunk_id for h in outcome.chunks} == {"x", "y"}


# ---- 预算 ----

def test_tool_budget_leaves_only_finish_for_the_last_model_call() -> None:
    llm = FakeLLM(
        _reply(_call("search_kb", "c1", query="q1"), _call("search_kb", "c2", query="q2")),
        _reply(_call("search_kb", "c3", query="q3"), _call("search_kb", "c4", query="q4"), _call("search_kb", "c5", query="q5")),
        _reply(_call("finish", "c6", selected_chunk_ids=["a"], verdict="sufficient")))
    tools, searched = _tools()
    outcome = _run(llm, tools)
    assert [q for q, _ in searched] == ["q1", "q2", "q3", "q4"] and outcome.tool_calls == 4
    assert json.loads(_tool_messages(llm.decisions[2])[-1]["content"]) == {"skipped": "tool_limit"}  # 第 5 个调用被拒绝
    last_tools = llm.decisions[2].tools
    assert last_tools is not None and [t["function"]["name"] for t in last_tools] == ["finish"]
    assert outcome.model_calls == 3 and [h.chunk_id for h in outcome.chunks] == ["a"]


def test_model_call_budget_stops_a_model_that_never_finishes() -> None:
    searches = [_reply(_call("search_kb", f"c{i}", query=f"q{i}")) for i in range(1, 6)]
    llm = FakeLLM(*searches)
    outcome = _run(llm, _tools()[0])
    assert outcome.model_calls == 5 and outcome.tool_calls == 4 and outcome.stop_reason == "tool_limit"
    assert [h.chunk_id for h in outcome.chunks] == ["a", "b", "c"]  # 用已有证据，不报错
    assert outcome.status == "ok"


def test_token_budget_stops_before_the_next_model_call() -> None:
    llm = FakeLLM(_reply(_call("search_kb", "c1", query="q1"), tokens=7000),
                  _reply(_call("search_kb", "c2", query="q2"), tokens=7000))
    outcome = _run(llm, _tools()[0])
    assert outcome.model_calls == 2 and outcome.stop_reason == "token_limit" and outcome.usage["total_tokens"] == 14000


def test_repeated_query_is_not_executed() -> None:
    llm = FakeLLM(_reply(_call("search_kb", "c1", query="屋里太闷")))
    tools, searched = _tools()
    outcome = _run(llm, tools)
    assert searched == [] and outcome.tool_calls == 0 and outcome.stop_reason == "no_new_tool"
    assert outcome.steps[1]["action"] == "model" and any(s.get("reason") == "duplicate_query" for s in outcome.steps)
    assert [h.chunk_id for h in outcome.chunks] == ["a", "b", "c"]


def test_search_filters_keep_only_applied_keys_and_merge_with_defaults() -> None:
    call = _call("search_kb", "c1", query="最新疫情", filters={"period": "latest", "topic": "疟疾"})
    llm = FakeLLM(_reply(call), _reply(_call("finish", "c2", selected_chunk_ids=[], verdict="sufficient")))
    tools, searched = _tools()
    _run(llm, tools, default_filters={"source_type": "epidemic_report"})
    assert searched == [("最新疫情", {"source_type": "epidemic_report", "period": "latest"})]


# ---- 检索评估（CRAG） ----

def test_all_incorrect_means_insufficient_evidence() -> None:
    llm = FakeLLM(
        _reply(_call("search_kb", "c1", {"a": "Incorrect", "b": "Incorrect", "c": "Incorrect"}, query="q1")),
        _reply(_call("finish", "c2", {"x": "Incorrect"}, selected_chunk_ids=["a", "x"], verdict="sufficient")))
    outcome = _run(llm, _tools({"q1": _hits("x")})[0])
    assert outcome.status == "insufficient" and outcome.chunks == []
    assert outcome.labels == {"a": "Incorrect", "b": "Incorrect", "c": "Incorrect", "x": "Incorrect"}


def test_verdict_insufficient_discards_evidence() -> None:
    llm = FakeLLM(_reply(_call("finish", "c1", {"a": "Correct"}, selected_chunk_ids=["a"], verdict="insufficient")))
    outcome = _run(llm, _tools()[0])
    assert outcome.status == "insufficient" and outcome.chunks == []


def test_selected_ids_must_be_observed_and_not_incorrect() -> None:
    llm = FakeLLM(_reply(_call("finish", "c1", {"a": "Correct", "b": "Incorrect", "ghost": "Correct"},
                               selected_chunk_ids=["a", "b", "ghost", "not_seen"], verdict="sufficient")))
    outcome = _run(llm, _tools()[0])
    assert [h.chunk_id for h in outcome.chunks] == ["a"] and outcome.status == "ok"
    dropped = [s for s in outcome.steps if s["action"] == "drop_selected"]
    assert dropped and set(dropped[0]["chunk_ids"]) == {"b", "ghost", "not_seen"}
    assert "ghost" not in outcome.labels  # 未观察过的块不接受标签


# ---- 降级 ----

def test_model_error_falls_back_to_evidence_so_far_plus_fixed_pipeline() -> None:
    llm = FakeLLM(_reply(_call("search_kb", "c1", {"a": "Incorrect"}, query="q1")), MediDiagError("LLM_TIMEOUT"))
    outcome = _run(llm, _tools({"q1": _hits("x")})[0], first=_hits("a", "b"))
    assert outcome.status == "fallback" and outcome.code == "LLM_TIMEOUT"
    assert {h.chunk_id for h in outcome.chunks} == {"a", "b", "x"}  # 固定链路的首轮结果补齐，已得证据保留
    assert outcome.steps[-1]["status"] == "fallback"


@pytest.mark.parametrize("reply,code", [
    (_reply(_call("unknown_tool", "c1")), "AGENT_BAD_TOOL_CALL"),
    (_reply({"id": "c1", "type": "function", "function": {"name": "search_kb", "arguments": "不是 JSON"}}), "AGENT_BAD_TOOL_CALL"),
    (_reply(_call("finish", "c1", selected_chunk_ids=["a"], verdict="maybe")), "AGENT_BAD_TOOL_CALL"),
    (_reply(model="other-model"), "AGENT_MODEL_MISMATCH"),
])
def test_unusable_model_output_degrades_instead_of_failing(reply: ProviderResult, code: str) -> None:
    outcome = _run(FakeLLM(reply), _tools()[0])
    assert outcome.status == "fallback" and outcome.code == code
    assert [h.chunk_id for h in outcome.chunks] == ["a", "b", "c"]


# ---- 与真实 RuntimeMedicalRAG 集成 ----

def _chunk(chunk_id: str, text: str, parent_id: str) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=chunk_id, source="资料", source_id="doc1", text=text,
                          metadata={"doc_id": "doc1", "title": "资料", "parent_id": parent_id, "source_type": "who_factsheet"})


CHUNKS = [_chunk("k1", "通风可以降低室内病原体浓度", "p1"), _chunk("k2", "勤洗手预防传染", "p2"),
          _chunk("k3", "孤立子块", "missing"), _chunk("k4", "假设回答命中的子块", "p1")]


class _FakeRetriever:
    """按查询返回预设的块；``dense`` 一路用于 HyDE。"""

    def __init__(self, table: Mapping[str, Sequence[str]], dense_table: Mapping[str, Sequence[str]] | None) -> None:
        self.table = table
        self.by_id = {c.chunk_id: c for c in CHUNKS}
        self.dense = _FakeRetriever(dense_table, None) if dense_table is not None else None
        self.queries: list[str] = []

    def build_index(self) -> None:
        pass

    def search(self, query: str, top_k: int = 5, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        self.queries.append(query)
        ids = [i for i in self.table.get(query, ["k2"]) if allowed_ids is None or i in allowed_ids]
        return [SearchResult(chunk_id=i, final_score=1.0 / (n + 1), chunk=self.by_id[i]) for n, i in enumerate(ids[:top_k])]


def _provider(llm: FakeLLM, *, dense: bool = True) -> tuple[RetrievalAgentV2WorkflowProvider, _FakeRetriever]:
    retriever = _FakeRetriever({"室内通风": ["k2", "k3"], "通风 病原体": ["k1"]}, {"假设回答：通风可以降低室内病原体浓度": ["k4", "k1"]} if dense else None)
    rag = RuntimeMedicalRAG(chunks=CHUNKS, corpus_version="t", retrieval_config={"method": "hybrid", "top_k": 3},
                            retriever=retriever, corpus_path="c.jsonl", leakage_gate={"status": "passed"})
    provider = RetrievalAgentV2WorkflowProvider(rag, llm=llm, parents={"p1": "父块 p1：通风与空气质量的完整段落"})
    return provider, retriever


def test_expand_context_returns_the_parent_but_never_cites_it() -> None:
    llm = FakeLLM(
        _reply(_call("expand_context", "c1", {"k2": "Ambiguous", "k3": "Ambiguous"}, chunk_id="k2"),
               _call("search_kb", "c2", query="通风 病原体")),
        _reply(_call("expand_context", "c3", {"k1": "Correct"}, chunk_id="k1")),
        _reply(_call("finish", "c4", selected_chunk_ids=["k1"], verdict="sufficient")))
    provider, _ = _provider(llm)
    result = provider.retrieve("室内通风")
    parents = [json.loads(m["content"]) for req in llm.decisions[1:] for m in _tool_messages(req) if "parent_text" in m["content"]]
    assert {"chunk_id": "k2", "parent_text": "勤洗手预防传染"} in parents  # 父块缺失时返回子块本身
    assert {"chunk_id": "k1", "parent_text": "父块 p1：通风与空气质量的完整段落"} in parents
    assert [c["chunk_id"] for c in result["chunks"]] == ["k1"]  # 证据只有子块，父块文本不进入证据包
    assert "父块 p1" not in json.dumps(result["evidence_bundle"], ensure_ascii=False)
    assert result["search_agent"]["version"] == 2 and result["search_agent"]["tool_calls"] == 3


def test_hyde_search_counts_as_a_model_call_and_uses_the_vector_path() -> None:
    llm = FakeLLM(
        _reply(_call("hyde_search", "c1", {"k2": "Incorrect", "k3": "Incorrect"}, question="室内通风"), tokens=200),
        _reply(_call("finish", "c2", {"k4": "Correct", "k1": "Correct"}, selected_chunk_ids=["k4", "k1"], verdict="sufficient"),
               tokens=200))
    provider, retriever = _provider(llm)
    result = provider.retrieve("室内通风")
    agent = result["search_agent"]
    assert len(llm.hyde) == 1 and agent["model_calls"] == 3  # 两次决策加一次 HyDE
    assert agent["usage"]["total_tokens"] == 450  # HyDE 的 token 也计入
    assert retriever.dense is not None and retriever.dense.queries == ["假设回答：通风可以降低室内病原体浓度"]
    assert sorted(c["chunk_id"] for c in result["chunks"]) == ["k1", "k4"]
    first_tools = llm.decisions[0].tools
    assert first_tools is not None and "hyde_search" in [t["function"]["name"] for t in first_tools]


def test_hyde_tool_is_not_offered_without_a_vector_path() -> None:
    llm = FakeLLM(_reply(_call("finish", "c1", {"k2": "Correct"}, selected_chunk_ids=["k2"], verdict="sufficient")))
    provider, _ = _provider(llm, dense=False)
    provider.retrieve("室内通风")
    tools = llm.decisions[0].tools
    assert tools is not None and "hyde_search" not in [t["function"]["name"] for t in tools]


def test_provider_without_evidence_does_not_reach_generation() -> None:
    llm = FakeLLM(_reply(_call("finish", "c1", {"k2": "Incorrect", "k3": "Incorrect"}, selected_chunk_ids=[], verdict="insufficient")))
    provider, _ = _provider(llm)
    result = provider.retrieve("室内通风")
    assert result["chunks"] == [] and result["top_k"] == 0 and result["search_agent"]["status"] == "insufficient"


# ---- 配置白名单 ----

def test_config_accepts_agent_version_2_and_rejects_illegal_combinations(tmp_path: Path) -> None:
    config = load_application_config(ROOT / "configs/application_v4_agent.yaml")
    assert config["generation"]["agent"] is True and config["generation"]["agent_version"] == 2
    assert not config["generation"].get("query_router")
    assert "agent_version" not in load_application_config(ROOT / "configs/application_v4_hybrid.yaml")["generation"]
    v1 = copy.deepcopy(config)
    v1["generation"]["agent_version"] = 1
    load_application_config(_dump(tmp_path, "v1", v1))
    bad: dict[str, dict[str, Any]] = {}
    for name, patch in {"no_agent": {"agent": False}, "version3": {"agent_version": 3}, "string": {"agent_version": "2"},
                        "bool": {"agent_version": True}, "with_rewrite": {"rewrite": True}}.items():
        bad[name] = copy.deepcopy(config)
        bad[name]["generation"].update(patch)
    for name, value in bad.items():
        with pytest.raises(ValueError):
            load_application_config(_dump(tmp_path, name, value))


def _dump(tmp_path: Path, name: str, content: dict[str, Any]) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(content, allow_unicode=True), encoding="utf-8")
    return path
