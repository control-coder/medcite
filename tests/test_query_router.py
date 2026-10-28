"""M10 查询路由与元数据过滤的离线测试：假 LLM、小语料，不联网、不读 .cache。"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from medidiag.errors import MediDiagError
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.rag.metadata import MetadataIndex, search_with_relaxation, section_type_of
from medidiag.rag.retrieval import Bm25Retriever
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.abstention import abstention_payload
from medidiag.workflow.application import load_application_config
from medidiag.workflow.query_router import (
    EMERGENCY_HINT_MESSAGE,
    PROMPT_VERSION,
    PROMPT_VERSION_V1,
    ROUTER_PROMPT_V1,
    ROUTER_PROMPT_V2,
    build_router_request,
    route_by_rules,
    route_query,
)
from medidiag.workflow.retrieval_mock import RetrievalMockWorkflowProvider

ROOT = Path(__file__).resolve().parents[1]


def _chunk(chunk_id: str, text: str, *, doc_id: str, title: str, source_type: str = "who_factsheet",
           period: str | None = None, section: str = "") -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=chunk_id, source=title, source_id=doc_id, text=text, metadata={
        "doc_id": doc_id, "title": title, "source_type": source_type, "period": period, "section_path": [section]})


CHUNKS = [
    _chunk("m1", "疟疾 症状 发热", doc_id="d_malaria", title="疟疾", section="症状"),
    _chunk("m2", "疟疾 预防 蚊帐", doc_id="d_malaria", title="疟疾", section="预防"),
    _chunk("f1", "流感 症状 发热", doc_id="d_flu", title="季节性流感", section="症状"),
    _chunk("f2", "流感 疫苗", doc_id="d_flu2", title="流行性感冒诊疗方案（2025年版）", source_type="nhc_document"),
    _chunk("r1", "疫情 流感 病例", doc_id="cdc_202503", title="2025年3月全国传染病疫情概况",
           source_type="epidemic_report", period="2025-03"),
    _chunk("r2", "疫情 流感 病例", doc_id="cdc_202504", title="2025年4月全国传染病疫情概况",
           source_type="epidemic_report", period="2025-04"),
]


class _FakeLLM:
    """按脚本返回；脚本项为异常时抛出。"""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, request: Any, *, timeout_s: float, idempotency_key: str) -> Any:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(model=ACTIVE_MIMO_MODEL, usage={"input_tokens": 10, "output_tokens": 5},
                               parsed_json=self.reply)


# ---- 规则 ----

@pytest.mark.parametrize("question", [
    "孩子现在喘不上气，嘴唇都发紫了，我该怎么做？",
    "发烧到40度，人已经抽搐了，现在该做什么？",
    "我爸突然胸痛，出冷汗，怎么办",
    "我想自杀，不想活了",
])
def test_rule_detects_emergency(question: str) -> None:
    decision = route_by_rules(question)
    assert decision is not None and decision.intent == "emergency" and decision.source == "rule"
    assert "120" in (decision.message or "") and decision.refused


@pytest.mark.parametrize("question", [
    "宝宝一岁半，体重11公斤，退烧药一次该喂多少毫升？",
    "我核磁报告写着腰椎间盘突出，你帮我看看我这个情况要不要做手术？",
    "我妈妈血压正常了，可以停药吗",
])
def test_rule_detects_individual_care(question: str) -> None:
    decision = route_by_rules(question)
    assert decision is not None and decision.intent == "individual_care" and decision.source == "rule"


@pytest.mark.parametrize("question", [
    "BMI 达到多少算肥胖？",
    "胸痛一般是什么原因引起的？",
    "我国高血压的患病率是多少？",
    "全球每年有多少人死于自杀？",
    "季节性流感每年大概有多少病例，重症有多少？",
    "每天应该摄入多少盐才算健康？",
    "全球每天有多少人死于药物过量？",
    "抽搐的常见原因和其他症状有哪些？",
])
def test_rule_does_not_hit_popular_science(question: str) -> None:
    assert route_by_rules(question) is None


# ---- LLM 路由 ----

def test_llm_route_validates_enums_and_drops_unknown_values() -> None:
    index = MetadataIndex(CHUNKS)
    llm = _FakeLLM({"intent": "popular_science", "filters": {
        "topic": "疟疾", "section_type": "symptoms", "source_type": "weibo", "period": "2099-01",
        "unknown_key": "x"}})
    decision = route_query("疟疾有什么症状", llm=llm, metadata=index)  # type: ignore[arg-type]
    assert decision.source == "llm" and decision.intent == "popular_science"
    assert decision.filters == {}  # topic、section_type 有效但不在 APPLIED_FILTER_KEYS 内，不应用
    assert {(d["key"], d["reason"]) for d in decision.dropped_filters} == {
        ("source_type", "not_in_enum"), ("period", "not_in_corpus"), ("unknown_key", "unknown_key"),
        ("topic", "not_applied"), ("section_type", "not_applied")}
    assert decision.usage == {"input_tokens": 10, "output_tokens": 5}


def test_llm_applies_only_source_type_and_period() -> None:
    llm = _FakeLLM({"intent": "popular_science", "filters": {
        "topic": "流感", "section_type": "epidemiology", "source_type": "epidemic_report", "period": "latest"}})
    decision = route_query("最近一期流感病例数", llm=llm, metadata=MetadataIndex(CHUNKS))  # type: ignore[arg-type]
    assert decision.filters == {"source_type": "epidemic_report", "period": "2025-04"}
    assert [d["key"] for d in decision.dropped_filters] == ["topic", "section_type"]


def test_llm_refusal_intents_use_templates_and_ignore_filters() -> None:
    llm = _FakeLLM({"intent": "out_of_scope", "filters": {"topic": "疟疾"}})
    decision = route_query("车险理赔需要什么材料", llm=llm, metadata=MetadataIndex(CHUNKS))  # type: ignore[arg-type]
    assert decision.refused and decision.filters == {} and decision.message


@pytest.mark.parametrize("reply", [
    MediDiagError("LLM_TIMEOUT"), {"intent": "chat", "filters": {}}, {"intent": "popular_science", "filters": []},
    None,
])
def test_llm_failure_falls_back_to_popular_science(reply: Any) -> None:
    decision = route_query("什么是疟疾", llm=_FakeLLM(reply), metadata=MetadataIndex(CHUNKS))  # type: ignore[arg-type]
    assert decision.intent == "popular_science" and decision.source == "fallback"
    assert decision.filters == {} and decision.fallback_reason


def test_rule_hit_skips_llm() -> None:
    llm = _FakeLLM({"intent": "popular_science", "filters": {}})
    decision = route_query("孩子现在喘不上气，嘴唇发紫", llm=llm, metadata=MetadataIndex(CHUNKS))  # type: ignore[arg-type]
    assert decision.intent == "emergency" and llm.calls == 0


# ---- query-router-v2：提示词定义与急症降级 ----

def test_v2_prompt_defines_intents_and_v1_text_is_kept() -> None:
    assert PROMPT_VERSION == "query-router-v2"
    request = build_router_request("什么是疟疾")
    assert request.prompt_version == "query-router-v2" and request.messages[0]["content"] == ROUTER_PROMPT_V2
    for phrase in ("此刻正在发生", "通用急救", "流行病统计", "与健康、医学、公共卫生完全无关", "本人或其家人具体病情"):
        assert phrase in ROUTER_PROMPT_V2
    v1 = build_router_request("什么是疟疾", prompt_version=PROMPT_VERSION_V1)
    assert v1.prompt_version == "query-router-v1" and v1.messages[0]["content"] == ROUTER_PROMPT_V1
    assert ROUTER_PROMPT_V1.startswith("你是公共卫生科普问答的查询路由助手。") and "emergency（描述正在发生的危急症状" in ROUTER_PROMPT_V1


def test_v2_llm_emergency_without_rule_hit_continues_as_popular_science_with_hint() -> None:
    llm = _FakeLLM({"intent": "emergency", "filters": {"topic": "疟疾"}})
    decision = route_query("被蛇咬了现场怎么急救", llm=llm, metadata=MetadataIndex(CHUNKS))  # type: ignore[arg-type]
    assert decision.intent == "popular_science" and decision.source == "llm" and not decision.refused
    assert decision.emergency_hint and decision.message == EMERGENCY_HINT_MESSAGE and decision.filters == {}
    data = decision.to_dict()
    assert data["emergency_hint"] is True and data["version"] == "query-router-v2"


def test_v1_keeps_old_behavior_and_report_shape() -> None:
    llm = _FakeLLM({"intent": "emergency", "filters": {}})
    decision = route_query("被蛇咬了现场怎么急救", llm=llm, metadata=MetadataIndex(CHUNKS),  # type: ignore[arg-type]
                           prompt_version=PROMPT_VERSION_V1)
    assert decision.intent == "emergency" and decision.refused and not decision.emergency_hint
    data = decision.to_dict()
    assert "emergency_hint" not in data and data["version"] == "query-router-v1"
    assert route_query("孩子现在喘不上气，嘴唇发紫", prompt_version=PROMPT_VERSION_V1).to_dict()["version"] == "query-router-v1"
    with pytest.raises(ValueError):
        route_query("什么是疟疾", prompt_version="query-router-v0")


def test_rule_emergency_still_refused_under_v2() -> None:
    decision = route_query("孩子现在喘不上气，嘴唇发紫", llm=_FakeLLM({"intent": "popular_science", "filters": {}}),  # type: ignore[arg-type]
                           metadata=MetadataIndex(CHUNKS))
    assert decision.intent == "emergency" and decision.refused and decision.source == "rule" and not decision.emergency_hint


def test_emergency_hint_is_shown_before_answer() -> None:
    rag = RuntimeMedicalRAG(
        chunks=CHUNKS, corpus_version="t", retrieval_config={"method": "bm25", "top_k": 2},
        retriever=Bm25Retriever(CHUNKS, bm25_tokenizer="whitespace"), corpus_path="c.jsonl",
        leakage_gate={"status": "passed"})
    rag.retriever.build_index()
    provider = RetrievalMockWorkflowProvider(rag, query_router=True)
    provider.llm = _FakeLLM({"intent": "emergency", "filters": {}})  # type: ignore[attr-defined]
    retrieval = provider.retrieve("疟疾 症状")
    assert retrieval["chunks"] and retrieval["query_router"]["emergency_hint"] is True
    generation = provider.generate("疟疾 症状", retrieval, provider.plan("疟疾 症状", retrieval))
    assert isinstance(generation, dict) and generation["emergency_hint"] == EMERGENCY_HINT_MESSAGE
    report = provider.report("case-1", generation, provider.review(generation, {}, retrieval))
    assert report["summary"].startswith(EMERGENCY_HINT_MESSAGE) and report["risk_warnings"][0] == EMERGENCY_HINT_MESSAGE
    # 无提示时报告与原来一致
    plain = provider.report("case-1", {**generation, "emergency_hint": None}, {})
    assert plain["summary"] == "仅展示相关摘录，不生成医学结论。" and plain["risk_warnings"] == ["不能用于真实医疗决策。"]


def test_router_prompt_config_choice(tmp_path: Path) -> None:
    config = load_application_config(ROOT / "configs/application_v4_router.yaml")
    bad = copy.deepcopy(config)
    bad["generation"]["router_prompt"] = "v9"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_application_config(path)
    good = copy.deepcopy(config)
    good["generation"]["router_prompt"] = "v1"
    path.write_text(yaml.safe_dump(good), encoding="utf-8")
    assert load_application_config(path)["generation"]["router_prompt"] == "v1"


# ---- 元数据 ----

def test_section_type_and_topic_mapping() -> None:
    assert section_type_of(["疟疾", "症状和体征"]) == "symptoms"
    assert section_type_of(["预防和控制"]) == "prevention"
    assert section_type_of(["重要事实"]) == "key_facts"
    assert section_type_of(["概述"]) == "definition"
    assert section_type_of(["世卫组织的应对"]) is None and section_type_of([""]) is None
    index = MetadataIndex(CHUNKS)
    assert index.resolve_topic("疟疾") == {"d_malaria"}
    assert index.resolve_topic("流行性感冒") == {"d_flu", "d_flu2"}  # 别名同时命中 WHO 与卫健委文档
    assert index.resolve_topic("不存在的病") == set()


def test_latest_period_resolves_to_max_period() -> None:
    index = MetadataIndex(CHUNKS)
    valid, dropped = index.validate({"period": "latest", "source_type": "epidemic_report"})
    assert valid == {"period": "2025-04", "source_type": "epidemic_report"} and not dropped
    assert index.allowed_ids(valid) == {"r2"}


def test_allowed_ids_masks_before_top_k() -> None:
    retriever = Bm25Retriever(CHUNKS, bm25_tokenizer="whitespace")
    filtered = retriever.search("流感 发热", top_k=3, allowed_ids={"m1", "f2"})
    assert {r.chunk_id for r in filtered} <= {"m1", "f2"} and filtered
    assert retriever.search("流感", top_k=3, allowed_ids=set()) == []


def test_default_search_unchanged_without_allowed_ids() -> None:
    retriever = Bm25Retriever(CHUNKS, bm25_tokenizer="whitespace")
    assert [r.chunk_id for r in retriever.search("流感")] == [
        r.chunk_id for r in retriever.search("流感", allowed_ids=None)]


def test_relaxation_drops_conditions_in_order() -> None:
    index = MetadataIndex(CHUNKS)
    seen: list[set[str] | None] = []

    def search(allowed: set[str] | None) -> list[str]:
        seen.append(allowed)
        # 只有放宽到只剩 source_type 时才凑够 2 条
        return ["a", "b"] if allowed is not None and len(allowed) >= 2 else ["a"]

    results, trace = search_with_relaxation(
        search, index, {"topic": "流感", "section_type": "symptoms", "source_type": "epidemic_report"}, 2)
    assert results == ["a", "b"]
    assert trace.relaxed == ["section_type", "topic"] and trace.applied == {"source_type": "epidemic_report"}
    assert len(trace.attempts) == 3


def test_runtime_search_filtered_records_trace() -> None:
    rag = RuntimeMedicalRAG(
        chunks=CHUNKS, corpus_version="t", retrieval_config={"method": "bm25", "top_k": 2},
        retriever=Bm25Retriever(CHUNKS, bm25_tokenizer="whitespace"), corpus_path="c.jsonl",
        leakage_gate={"status": "passed"})
    rag.retriever.build_index()
    evidence, trace = rag.search_filtered("流感 病例", filters={"topic": "疟疾", "source_type": "nhc_document",
                                                                "period": "bad"})
    assert trace is not None and [d["key"] for d in trace.dropped] == ["period"]
    assert trace.relaxed  # 疟疾与卫健委文档没有交集，必须放宽
    assert evidence and rag.search_filtered("流感")[1] is None


# ---- 应用集成 ----

def test_mock_provider_router_refuses_without_retrieval() -> None:
    rag = RuntimeMedicalRAG(
        chunks=CHUNKS, corpus_version="t", retrieval_config={"method": "bm25", "top_k": 2},
        retriever=Bm25Retriever(CHUNKS, bm25_tokenizer="whitespace"), corpus_path="c.jsonl",
        leakage_gate={"status": "passed"})
    provider = RetrievalMockWorkflowProvider(rag, query_router=True)
    result = provider.retrieve("孩子现在喘不上气，嘴唇发紫了")
    assert result["chunks"] == [] and result["query_router"]["intent"] == "emergency"
    report = abstention_payload("report", "case-1", result)
    assert "120" in report["summary"] and report["provenance"]["guard"] == "query-router-v1"
    rag.retriever.build_index()
    passed = provider.retrieve("疟疾 症状")
    assert passed["chunks"] and passed["query_router"]["source"] == "rule"
    assert RetrievalMockWorkflowProvider(rag).retrieve("疟疾 症状").get("query_router") is None


# ---- 配置白名单 ----

def test_router_config_accepted_and_illegal_value_rejected(tmp_path: Path) -> None:
    config = load_application_config(ROOT / "configs/application_v4_router.yaml")
    assert config["generation"]["query_router"] is True
    assert "query_router" not in load_application_config(ROOT / "configs/application_v4_hybrid.yaml")["generation"]
    bad = copy.deepcopy(config)
    bad["generation"]["query_router"] = "yes"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_application_config(path)
