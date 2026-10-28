"""真实正文检索、确定性摘录展示；不调用生成模型或语义审核器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from medidiag.compliance.guard import MANDATORY_DISCLAIMER, ComplianceGuard
from medidiag.compliance.status import ComplianceStatus
from medidiag.errors import MediDiagError
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.query_router import PROMPT_VERSION as ROUTER_PROMPT_VERSION
from medidiag.workflow.query_router import (
    RouteDecision,
    apply_emergency_hint,
    emergency_hint_of,
    route_query,
)

if TYPE_CHECKING:
    from medidiag.workflow.provider import StageResult

LIMITATION = "真实检索、模拟生成；未调用真实模型或 NLI。词面命中和引用关联不等于语义支持。"


@dataclass
class RetrievalMockWorkflowProvider:
    """仅展示本轮实际命中的短摘录，复用 worker 的空证据拒答分支。"""

    rag_stage: RuntimeMedicalRAG
    version: str = "retrieval-mock-v1"
    # 检索前先做查询路由（generation.query_router）：本类只用规则级，带 llm 的子类还会调用 LLM 路由
    query_router: bool = False
    router_prompt_version: str = ROUTER_PROMPT_VERSION  # 回放 M10–M15 旧调用记录时设为 query-router-v1

    def _route(self, question: str) -> RouteDecision | None:
        """开启路由时返回路由决定；未开启返回 None。"""
        if not self.query_router:
            return None
        return route_query(question, llm=getattr(self, "llm", None), metadata=self.rag_stage.metadata,
                           model=getattr(self, "model", ACTIVE_MIMO_MODEL), prompt_version=self.router_prompt_version)

    def _refusal(self, question: str, route: RouteDecision) -> dict[str, Any]:
        """路由拒答：不检索、不进生成，沿用空证据的拒答分支，路由信息写在结果里。"""
        return {"query": question, "chunks": [], "top_k": 0, "config_hash": self.rag_stage.retrieval_config_hash,
                "corpus_hash": self.rag_stage.corpus_hash, "execution_mode": "retrieval_mock",
                "query_router": route.to_dict()}

    def _page_count(self) -> int:
        return len({chunk.source_id for chunk in self.rag_stage.chunks})

    def normalize(self, question: str) -> dict[str, Any]:
        # ConsultationCreateRequest 的字段标签和缺省占位不是用户问题。
        # 特别是“背景：未提供”中的“提供”会让任何输入误匹配口罩摘录。
        fields = re.fullmatch(r"症状：(.*?)\n持续时间：(.*?)\n背景：(.*)", question, re.S)
        if fields:
            symptoms, duration, background = fields.groups()
            question = " ".join([symptoms, duration, "" if background == "未提供" else background])
        result = self.rag_stage.normalize(question)
        result["normalizer_version"] = "application-fields-whitespace-v1"
        return result

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        route = self._route(normalized_query)
        if route is not None and route.refused:
            return self._refusal(normalized_query, route)
        return self._retrieve_routed(normalized_query, route)

    def _retrieve_routed(self, query: str, route: RouteDecision | None) -> dict[str, Any]:
        """按路由给出的过滤条件检索（不再路由）；没有过滤条件时与原来完全一致。"""
        try:
            if route is not None and route.filters:
                result = self.rag_stage.retrieve(query, filters=route.filters)
            else:
                result = self.rag_stage.retrieve(query)
        except MediDiagError as exc:
            if exc.code != "RAG_NO_EVIDENCE":
                raise
            result = {"query": query, "chunks": [], "top_k": 0,
                      "config_hash": self.rag_stage.retrieval_config_hash,
                      "corpus_hash": self.rag_stage.corpus_hash}
        if route is not None:
            result["query_router"] = route.to_dict()
        return {**result, "execution_mode": "retrieval_mock"}

    def plan(self, normalized_query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
        return {"objective": "展示检索到的相关摘录，不回答个体医疗问题", "topology": "single",
                "evidence_ids": [item["chunk_id"] for item in retrieval["chunks"]],
                "requires_uncertainty": True}

    def generate(self, question: str, retrieval: dict[str, Any], plan: dict[str, Any]) -> StageResult:
        result: dict[str, Any] = {"agents": [], "claims": [
            {"claim_id": f"excerpt_{i:04d}", "text": "检索摘录（不代表问题已获解答）：" + item["text"],
             "citation_chunk_ids": [item["chunk_id"]], "confidence": None}
            for i, item in enumerate(retrieval["chunks"], 1)
        ], "risk_flags": [], "uncertainty": LIMITATION}
        hint = emergency_hint_of(retrieval)
        if hint:
            result["emergency_hint"] = hint
        return result

    def arbitrate(self, generation: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
        return {"verdict": "SINGLE_EXCERPT_DISPLAY", "conflicts": [],
                "selected_claim_ids": [item["claim_id"] for item in generation["claims"]],
                "limitation": LIMITATION}

    def review(self, generation: dict[str, Any], arbitration: dict[str, Any],
               retrieval: dict[str, Any]) -> dict[str, Any]:
        compliance = ComplianceGuard().check_output(generation)
        ids = {item["chunk_id"] for item in retrieval["chunks"]}
        invalid = any(not claim["citation_chunk_ids"] or not set(claim["citation_chunk_ids"]) <= ids
                      for claim in generation["claims"])
        if compliance.blocked or invalid:
            return {"verdict": "ESCALATED", "citation_verdicts": [],
                    "issues": compliance.block_reasons + (["引用无法关联本轮证据"] if invalid else []),
                    "compliance_status": ComplianceStatus.BLOCKED.value}
        return {"verdict": "APPROVED", "issues": [LIMITATION],
                "compliance_status": ComplianceStatus.PASS_WITH_DEMO_LIMITATION.value,
                "citation_verdicts": [
                    {"claim_id": claim["claim_id"], "chunk_id": cid, "verdict": "PARTIAL",
                     "confidence": None, "method": "excerpt_structure_binding_not_nli"}
                    for claim in generation["claims"] for cid in claim["citation_chunk_ids"]]}

    def report(self, case_id: str, generation: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        """路由给出急症提示（``emergency_hint``）时，在摘要和风险提示前附上；子类改 ``_report``。"""
        return apply_emergency_hint(self._report(case_id, generation, review), generation.get("emergency_hint"))

    def _report(self, case_id: str, generation: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        return {"schema_version": "assistant-report-v1", "case_id": case_id,
                "title": "公开正文检索摘录（模拟生成）", "summary": "仅展示相关摘录，不生成医学结论。",
                "claims": generation["claims"], "filtered_claim_count": 0,
                "limitations": [LIMITATION, f"仅含 {self._page_count()} 篇历史网页的摘录；片段可能只部分相关，不能保证答案完整或现行适用。"],
                "risk_warnings": ["不能用于真实医疗决策。"],
                "next_steps": ["请打开原始来源阅读上下文；真实健康问题请咨询专业人员。"],
                "disclaimer": MANDATORY_DISCLAIMER,
                "provenance": {"provider": self.version, "execution_mode": "retrieval_mock"}}
