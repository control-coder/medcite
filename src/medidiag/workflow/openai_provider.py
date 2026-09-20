"""供应商无关的 OpenAI-compatible 工作流适配器。

P1 解除供应商类名绑定；P2 统一阶段执行契约；P3 接入版本化医学 RAG；
P4 将生成与仲裁切换为可审计的单/双专科 Agent runtime。
"""

from __future__ import annotations

from typing import Any

from medidiag.agents.runtime import RuntimeMedicalAgents
from medidiag.errors import MediDiagError
from medidiag.llm import LLMProvider, build_llm_provider
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.review.runtime import RuntimeMedicalReview
from medidiag.workflow.demo_provider_base import DemoWorkflowSupport


class OpenAICompatibleWorkflowProvider(DemoWorkflowSupport):
    """通过 profile 注入统一 LLMProvider，不在业务代码中区分供应商。"""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        *,
        profile_id: str | None = None,
        rag_stage: RuntimeMedicalRAG | None = None,
        agent_stage: RuntimeMedicalAgents | None = None,
        review_stage: RuntimeMedicalReview | None = None,
    ) -> None:
        # 业务层只接收统一 LLMProvider；供应商差异由 profile 与 adapter 收敛。
        from medidiag.compliance.guard import ComplianceGuard

        self.llm = llm or build_llm_provider(profile_id, max_retries=0)
        model = getattr(self.llm, "model", "profile-default")
        self.version = f"{self.llm.profile_id}:{model}"
        self._guard = ComplianceGuard()
        self.rag_stage = rag_stage
        self.agent_stage = agent_stage
        self.review_stage = review_stage

    def normalize(self, question: str) -> dict[str, Any]:
        """使用运行时医学术语归一化，不再回退到演示 fixture。"""
        if self.rag_stage is None:
            raise MediDiagError(
                "RAG_CORPUS_INVALID", detail="live workflow 未装配 RuntimeMedicalRAG"
            )
        return self.rag_stage.normalize(question)

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        """在规划前生成可审计 EvidenceBundle。"""
        if self.rag_stage is None:
            raise MediDiagError(
                "RAG_CORPUS_INVALID", detail="live workflow 未装配 RuntimeMedicalRAG"
            )
        # 只标记适配器链路，不证明真实网络调用或在线验收已发生。
        return {**self.rag_stage.retrieve(normalized_query), "execution_mode": "model_pipeline"}

    @property
    def is_configured(self) -> bool:
        return bool(getattr(self.llm, "is_configured", True))

    def generate(
        self,
        question: str,
        retrieval: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """执行真实单/双专科 Agent；未装配时 fail-closed。"""
        if self.agent_stage is None:
            raise MediDiagError(
                "AGENT_RUNTIME_INVALID", detail="live workflow 未装配 RuntimeMedicalAgents"
            )
        return self.agent_stage.generate(question, retrieval, plan)

    def arbitrate(
        self,
        generation: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        """执行规则基线与受约束仲裁；不允许回退到单起草器说明文本。"""
        if self.agent_stage is None:
            raise MediDiagError(
                "AGENT_RUNTIME_INVALID", detail="live workflow 未装配 RuntimeMedicalAgents"
            )
        return self.agent_stage.arbitrate(generation, retrieval)

    def review(
        self,
        generation: dict[str, Any],
        arbitration: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        """执行固定 NLI、逻辑与合规审核；未装配时 fail-closed。"""
        if self.review_stage is None:
            raise MediDiagError(
                "CITATION_REVIEW_INVALID", detail="live workflow 未装配 RuntimeMedicalReview"
            )
        return self.review_stage.review(generation, arbitration, retrieval)

    def report(
        self,
        case_id: str,
        generation: dict[str, Any],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        """只编排通过审核的 canonical claim，不调用 LLM 做中文医学改写。"""
        if self.review_stage is None:
            raise MediDiagError(
                "REPORT_PROVENANCE_INVALID", detail="live workflow 未装配 RuntimeMedicalReview"
            )
        return self.review_stage.report(case_id, generation, review)
