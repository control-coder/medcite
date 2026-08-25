"""供应商无关的 OpenAI-compatible 工作流演示适配器。

P1 解除 CLI/工作流装配对供应商类名的绑定；P2 统一阶段执行契约；P3 通过
可注入 RuntimeMedicalRAG 接入版本化医学 corpus。
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import ValidationError

from medidiag.errors import MediDiagError
from medidiag.llm import LLMProvider, LLMRequest, build_llm_provider
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.demo_provider_base import DemoWorkflowSupport, DraftPayload
from medidiag.workflow.provider_runtime import ProviderResponse


class OpenAICompatibleWorkflowProvider(DemoWorkflowSupport):
    """通过 profile 注入统一 LLMProvider，不在业务代码中区分供应商。"""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        *,
        profile_id: str | None = None,
        rag_stage: RuntimeMedicalRAG | None = None,
    ) -> None:
        # 业务层只接收统一 LLMProvider；供应商差异由 profile 与 adapter 收敛。
        from medidiag.compliance.guard import ComplianceGuard

        self.llm = llm or build_llm_provider(profile_id, max_retries=0)
        model = getattr(self.llm, "model", "profile-default")
        self.version = f"{self.llm.profile_id}:{model}"
        self._guard = ComplianceGuard()
        self.rag_stage = rag_stage

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
        return self.rag_stage.retrieve(normalized_query)

    @property
    def is_configured(self) -> bool:
        return bool(getattr(self.llm, "is_configured", True))

    def generate(
        self,
        question: str,
        retrieval: dict[str, Any],
        plan: dict[str, Any],
    ) -> ProviderResponse:
        evidence = retrieval.get("chunks", [])
        allowed_ids = {str(item.get("chunk_id")) for item in evidence}
        if not allowed_ids:
            raise MediDiagError("LLM_JSON_INVALID", detail="live demo has no evidence ids")

        result = self.llm.generate(
            LLMRequest(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are the constrained drafting component of an engineering demo. "
                            "The system is not a doctor and must not diagnose, prescribe, triage, or "
                            "make certainty claims. Use only the supplied evidence. Return JSON only; "
                            "do not use Markdown fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self.generation_prompt(question, evidence, plan),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=900,
                prompt_version="live-demo-generation-v2",
            ),
            timeout_s=60,
            idempotency_key=(
                "live-demo:" + hashlib.sha256(question.encode("utf-8")).hexdigest()
            ),
        )
        raw = result.parsed_json
        if raw is None:
            raise MediDiagError(
                "STRUCTURED_OUTPUT_INVALID",
                detail="provider 未返回可校验的 JSON object",
            )
        try:
            draft = DraftPayload.model_validate(raw)
        except ValidationError as exc:
            raise MediDiagError(
                "LLM_JSON_INVALID",
                detail="live provider output failed the drafting schema",
            ) from exc

        claims: list[dict[str, Any]] = []
        for index, claim in enumerate(draft.claims, start=1):
            citation_ids = list(dict.fromkeys(claim.citation_chunk_ids))
            if not set(citation_ids).issubset(allowed_ids):
                raise MediDiagError(
                    "LLM_JSON_INVALID",
                    detail="live provider output referenced an unknown evidence id",
                )
            claims.append(
                {
                    "claim_id": f"claim_{index:04d}",
                    "text": claim.text.strip(),
                    "citation_chunk_ids": citation_ids,
                    "confidence": claim.confidence,
                }
            )

        return ProviderResponse(
            payload={
                "agents": [
                    {
                        "agent_name": "live_evidence_drafter",
                        "status": "SUCCEEDED",
                        "claims": claims,
                        "model": result.model,
                    }
                ],
                "claims": claims,
                "risk_flags": list(dict.fromkeys(draft.risk_flags))
                + ["qualified_clinician_review_required"],
                "uncertainty": draft.uncertainty.strip(),
            },
            request_id=result.response_id,
            metadata={
                "provider_id": result.provider_id,
                "provider_profile": result.profile_id,
                "generation_model": result.model,
                "system_fingerprint": result.system_fingerprint,
                "provenance_mode": result.provenance_mode,
                "provider_retry_count": result.retry_count,
                "provider_latency_ms": result.latency_ms,
                "filtered_parameters": list(result.filtered_parameters),
                "prompt_cache_hit_tokens": result.usage.get("prompt_cache_hit_tokens", 0),
                "prompt_cache_miss_tokens": result.usage.get("prompt_cache_miss_tokens", 0),
            },
        )
