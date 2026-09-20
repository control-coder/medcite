"""用于最小本地演示的在线 DeepSeek Provider。

该适配器只在结构化起草阶段发起一次外部调用。归一化、检索和审核保持确定性，使演示流程快速、有界且可审计。检索载荷明确是本地演示 fixture，不是医学知识库或正式 RAG 评测。
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from medidiag.agents.llm_client import LLMClient
from medidiag.compliance.guard import MANDATORY_DISCLAIMER, ComplianceGuard
from medidiag.compliance.status import ComplianceStatus
from medidiag.errors import MediDiagError
from medidiag.workflow.provider_runtime import ProviderResponse


class _DraftClaim(BaseModel):
    text: str = Field(min_length=8, max_length=1200)
    citation_chunk_ids: list[str] = Field(min_length=1, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)


class _DraftPayload(BaseModel):
    claims: list[_DraftClaim] = Field(min_length=1, max_length=4)
    uncertainty: str = Field(min_length=8, max_length=1200)
    risk_flags: list[str] = Field(default_factory=list, max_length=8)


class DeepSeekWorkflowProvider:
    """使用在线 DeepSeek 兼容补全服务完成保守的证据起草。

    它刻意不是临床诊断 Provider。输出受本地演示证据约束，并标记为仅作结构审核，因为固定 NLI 引用评测不属于该演示路径。
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        # 该 Provider 始终由 ProviderCallRunner 驱动，后者负责
        # 有界重试预算，并为每次尝试写入一个可审计的 provider_call 事件。
        # 如果继续启用 LLMClient 自身的重试，两层重试会相乘
        # （3 x 3 = 9 次上游调用），并且内层尝试会从审计日志中隐藏；
        # 因此默认客户端重试次数为零。
        self.client = client or LLMClient(max_retries=0)
        self.version = f"deepseek-live-demo:{self.client.model}"
        self._guard = ComplianceGuard()

    def normalize(self, question: str) -> dict[str, Any]:
        return {
            "normalized_query": " ".join(question.strip().split()),
            "normalizer_version": "demo-whitespace-v1",
        }

    def retrieve(self, normalized_query: str) -> dict[str, Any]:
        chunk_id = "live_demo_" + hashlib.sha256(
            normalized_query.encode("utf-8")
        ).hexdigest()[:12]
        return {
            "query": normalized_query,
            "top_k": 1,
            "config_hash": "live-demo-local-fixture-v1",
            "chunks": [
                {
                    "chunk_id": chunk_id,
                    "source": "local_demo_fixture",
                    "source_id": "local-demo-v1",
                    "evidence_level": "demo_fixture_not_for_evaluation",
                    "score": 1.0,
                    "text": (
                        "This local demonstration fixture does not establish a diagnosis. "
                        "Use it only to draft a cautious evidence-bound summary and request "
                        "qualified clinician review."
                    ),
                }
            ],
        }

    def plan(self, normalized_query: str, retrieval: dict[str, Any]) -> dict[str, Any]:
        return {
            "objective": "draft a cautious, non-diagnostic summary strictly bound to supplied evidence",
            "evidence_ids": [item["chunk_id"] for item in retrieval["chunks"]],
            "requires_uncertainty": True,
        }

    def generate(
        self, question: str, retrieval: dict[str, Any], plan: dict[str, Any]
    ) -> ProviderResponse:
        evidence = retrieval.get("chunks", [])
        allowed_ids = {str(item.get("chunk_id")) for item in evidence}
        if not allowed_ids:
            raise MediDiagError("LLM_JSON_INVALID", detail="live demo has no evidence ids")

        completion = self.client.complete(
            self._generation_prompt(question, evidence, plan),
            system_prompt=(
                "You are the constrained drafting component of an engineering demo. "
                "The system is not a doctor and must not diagnose, prescribe, triage, or "
                "make certainty claims. Use only the supplied evidence. Return JSON only; "
                "do not use Markdown fences."
            ),
            temperature=0.0,
            max_tokens=900,
        )
        raw = self.client.parse_json_object(completion.content)
        try:
            draft = _DraftPayload.model_validate(raw)
        except ValidationError as exc:
            raise MediDiagError(
                "LLM_JSON_INVALID",
                detail="live DeepSeek draft does not match the required demonstration schema",
            ) from exc

        claims: list[dict[str, Any]] = []
        for index, claim in enumerate(draft.claims, start=1):
            citation_ids = [chunk_id for chunk_id in claim.citation_chunk_ids if chunk_id in allowed_ids]
            if not citation_ids:
                raise MediDiagError(
                    "LLM_JSON_INVALID",
                    detail="live DeepSeek draft cited an unknown evidence chunk",
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
                        "model": completion.model,
                    }
                ],
                "claims": claims,
                "risk_flags": list(dict.fromkeys(draft.risk_flags))
                + ["qualified_clinician_review_required"],
                "uncertainty": draft.uncertainty.strip(),
            },
            request_id=completion.request_id,
            metadata={
                "generation_model": completion.model,
                "usage": dict(completion.usage),
                "prompt_cache_hit_tokens": completion.usage.get("prompt_cache_hit_tokens", 0),
                "prompt_cache_miss_tokens": completion.usage.get("prompt_cache_miss_tokens", 0),
                "prompt_cache_hit_rate_ppm": completion.usage.get("prompt_cache_hit_rate", 0),
            },
        )

    def arbitrate(self, generation: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
        return {
            "verdict": "SINGLE_DRAFTER_LIMITED_REVIEW",
            "selected_claim_ids": [item["claim_id"] for item in generation["claims"]],
            "conflicts": [],
            "limitation": (
                "Minimal live demo uses one constrained drafting call; it is not a "
                "dual-specialist or formal evidence adjudication result."
            ),
        }

    def review(
        self,
        generation: dict[str, Any],
        arbitration: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        del arbitration, retrieval
        compliance = self._guard.check_output(generation)
        if compliance.blocked:
            return {
                "verdict": "ESCALATED",
                "issues": compliance.block_reasons,
                "citation_verdicts": [],
                "compliance_status": ComplianceStatus.BLOCKED.value,
            }

        citation_verdicts = [
            {
                "claim_id": claim["claim_id"],
                "chunk_id": chunk_id,
                "verdict": "PARTIAL",
                "confidence": None,
                "method": "demo_structure_binding_not_nli",
            }
            for claim in generation["claims"]
            for chunk_id in claim["citation_chunk_ids"]
        ]
        return {
            "verdict": "APPROVED",
            "issues": [
                "Citation verdict is structural binding only; no fixed NLI judge ran in the live demo."
            ],
            "citation_verdicts": citation_verdicts,
            "compliance_status": ComplianceStatus.PASS_WITH_DEMO_LIMITATION.value,
        }

    def report(self, case_id: str, generation: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "assistant-report-v1",
            "case_id": case_id,
            "title": "历史 DeepSeek 演示草稿",
            "summary": " ".join(item["text"] for item in generation["claims"]),
            "claims": generation["claims"],
            "filtered_claim_count": 0,
            "risk_warnings": ["历史结构绑定演示不能用于真实医疗决策。"],
            "limitations": [
                "Live model output is constrained to a local demo fixture, not a medical knowledge base.",
                "Citation verdicts are structural-only in this demo and do not represent NLI validation.",
                "Requires review by a qualified clinician.",
            ],
            "next_steps": ["如有真实健康问题，请咨询具备资质的医疗专业人员。"],
            "disclaimer": MANDATORY_DISCLAIMER,
            "provenance": {"legacy_structural_demo": True},
        }

    @staticmethod
    def _generation_prompt(
        question: str, evidence: list[dict[str, Any]], plan: dict[str, Any]
    ) -> str:
        evidence_text = "\n".join(
            f"- id={item['chunk_id']}; source={item.get('source', 'unknown')}; text={item['text']}"
            for item in evidence
        )
        allowed_ids = [item["chunk_id"] for item in evidence]
        return f"""MediDiag live generation prompt v2. Follow the fixed output contract below.

Draft a cautious Chinese evidence-bound summary for this deidentified/public demo input.
Do not infer facts not present in the supplied evidence. Keep every claim citation-bound.

Input question:
{question}

Plan:
{plan['objective']}

Allowed evidence:
{evidence_text}

Return exactly one JSON object with this schema:
{{
  "claims": [{{"text": "cautious non-diagnostic statement", "citation_chunk_ids": ["one allowed id"], "confidence": 0.0}}],
  "uncertainty": "state the limitations and need for qualified clinician review",
  "risk_flags": ["optional short risk flag"]
}}

Rules: produce 1-3 claims; every claim must use one or more IDs from {allowed_ids}; do not use any other citation ID; do not state a diagnosis, treatment, dosage, emergency triage instruction, or certainty claim; keep all text concise."""
