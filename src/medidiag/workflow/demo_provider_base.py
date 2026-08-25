"""OpenAI-compatible 医疗助手起草阶段与结构化 schema。"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from medidiag.compliance.guard import MANDATORY_DISCLAIMER, ComplianceGuard
from medidiag.compliance.status import ComplianceStatus


class DraftClaim(BaseModel):
    text: str = Field(min_length=8, max_length=1200)
    citation_chunk_ids: list[str] = Field(min_length=1, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)


class DraftPayload(BaseModel):
    claims: list[DraftClaim] = Field(min_length=1, max_length=4)
    uncertainty: str = Field(min_length=8, max_length=1200)
    risk_flags: list[str] = Field(default_factory=list, max_length=8)


class DemoWorkflowSupport:
    """提供规划、审核与报告骨架；真实检索由子类显式装配。"""

    version: str
    _guard: ComplianceGuard

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
        del normalized_query
        return {
            "objective": "draft a cautious, non-diagnostic summary strictly bound to supplied evidence",
            "evidence_ids": [item["chunk_id"] for item in retrieval["chunks"]],
            "requires_uncertainty": True,
        }

    def arbitrate(
        self,
        generation: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        del retrieval
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
    ) -> dict[str, Any]:
        del arbitration
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

    def report(
        self,
        case_id: str,
        generation: dict[str, Any],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        del review
        return {
            "case_id": case_id,
            "title": "OpenAI-compatible Evidence-Bound Draft",
            "summary": " ".join(item["text"] for item in generation["claims"]),
            "claims": generation["claims"],
            "limitations": [
                "Evidence comes from the versioned runtime corpus; it is not a clinical knowledge service.",
                "Citation verdicts are structural-only in this demo and do not represent NLI validation.",
                "Requires review by a qualified clinician.",
            ],
            "disclaimer": MANDATORY_DISCLAIMER,
        }

    @staticmethod
    def generation_prompt(
        question: str,
        evidence: list[dict[str, Any]],
        plan: dict[str, Any],
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
