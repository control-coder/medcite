"""运行时 citation、逻辑、合规审核与医疗助手报告编排。"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medidiag.compliance.guard import MANDATORY_DISCLAIMER, ComplianceGuard
from medidiag.compliance.status import ComplianceStatus
from medidiag.errors import MediDiagError
from medidiag.review.citation import (
    CitationResult,
    CitationVerdict,
    CitationVerifier,
    JudgeInferenceError,
    JudgeInitializationError,
    JudgeInputLanguageError,
)

_HAN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")
_VERDICT_PRIORITY = {
    CitationVerdict.UNSUPPORTED: 0,
    CitationVerdict.PARTIAL: 1,
    CitationVerdict.SUPPORTED: 2,
}


class CanonicalClaim(BaseModel):
    """直接送入固定英文 NLI judge 的规范 claim。"""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=8, max_length=1200)
    citation_chunk_ids: list[str] = Field(min_length=1, max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
    specialty: str = Field(default="", max_length=64)
    agent_run_id: str = Field(default="", max_length=128)

    @model_validator(mode="after")
    def validate_language(self) -> CanonicalClaim:
        if _HAN_PATTERN.search(self.text) or not _LATIN_PATTERN.search(self.text):
            raise ValueError("canonical claim 必须是用于固定 NLI judge 的英文陈述")
        self.citation_chunk_ids = list(dict.fromkeys(self.citation_chunk_ids))
        return self


class ClaimDecision(BaseModel):
    """claim 级审核结论；只有 SUPPORTED 可进入普通报告。"""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    citation_chunk_ids: list[str]
    best_verdict: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    eligible_for_report: bool
    specialty: str = ""
    agent_run_id: str = ""


class AssistantReportClaim(BaseModel):
    """用户报告中保留的原始英文 claim 与可追溯引用。"""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    canonical_text_en: str
    citation_verdict: Literal["SUPPORTED"]
    citations: list[dict[str, Any]] = Field(min_length=1)
    specialty: str = ""


class AssistantReport(BaseModel):
    """医疗助手强类型用户报告；中文只使用固定模板，不改写医学 claim。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["assistant-report-v1"] = "assistant-report-v1"
    case_id: str = Field(min_length=1)
    title: str = "医疗信息辅助报告（工程演示）"
    summary: str
    claims: list[AssistantReportClaim]
    filtered_claim_count: int = Field(ge=0)
    risk_warnings: list[str] = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)
    disclaimer: str = MANDATORY_DISCLAIMER
    provenance: dict[str, Any]


class RuntimeMedicalReview:
    """把固定 NLI、逻辑规则、合规管控和报告过滤收敛为同一运行时边界。"""

    review_schema_version = "medical-review-v1"

    def __init__(
        self,
        verifier: CitationVerifier,
        *,
        guard: ComplianceGuard | None = None,
    ) -> None:
        if verifier.method != "nli":
            raise ValueError("RuntimeMedicalReview 必须使用固定 NLI judge，禁止 rule fallback")
        if not verifier.model_name or not verifier.model_revision:
            raise ValueError("固定 NLI judge 必须提供 model 与 immutable revision")
        self.verifier = verifier
        self.guard = guard or ComplianceGuard()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RuntimeMedicalReview:
        judge = config.get("judge", {})
        if judge.get("method") != "nli":
            raise MediDiagError(
                "CITATION_REVIEW_INVALID",
                detail="医疗助手运行时要求 judge.method=nli，禁止结构绑定或规则 fallback",
            )
        runtime = config.get("runtime", {})
        verifier = CitationVerifier(
            model_name=str(judge.get("model", "")),
            model_revision=str(judge.get("revision", "")),
            method="nli",
            device=str(runtime.get("device", "auto")),
            batch_size=int(runtime.get("batch_size", 32)),
            input_language=str(judge.get("input_language", "en")),
        )
        return cls(verifier)

    def review(
        self,
        generation: dict[str, Any],
        arbitration: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> dict[str, Any]:
        claims = self._canonical_claims(generation, arbitration)
        evidence = self._evidence(retrieval)
        try:
            pair_results = self.verifier.verify_batch(
                [claim.model_dump(mode="json") for claim in claims], evidence
            )
        except JudgeInputLanguageError as exc:
            raise MediDiagError("CLAIM_LANGUAGE_INVALID", detail=str(exc)) from exc
        except (JudgeInitializationError, JudgeInferenceError) as exc:
            raise MediDiagError(
                "JUDGE_UNAVAILABLE",
                detail=f"固定 NLI judge 执行失败: {type(exc).__name__}",
            ) from exc

        decisions = self._claim_decisions(claims, pair_results)
        supported_ids = [item.claim_id for item in decisions if item.eligible_for_report]
        filtered_ids = [item.claim_id for item in decisions if not item.eligible_for_report]
        compliance = self.guard.check_output(
            {
                "claims": [claim.model_dump(mode="json") for claim in claims],
                "risk_flags": list(generation.get("risk_flags", [])),
                "recommended_tests": list(generation.get("recommended_tests", [])),
                "uncertainty": str(generation.get("uncertainty", "")),
            }
        )

        issues: list[str] = []
        if filtered_ids:
            issues.append(
                f"{len(filtered_ids)} 个 PARTIAL/UNSUPPORTED claim 已从普通报告候选中移除"
            )
        if not generation.get("uncertainty"):
            issues.append("缺少不确定性说明")
        if not generation.get("risk_flags"):
            issues.append("缺少风险提示")

        agent_abstained = any(
            bool((artifact.get("output") or {}).get("abstain"))
            for artifact in generation.get("agents", [])
        )
        if compliance.blocked:
            verdict = "ESCALATED"
            issues.extend(compliance.block_reasons)
        elif agent_abstained:
            verdict = "REVISION_REQUIRED"
            issues.append("至少一个 Agent 因证据不足而弃权")
        elif not supported_ids:
            verdict = "REVISION_REQUIRED"
            issues.append("没有通过固定 NLI judge 的 SUPPORTED claim")
        else:
            verdict = "APPROVED"

        citation_verdicts: list[dict[str, Any]] = []
        for item in pair_results:
            citation_payload = item.to_dict()
            # worker 和持久化模型统一使用 chunk_id；保留 judge 模型来源供审计。
            citation_payload["chunk_id"] = citation_payload.pop("evidence_chunk_id")
            citation_verdicts.append(citation_payload)

        return {
            "schema_version": self.review_schema_version,
            "verdict": verdict,
            "issues": list(dict.fromkeys(issues)),
            "canonical_claims": [item.model_dump(mode="json") for item in claims],
            "citation_verdicts": citation_verdicts,
            "claim_decisions": [item.model_dump(mode="json") for item in decisions],
            "approved_claim_ids": supported_ids,
            "filtered_claim_ids": filtered_ids,
            "compliance_status": (
                ComplianceStatus.BLOCKED.value
                if compliance.blocked
                else ComplianceStatus.PASSED_FIXED_NLI.value
            ),
            "compliance": compliance.to_dict(),
            "logic_review": {
                "has_uncertainty": bool(generation.get("uncertainty")),
                "has_risk_flags": bool(generation.get("risk_flags")),
                "supported_claim_count": len(supported_ids),
                "filtered_claim_count": len(filtered_ids),
            },
            "judge": {
                "method": "nli",
                "model": self.verifier.model_name,
                "revision": self.verifier.model_revision,
                "input_language": self.verifier.input_language,
                "device": self.verifier.actual_device,
            },
            "evidence_bundle_id": retrieval.get("evidence_bundle_id"),
            "evidence_hash": generation.get("evidence_hash"),
            "evidence_provenance": self._evidence_provenance(retrieval),
        }

    def report(
        self,
        case_id: str,
        generation: dict[str, Any],
        review: dict[str, Any],
    ) -> dict[str, Any]:
        if review.get("verdict") != "APPROVED":
            raise MediDiagError(
                "REPORT_PROVENANCE_INVALID", detail="只有 APPROVED 审核结果可以生成普通报告"
            )
        decisions = {
            str(item["claim_id"]): ClaimDecision.model_validate(item)
            for item in review.get("claim_decisions", [])
        }
        canonical = {
            str(item["claim_id"]): CanonicalClaim.model_validate(item)
            for item in review.get("canonical_claims", [])
        }
        evidence = {
            str(item["chunk_id"]): dict(item)
            for item in review.get("evidence_provenance", [])
        }
        approved_ids = [str(value) for value in review.get("approved_claim_ids", [])]
        report_claims: list[AssistantReportClaim] = []
        for claim_id in approved_ids:
            decision = decisions.get(claim_id)
            claim = canonical.get(claim_id)
            if (
                decision is None
                or claim is None
                or not decision.eligible_for_report
                or decision.best_verdict != CitationVerdict.SUPPORTED.value
                or decision.text != claim.text
            ):
                raise MediDiagError(
                    "REPORT_PROVENANCE_INVALID",
                    detail=f"报告 claim 与审核 provenance 不一致: {claim_id}",
                )
            citations = [evidence[cid] for cid in claim.citation_chunk_ids if cid in evidence]
            if not citations:
                raise MediDiagError(
                    "REPORT_PROVENANCE_INVALID",
                    detail=f"报告 claim 缺少可追溯证据: {claim_id}",
                )
            report_claims.append(
                AssistantReportClaim(
                    claim_id=claim_id,
                    canonical_text_en=claim.text,
                    citation_verdict="SUPPORTED",
                    citations=citations,
                    specialty=claim.specialty,
                )
            )
        if not report_claims:
            raise MediDiagError(
                "REPORT_PROVENANCE_INVALID", detail="普通报告没有 SUPPORTED claim"
            )

        report = AssistantReport(
            case_id=case_id,
            summary=(
                "系统已完成证据检索、固定 NLI 引用审核与合规检查。"
                "下方仅展示通过审核的英文原始 claim，不进行中文医学改写。"
            ),
            claims=report_claims,
            filtered_claim_count=len(review.get("filtered_claim_ids", [])),
            risk_warnings=[
                "本报告不能用于自行诊断、用药或替代医生面诊。",
                "如症状持续、加重或出现紧急情况，请及时联系当地医疗机构或急救服务。",
            ],
            limitations=[
                "证据来自版本化公开语料，不是实时临床知识服务。",
                "SUPPORTED 仅表示固定 NLI judge 判定 claim 与所引证据相符，不证明临床结论正确。",
                "Provider snapshot 可能不可核验，response ID 仅提供受限溯源。",
            ],
            next_steps=[
                "携带症状经过、既往史和检查资料咨询具备资质的医疗专业人员。",
                "由专业人员结合体格检查、检验和影像结果作进一步判断。",
            ],
            provenance={
                "review_schema_version": review.get("schema_version"),
                "judge": review.get("judge"),
                "evidence_bundle_id": review.get("evidence_bundle_id"),
                "evidence_hash": review.get("evidence_hash"),
                "claim_text_policy": "exact_canonical_english_no_translation",
            },
        )
        payload = report.model_dump(mode="json")
        self._postcheck_report(payload, canonical, set(approved_ids))
        return payload

    @staticmethod
    def _canonical_claims(
        generation: dict[str, Any], arbitration: dict[str, Any]
    ) -> list[CanonicalClaim]:
        raw_claims = list(
            arbitration.get("claims_after_arbitration")
            or generation.get("claims_before_arbitration")
            or generation.get("claims")
            or []
        )
        selected = set(arbitration.get("selected_claim_ids") or [])
        if selected:
            raw_claims = [item for item in raw_claims if item.get("claim_id") in selected]
        try:
            claims = [CanonicalClaim.model_validate(item) for item in raw_claims]
        except Exception as exc:
            raise MediDiagError(
                "CLAIM_LANGUAGE_INVALID",
                detail=f"canonical claim schema/language 校验失败: {type(exc).__name__}",
            ) from exc
        if not claims and not generation.get("all_agents_abstained"):
            raise MediDiagError("CITATION_REVIEW_INVALID", detail="仲裁后没有可审核 claim")
        if len({item.claim_id for item in claims}) != len(claims):
            raise MediDiagError("CITATION_REVIEW_INVALID", detail="仲裁后 claim_id 重复")
        return claims

    @staticmethod
    def _evidence(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
        evidence = retrieval.get("chunks")
        if not isinstance(evidence, list) or not evidence:
            raise MediDiagError("CITATION_REVIEW_INVALID", detail="citation 审核缺少检索证据")
        return [dict(item) for item in evidence]

    @staticmethod
    def _claim_decisions(
        claims: list[CanonicalClaim], results: list[CitationResult]
    ) -> list[ClaimDecision]:
        grouped: dict[str, list[CitationResult]] = defaultdict(list)
        for item in results:
            grouped[item.claim_id].append(item)
        decisions: list[ClaimDecision] = []
        for claim in claims:
            values = grouped.get(claim.claim_id, [])
            best = max(
                (item.verdict for item in values),
                key=lambda verdict: _VERDICT_PRIORITY[verdict],
                default=CitationVerdict.UNSUPPORTED,
            )
            decisions.append(
                ClaimDecision(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    citation_chunk_ids=claim.citation_chunk_ids,
                    best_verdict=best.value,
                    eligible_for_report=best == CitationVerdict.SUPPORTED,
                    specialty=claim.specialty,
                    agent_run_id=claim.agent_run_id,
                )
            )
        return decisions

    @staticmethod
    def _evidence_provenance(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": str(item.get("chunk_id", "")),
                "source": str(item.get("source", "")),
                "source_id": str(item.get("source_id", "")),
                "chunk_hash": str(item.get("chunk_hash", "")),
                "evidence_level": str(item.get("evidence_level", "")),
            }
            for item in retrieval.get("chunks", [])
        ]

    @staticmethod
    def _postcheck_report(
        payload: dict[str, Any],
        canonical: dict[str, CanonicalClaim],
        approved_ids: set[str],
    ) -> None:
        for item in payload.get("claims", []):
            claim_id = str(item.get("claim_id", ""))
            source = canonical.get(claim_id)
            if (
                claim_id not in approved_ids
                or source is None
                or item.get("canonical_text_en") != source.text
                or item.get("citation_verdict") != CitationVerdict.SUPPORTED.value
            ):
                raise MediDiagError(
                    "REPORT_PROVENANCE_INVALID",
                    detail=f"用户报告引入或改写了未审核 claim: {claim_id}",
                )
