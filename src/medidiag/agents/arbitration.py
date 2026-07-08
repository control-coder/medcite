"""仲裁 Agent：比较双专科输出，判定流程走向。

接收两份专科结构化输出，自动识别一致/冲突诊断，
对比证据支撑强度，输出冲突明细、最终鉴别列表，
判定流程走向: APPROVED / REVISION_REQUIRED / ESCALATED。

规则驱动（不需要 LLM），确保可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medidiag.agents.base import AgentOutput, DiagnosisItem
from medidiag.agents.llm_client import LLMClient


@dataclass
class ArbitrationResult:
    """仲裁结果。"""

    consensus: list[str] = field(default_factory=list)
    """一致诊断列表（两方都给出的诊断）。"""

    conflicts: list[dict[str, Any]] = field(default_factory=list)
    """冲突明细: [{type, diagnoses, specialty}]。"""

    final_diagnosis: list[DiagnosisItem] = field(default_factory=list)
    """最终鉴别诊断列表（两方并集，按概率排序）。"""

    verdict: str = "REVISION_REQUIRED"
    """流程走向: APPROVED / REVISION_REQUIRED / ESCALATED。"""

    reason: str = ""
    """判定理由。"""

    evidence_comparison: dict[str, Any] = field(default_factory=dict)
    """证据支撑强度对比。"""

    def to_dict(self) -> dict:
        """转为字典。"""
        return {
            "consensus": self.consensus,
            "conflicts": self.conflicts,
            "final_diagnosis": [
                {
                    "diagnosis": d.diagnosis,
                    "probability": d.probability,
                    "supporting_claim_indices": d.supporting_claim_indices,
                }
                for d in self.final_diagnosis
            ],
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence_comparison": self.evidence_comparison,
        }


class ArbitrationAgent:
    """仲裁 Agent。

    规则驱动，不依赖 LLM，确保可复现。

    判定逻辑:
        1. 双方均弃权 → ESCALATED
        2. 一方弃权 → REVISION_REQUIRED
        3. 有一致诊断 → APPROVED
        4. 有冲突但无一致 → 检查证据支撑
           - 双方均无引用 → ESCALATED
           - 否则 → REVISION_REQUIRED
        5. 无诊断输出 → REVISION_REQUIRED
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client

    def arbitrate(
        self,
        output1: AgentOutput,
        output2: AgentOutput,
    ) -> ArbitrationResult:
        """仲裁两份专科输出。

        Args:
            output1: 第一份专科输出。
            output2: 第二份专科输出。

        Returns:
            ArbitrationResult。
        """
        # 1. 识别一致诊断（两方都给出的）
        diag1 = {d.diagnosis.lower() for d in output1.differential_diagnosis}
        diag2 = {d.diagnosis.lower() for d in output2.differential_diagnosis}
        consensus = sorted(diag1 & diag2)

        # 2. 识别冲突
        conflicts: list[dict[str, Any]] = []
        only1 = diag1 - diag2
        only2 = diag2 - diag1
        if only1:
            conflicts.append(
                {
                    "type": "only_in_specialist_1",
                    "diagnoses": sorted(only1),
                    "specialty": output1.specialty,
                }
            )
        if only2:
            conflicts.append(
                {
                    "type": "only_in_specialist_2",
                    "diagnoses": sorted(only2),
                    "specialty": output2.specialty,
                }
            )

        # 3. 对比证据支撑强度
        claims1_total = len(output1.claims)
        claims1_with_citation = sum(
            1 for c in output1.claims if c.citation_chunk_ids
        )
        claims2_total = len(output2.claims)
        claims2_with_citation = sum(
            1 for c in output2.claims if c.citation_chunk_ids
        )

        evidence_comparison = {
            "specialist_1": {
                "specialty": output1.specialty,
                "total_claims": claims1_total,
                "claims_with_citation": claims1_with_citation,
                "citation_rate": (
                    claims1_with_citation / claims1_total
                    if claims1_total > 0
                    else 0.0
                ),
            },
            "specialist_2": {
                "specialty": output2.specialty,
                "total_claims": claims2_total,
                "claims_with_citation": claims2_with_citation,
                "citation_rate": (
                    claims2_with_citation / claims2_total
                    if claims2_total > 0
                    else 0.0
                ),
            },
        }

        # 4. 判定流程走向
        if output1.abstain and output2.abstain:
            verdict = "ESCALATED"
            reason = "双专科均弃权，无法形成诊断建议"
        elif output1.abstain or output2.abstain:
            abstain_side = (
                output1.specialty if output1.abstain else output2.specialty
            )
            verdict = "REVISION_REQUIRED"
            reason = f"{abstain_side} 弃权，需补充诊断"
        elif consensus:
            verdict = "APPROVED"
            reason = f"双专科一致诊断: {consensus}"
        elif conflicts:
            if claims1_with_citation == 0 and claims2_with_citation == 0:
                verdict = "ESCALATED"
                reason = "诊断冲突且双方均无引用支撑，需人工介入"
            else:
                verdict = "REVISION_REQUIRED"
                reason = (
                    f"诊断冲突: specialist_1独有={sorted(only1)}, "
                    f"specialist_2独有={sorted(only2)}"
                )
        else:
            verdict = "REVISION_REQUIRED"
            reason = "双方均无诊断输出"

        # 5. 合并最终诊断（两方并集，按概率排序）
        all_diagnosis = list(output1.differential_diagnosis) + list(
            output2.differential_diagnosis
        )
        all_diagnosis.sort(key=lambda d: d.probability, reverse=True)

        return ArbitrationResult(
            consensus=consensus,
            conflicts=conflicts,
            final_diagnosis=all_diagnosis[:5],
            verdict=verdict,
            reason=reason,
            evidence_comparison=evidence_comparison,
        )
