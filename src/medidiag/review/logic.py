"""诊断逻辑审核模块。

检查诊断建议是否包含:
- 证据（每个 claim 有 citation）
- 风险提示
- 检查建议
- 不确定性说明

规则检查为主。审核驳回闭环：连续失败超过阈值进入人工升级，不进死状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medidiag.agents.base import AgentOutput


@dataclass
class ReviewResult:
    """审核结果。"""

    verdict: str = "APPROVED"
    """APPROVED / REVISION_REQUIRED。"""

    issues: list[str] = field(default_factory=list)
    """发现的问题列表。"""

    details: dict[str, Any] = field(default_factory=dict)
    """审核详情。"""

    @property
    def is_approved(self) -> bool:
        return self.verdict == "APPROVED"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "issues": self.issues,
            "details": self.details,
        }


# 连续失败阈值（超过则进入人工升级）
MAX_CONSECUTIVE_FAILURES = 3


class ClinicalLogicReviewer:
    """诊断逻辑审核器。

    规则检查为主，不依赖 LLM。
    检查项:
        1. 是否有诊断输出
        2. 每个 claim 是否有 citation
        3. 是否有风险提示
        4. 是否有建议检查
        5. 是否有不确定性说明
        6. 未支撑 claim 比例是否超阈值
    """

    def __init__(
        self,
        unsupported_claim_threshold: float = 0.20,
    ) -> None:
        self.unsupported_claim_threshold = unsupported_claim_threshold

    def check(self, output: AgentOutput) -> ReviewResult:
        """审核 Agent 输出。

        Args:
            output: Agent 统一输出。

        Returns:
            ReviewResult。
        """
        issues: list[str] = []
        details: dict[str, Any] = {}

        # 1. 弃权检查
        if output.abstain:
            issues.append(f"agent abstained: {output.abstain_reason}")
            details["abstain"] = True
            return ReviewResult(
                verdict="REVISION_REQUIRED",
                issues=issues,
                details=details,
            )

        # 2. 诊断输出检查
        if not output.differential_diagnosis:
            issues.append("no differential diagnosis provided")
        else:
            details["diagnosis_count"] = len(output.differential_diagnosis)

        # 3. Claim 引用检查
        claims_without_citation = sum(
            1 for c in output.claims if not c.citation_chunk_ids
        )
        total_claims = len(output.claims)
        details["total_claims"] = total_claims
        details["claims_without_citation"] = claims_without_citation

        if total_claims == 0:
            issues.append("no claims provided")
        elif claims_without_citation > 0:
            issues.append(
                f"{claims_without_citation}/{total_claims} claims without citation"
            )

        # 4. 风险提示检查
        if not output.risk_flags:
            issues.append("no risk flags provided")
        else:
            details["risk_flag_count"] = len(output.risk_flags)

        # 5. 建议检查检查
        if not output.recommended_tests:
            issues.append("no recommended tests provided")
        else:
            details["recommended_test_count"] = len(output.recommended_tests)

        # 6. 不确定性说明检查
        if not output.uncertainty:
            issues.append("no uncertainty statement provided")

        # 判定
        verdict = "APPROVED" if not issues else "REVISION_REQUIRED"

        return ReviewResult(
            verdict=verdict,
            issues=issues,
            details=details,
        )

    @staticmethod
    def should_escalate(
        consecutive_failures: int,
        max_rounds: int = MAX_CONSECUTIVE_FAILURES,
    ) -> bool:
        """判断是否应该升级人工。

        连续失败达到 max_rounds 进入人工升级，不进死状态。
        SingleMachineWorker 用运行时配置的 max_review_rounds 调用本函数，
        使复核轮次上限只有这一处判定逻辑。

        Args:
            consecutive_failures: 连续审核失败次数。
            max_rounds: 上限轮次，默认为 MAX_CONSECUTIVE_FAILURES。

        Returns:
            True 如果应该升级。
        """
        return consecutive_failures >= max_rounds
