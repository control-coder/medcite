"""面向医疗诊断场景的状态机执行器。

阶段 0：仅定义状态枚举与触发主体。
阶段 2：实现合法跳转表、跳转校验、非法跳转拦截、触发主体权限边界。

状态语义（PLAN.md）：
    CREATED -> NORMALIZED -> EVIDENCE_RETRIEVED -> PLAN_GENERATED
      -> SPECIALIST_REVIEWING -> ARBITRATION_REVIEWING
      -> APPROVED -> REPORT_GENERATED -> CLOSED_SUCCESS

    ESCALATED 是中间等待态（非终态），人工回流只能流向：
      REVISION_REQUIRED / APPROVED / CLOSED_ESCALATED

    终态：CLOSED_SUCCESS / CLOSED_CANCELLED / CLOSED_ESCALATED / CLOSED_FAILED
"""

from __future__ import annotations

from enum import Enum


class CaseState(str, Enum):
    """病例状态枚举。共 14 个状态。"""

    # 主链路
    CREATED = "CREATED"
    NORMALIZED = "NORMALIZED"
    EVIDENCE_RETRIEVED = "EVIDENCE_RETRIEVED"
    PLAN_GENERATED = "PLAN_GENERATED"
    SPECIALIST_REVIEWING = "SPECIALIST_REVIEWING"
    ARBITRATION_REVIEWING = "ARBITRATION_REVIEWING"

    # 审核分支
    REVISION_REQUIRED = "REVISION_REQUIRED"
    APPROVED = "APPROVED"
    REPORT_GENERATED = "REPORT_GENERATED"

    # 等待态（非终态）
    ESCALATED = "ESCALATED"

    # 终态
    CLOSED_SUCCESS = "CLOSED_SUCCESS"
    CLOSED_CANCELLED = "CLOSED_CANCELLED"
    CLOSED_ESCALATED = "CLOSED_ESCALATED"
    CLOSED_FAILED = "CLOSED_FAILED"


class TriggerSubject(str, Enum):
    """状态变更触发主体。

    每个状态转移必须标明触发主体，用于权限边界校验与审计。
    """

    API = "api"                  # API 请求触发
    WORKER = "worker"            # 普通 worker
    AGENT_WORKER = "agent_worker"  # Agent worker（诊断生成、双专科）
    REVIEWER_WORKER = "reviewer_worker"  # reviewer worker（仲裁、审核）
    HUMAN = "human"              # 人工干预
    SYSTEM = "system"            # 系统自动（超时、重试耗尽）


# 终态集合：不可再变更
TERMINAL_STATES: frozenset[CaseState] = frozenset({
    CaseState.CLOSED_SUCCESS,
    CaseState.CLOSED_CANCELLED,
    CaseState.CLOSED_ESCALATED,
    CaseState.CLOSED_FAILED,
})


def is_terminal(state: CaseState) -> bool:
    """判断是否为终态。"""
    return state in TERMINAL_STATES


# TODO[阶段2]: 实现合法跳转表 TRANSITIONS: dict[CaseState, dict[TriggerSubject, list[CaseState]]]
# TODO[阶段2]: 实现 validate_transition(from, to, subject) -> None
# TODO[阶段2]: 实现非法跳转测试（终态不可变更、ESCALATED 人工回流、权限边界）
