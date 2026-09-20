"""面向医疗诊断场景的状态机执行器。

状态语义（PLAN.md）：
    CREATED -> NORMALIZED -> EVIDENCE_RETRIEVED -> PLAN_GENERATED
      -> SPECIALIST_REVIEWING -> ARBITRATION_REVIEWING
      -> APPROVED -> REPORT_GENERATED -> CLOSED_SUCCESS

    ESCALATED 是中间等待态（非终态），人工回流只能流向：
      REVISION_REQUIRED / APPROVED / CLOSED_ESCALATED

    终态：CLOSED_SUCCESS / CLOSED_CANCELLED / CLOSED_ESCALATED / CLOSED_FAILED

权限边界：
    每个状态转移必须标明触发主体（TriggerSubject）。
    不同触发主体对同一状态有不同的合法后继。
    ESCALATED 只能由 HUMAN 回流（PLAN.md 要求）。
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


# ===== 合法跳转表 =====
# 结构: {当前状态: {触发主体: [允许的后继状态]}}
# 依据 PLAN.md 状态转移表

TRANSITIONS: dict[CaseState, dict[TriggerSubject, list[CaseState]]] = {
    CaseState.CREATED: {
        TriggerSubject.API: [CaseState.NORMALIZED, CaseState.CLOSED_CANCELLED],
        # ESCALATED: normalize 阶段 provider 失败后不能留在 CREATED 死等。
        TriggerSubject.WORKER: [CaseState.NORMALIZED, CaseState.ESCALATED],
    },
    CaseState.NORMALIZED: {
        TriggerSubject.WORKER: [CaseState.EVIDENCE_RETRIEVED, CaseState.ESCALATED],
    },
    CaseState.EVIDENCE_RETRIEVED: {
        TriggerSubject.WORKER: [CaseState.PLAN_GENERATED, CaseState.ESCALATED],
    },
    CaseState.PLAN_GENERATED: {
        TriggerSubject.AGENT_WORKER: [
            CaseState.SPECIALIST_REVIEWING,
            CaseState.REVISION_REQUIRED,
            CaseState.ESCALATED,
        ],
    },
    CaseState.SPECIALIST_REVIEWING: {
        TriggerSubject.AGENT_WORKER: [
            CaseState.ARBITRATION_REVIEWING,
            CaseState.REVISION_REQUIRED,
            CaseState.ESCALATED,
        ],
    },
    CaseState.ARBITRATION_REVIEWING: {
        TriggerSubject.REVIEWER_WORKER: [
            CaseState.APPROVED,
            CaseState.REVISION_REQUIRED,
            CaseState.ESCALATED,
        ],
    },
    CaseState.REVISION_REQUIRED: {
        TriggerSubject.REVIEWER_WORKER: [
            CaseState.PLAN_GENERATED,
            CaseState.ESCALATED,
            CaseState.CLOSED_FAILED,
        ],
        TriggerSubject.HUMAN: [
            CaseState.PLAN_GENERATED,
            CaseState.ESCALATED,
            CaseState.CLOSED_FAILED,
        ],
    },
    CaseState.APPROVED: {
        # ESCALATED: report 阶段 provider 失败后不能留在 APPROVED 死等。
        TriggerSubject.REVIEWER_WORKER: [
            CaseState.REPORT_GENERATED,
            CaseState.ESCALATED,
        ],
        TriggerSubject.HUMAN: [CaseState.REPORT_GENERATED],
    },
    CaseState.REPORT_GENERATED: {
        TriggerSubject.WORKER: [CaseState.CLOSED_SUCCESS],
    },
    CaseState.ESCALATED: {
        # ESCALATED 只能由 HUMAN 回流（PLAN.md 要求）
        TriggerSubject.HUMAN: [
            CaseState.REVISION_REQUIRED,
            CaseState.APPROVED,
            CaseState.CLOSED_ESCALATED,
        ],
    },
    # 终态：无后继
    CaseState.CLOSED_SUCCESS: {},
    CaseState.CLOSED_CANCELLED: {},
    CaseState.CLOSED_ESCALATED: {},
    CaseState.CLOSED_FAILED: {},
}


# 用户可取消尚未结束的自动处理；人工升级仍只能由 HUMAN 回流，不能绕过。
for _state, _rules in TRANSITIONS.items():
    if _state not in TERMINAL_STATES and _state != CaseState.ESCALATED:
        _api_targets = _rules.setdefault(TriggerSubject.API, [])
        if CaseState.CLOSED_CANCELLED not in _api_targets:
            _api_targets.append(CaseState.CLOSED_CANCELLED)


# ===== 异常 =====


class IllegalTransitionError(Exception):
    """非法状态跳转。

    属性:
        from_state: 起始状态
        to_state: 目标状态
        subject: 触发主体
        reason: 失败原因
    """

    def __init__(
        self,
        from_state: CaseState,
        to_state: CaseState,
        subject: TriggerSubject,
        reason: str = "",
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.subject = subject
        self.reason = reason
        msg = (
            f"非法跳转: {from_state.value} -> {to_state.value} "
            f"(by {subject.value})"
        )
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


# ===== 校验函数 =====


def validate_transition(
    from_state: CaseState,
    to_state: CaseState,
    subject: TriggerSubject,
) -> None:
    """校验状态跳转是否合法。非法则抛 IllegalTransitionError。

    规则:
        1. 终态不可变更
        2. from == to 视为无操作，允许（幂等更新）
        3. 触发主体必须有权限（在 TRANSITIONS[from][subject] 中）
        4. 目标状态必须在合法后继列表中

    Raises:
        IllegalTransitionError: 跳转非法。
    """
    # 规则 1: 终态不可变更
    if is_terminal(from_state):
        raise IllegalTransitionError(
            from_state, to_state, subject,
            reason=f"终态 {from_state.value} 不可变更",
        )

    # 规则 2: 无操作（幂等）
    if from_state == to_state:
        return

    # 规则 3 + 4: 触发主体权限与目标状态校验
    subject_map = TRANSITIONS.get(from_state, {})
    allowed = subject_map.get(subject, [])

    if not allowed:
        raise IllegalTransitionError(
            from_state, to_state, subject,
            reason=f"{from_state.value} 不允许由 {subject.value} 触发跳转",
        )

    if to_state not in allowed:
        raise IllegalTransitionError(
            from_state, to_state, subject,
            reason=(
                f"{subject.value} 无权将 {from_state.value} 跳转到 "
                f"{to_state.value}（合法后继: {[s.value for s in allowed]}）"
            ),
        )


def can_transition(
    from_state: CaseState,
    to_state: CaseState,
    subject: TriggerSubject,
) -> bool:
    """判断跳转是否合法（不抛异常）。"""
    try:
        validate_transition(from_state, to_state, subject)
        return True
    except IllegalTransitionError:
        return False


def get_allowed_transitions(
    state: CaseState,
    subject: TriggerSubject,
) -> list[CaseState]:
    """返回指定状态下、指定触发主体的合法后继列表。"""
    return list(TRANSITIONS.get(state, {}).get(subject, []))


def get_all_allowed_transitions(state: CaseState) -> list[CaseState]:
    """返回指定状态下、所有触发主体的合法后继（去重）。"""
    seen: set[CaseState] = set()
    result: list[CaseState] = []
    for subject_list in TRANSITIONS.get(state, {}).values():
        for s in subject_list:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result
