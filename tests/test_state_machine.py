"""状态机跳转表与校验测试。

覆盖:
1. 主链路合法跳转
2. 非法跳转拦截
3. 终态不可变更
4. ESCALATED 只能由 HUMAN 回流
5. 触发主体权限边界
6. 幂等跳转（from == to）
7. can_transition / get_allowed_transitions 辅助函数
"""

from __future__ import annotations

import pytest

from medidiag.workflow.state_machine import (
    TRANSITIONS,
    TERMINAL_STATES,
    CaseState,
    IllegalTransitionError,
    TriggerSubject,
    can_transition,
    get_all_allowed_transitions,
    get_allowed_transitions,
    is_terminal,
    validate_transition,
)


# ===== 终态测试 =====


class TestTerminalStates:
    def test_four_terminal_states(self) -> None:
        assert len(TERMINAL_STATES) == 4

    def test_terminal_cannot_transition_to_anything(self) -> None:
        """终态不可变更：任何跳转都被拒绝。"""
        for state in TERMINAL_STATES:
            for subject in TriggerSubject:
                with pytest.raises(IllegalTransitionError, match="终态"):
                    validate_transition(state, CaseState.CREATED, subject)

    def test_terminal_same_state_also_rejected(self) -> None:
        """终态即使 from == to 也拒绝（上层应先检查状态）。"""
        for state in TERMINAL_STATES:
            with pytest.raises(IllegalTransitionError, match="终态"):
                validate_transition(state, state, TriggerSubject.SYSTEM)

    def test_escalated_is_not_terminal(self) -> None:
        """ESCALATED 是中间等待态，不是终态。"""
        assert not is_terminal(CaseState.ESCALATED)

    def test_non_terminal_states(self) -> None:
        assert not is_terminal(CaseState.CREATED)
        assert not is_terminal(CaseState.NORMALIZED)
        assert not is_terminal(CaseState.PLAN_GENERATED)
        assert not is_terminal(CaseState.REVISION_REQUIRED)


# ===== 主链路合法跳转测试 =====


class TestLegalMainPath:
    def test_full_happy_path(self) -> None:
        """主链路: CREATED -> ... -> CLOSED_SUCCESS。"""
        steps = [
            (CaseState.CREATED, CaseState.NORMALIZED, TriggerSubject.API),
            (CaseState.NORMALIZED, CaseState.EVIDENCE_RETRIEVED, TriggerSubject.WORKER),
            (CaseState.EVIDENCE_RETRIEVED, CaseState.PLAN_GENERATED, TriggerSubject.WORKER),
            (CaseState.PLAN_GENERATED, CaseState.SPECIALIST_REVIEWING, TriggerSubject.AGENT_WORKER),
            (CaseState.SPECIALIST_REVIEWING, CaseState.ARBITRATION_REVIEWING, TriggerSubject.AGENT_WORKER),
            (CaseState.ARBITRATION_REVIEWING, CaseState.APPROVED, TriggerSubject.REVIEWER_WORKER),
            (CaseState.APPROVED, CaseState.REPORT_GENERATED, TriggerSubject.REVIEWER_WORKER),
            (CaseState.REPORT_GENERATED, CaseState.CLOSED_SUCCESS, TriggerSubject.WORKER),
        ]
        for from_s, to_s, subject in steps:
            validate_transition(from_s, to_s, subject)  # 不抛异常即通过

    def test_created_to_cancelled(self) -> None:
        validate_transition(
            CaseState.CREATED, CaseState.CLOSED_CANCELLED, TriggerSubject.API
        )

    def test_escalation_at_each_stage(self) -> None:
        """每个阶段都可以进入 ESCALATED。"""
        escalatable = [
            (CaseState.NORMALIZED, TriggerSubject.WORKER),
            (CaseState.EVIDENCE_RETRIEVED, TriggerSubject.WORKER),
            (CaseState.PLAN_GENERATED, TriggerSubject.AGENT_WORKER),
            (CaseState.SPECIALIST_REVIEWING, TriggerSubject.AGENT_WORKER),
            (CaseState.ARBITRATION_REVIEWING, TriggerSubject.REVIEWER_WORKER),
        ]
        for state, subject in escalatable:
            validate_transition(state, CaseState.ESCALATED, subject)


# ===== ESCALATED 人工回流测试 =====


class TestEscalatedReturn:
    def test_human_can_return_to_revision(self) -> None:
        validate_transition(
            CaseState.ESCALATED,
            CaseState.REVISION_REQUIRED,
            TriggerSubject.HUMAN,
        )

    def test_human_can_approve(self) -> None:
        validate_transition(
            CaseState.ESCALATED, CaseState.APPROVED, TriggerSubject.HUMAN
        )

    def test_human_can_close_escalated(self) -> None:
        validate_transition(
            CaseState.ESCALATED,
            CaseState.CLOSED_ESCALATED,
            TriggerSubject.HUMAN,
        )

    def test_non_human_cannot_return_from_escalated(self) -> None:
        """ESCALATED 只能由 HUMAN 回流。"""
        non_human = [
            TriggerSubject.API,
            TriggerSubject.WORKER,
            TriggerSubject.AGENT_WORKER,
            TriggerSubject.REVIEWER_WORKER,
            TriggerSubject.SYSTEM,
        ]
        for subject in non_human:
            with pytest.raises(IllegalTransitionError):
                validate_transition(
                    CaseState.ESCALATED,
                    CaseState.REVISION_REQUIRED,
                    subject,
                )


# ===== 非法跳转测试 =====


class TestIllegalTransitions:
    def test_wrong_subject_for_normalization(self) -> None:
        """API 不能做 NORMALIZED -> EVIDENCE_RETRIEVED（只有 WORKER 可以）。"""
        with pytest.raises(IllegalTransitionError):
            validate_transition(
                CaseState.NORMALIZED,
                CaseState.EVIDENCE_RETRIEVED,
                TriggerSubject.API,
            )

    def test_skip_states(self) -> None:
        """不能跳过中间状态（CREATED 不能直接到 PLAN_GENERATED）。"""
        with pytest.raises(IllegalTransitionError):
            validate_transition(
                CaseState.CREATED,
                CaseState.PLAN_GENERATED,
                TriggerSubject.API,
            )

    def test_worker_cannot_do_review(self) -> None:
        """WORKER 不能执行审核跳转。"""
        with pytest.raises(IllegalTransitionError):
            validate_transition(
                CaseState.ARBITRATION_REVIEWING,
                CaseState.APPROVED,
                TriggerSubject.WORKER,
            )

    def test_api_cannot_cancel_after_created(self) -> None:
        """CLOSED_CANCELLED 只能从 CREATED 触发。"""
        with pytest.raises(IllegalTransitionError):
            validate_transition(
                CaseState.NORMALIZED,
                CaseState.CLOSED_CANCELLED,
                TriggerSubject.API,
            )

    def test_revision_required_wrong_subject(self) -> None:
        """REVISION_REQUIRED -> PLAN_GENERATED 只能由 REVIEWER_WORKER 或 HUMAN。"""
        with pytest.raises(IllegalTransitionError):
            validate_transition(
                CaseState.REVISION_REQUIRED,
                CaseState.PLAN_GENERATED,
                TriggerSubject.API,
            )

    def test_backwards_transition_blocked(self) -> None:
        """不能回退状态（如 PLAN_GENERATED 不能回到 EVIDENCE_RETRIEVED）。"""
        with pytest.raises(IllegalTransitionError):
            validate_transition(
                CaseState.PLAN_GENERATED,
                CaseState.EVIDENCE_RETRIEVED,
                TriggerSubject.WORKER,
            )


# ===== 幂等跳转测试 =====


class TestIdempotentTransition:
    def test_same_state_non_terminal_allowed(self) -> None:
        """非终态 from == to 视为无操作，允许。"""
        validate_transition(
            CaseState.CREATED, CaseState.CREATED, TriggerSubject.API
        )
        validate_transition(
            CaseState.NORMALIZED, CaseState.NORMALIZED, TriggerSubject.WORKER
        )

    def test_same_state_with_wrong_subject_still_allowed(self) -> None:
        """from == to 时即使 subject 无权限也允许（幂等更新）。"""
        validate_transition(
            CaseState.CREATED, CaseState.CREATED, TriggerSubject.SYSTEM
        )


# ===== 辅助函数测试 =====


class TestCanTransition:
    def test_legal_returns_true(self) -> None:
        assert can_transition(
            CaseState.CREATED, CaseState.NORMALIZED, TriggerSubject.API
        )

    def test_illegal_returns_false(self) -> None:
        assert not can_transition(
            CaseState.CREATED, CaseState.PLAN_GENERATED, TriggerSubject.API
        )
        assert not can_transition(
            CaseState.CLOSED_SUCCESS, CaseState.CREATED, TriggerSubject.SYSTEM
        )


class TestGetAllowed:
    def test_get_allowed_transitions(self) -> None:
        allowed = get_allowed_transitions(CaseState.CREATED, TriggerSubject.API)
        assert CaseState.NORMALIZED in allowed
        assert CaseState.CLOSED_CANCELLED in allowed
        assert len(allowed) == 2

    def test_get_allowed_empty_for_terminal(self) -> None:
        allowed = get_allowed_transitions(
            CaseState.CLOSED_SUCCESS, TriggerSubject.SYSTEM
        )
        assert allowed == []

    def test_get_all_allowed_transitions(self) -> None:
        all_allowed = get_all_allowed_transitions(CaseState.CREATED)
        assert CaseState.NORMALIZED in all_allowed
        assert CaseState.CLOSED_CANCELLED in all_allowed

    def test_transitions_table_covers_all_states(self) -> None:
        """跳转表必须覆盖所有 14 个状态。"""
        for state in CaseState:
            assert state in TRANSITIONS, f"state {state} missing from TRANSITIONS"
