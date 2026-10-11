"""阶段 0 冒烟测试。

验证：
1. errors 模块可导入、错误码注册表完整、关键错误码存在
2. state_machine 状态枚举完整（14 状态）、终态判断正确、ESCALATED 非终态
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medidiag.errors import (
    ErrorCategory,
    MediDiagError,
    alertable_codes,
    all_error_codes,
    get_error_spec,
    retryable_codes,
)
from medidiag.workflow.state_machine import (
    TERMINAL_STATES,
    CaseState,
    TriggerSubject,
    is_terminal,
)

_ROOT = Path(__file__).resolve().parent.parent

# ===== errors 模块测试 =====

def test_error_registry_non_empty() -> None:
    codes = all_error_codes()
    assert len(codes) >= 15, f"expected >=15 error codes, got {len(codes)}"


def test_every_registered_code_is_actually_raised() -> None:
    """注册表只描述真实存在的行为。

    描述了未实现 `default_action` 的错误码比更小的注册表更糟：它承诺了一条
    实际不存在的处理路径。本测试扫描 src/ 与 eval/，任何只存在于注册表中的
    错误码都会失败。
    """
    roots = [_ROOT / "src", _ROOT / "eval"]
    sources = [
        path.read_text(encoding="utf-8")
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in str(path) and path.name != "errors.py"
    ]
    unreferenced = [
        code
        for code in all_error_codes()
        if not any(code in text for text in sources)
    ]
    assert unreferenced == [], f"registered but never raised: {unreferenced}"


@pytest.mark.parametrize("expected_code", [
    "CASE_INVALID_INPUT",
    "CASE_NOT_FOUND",
    "CASE_ALREADY_CLOSED",
    "IDEMPOTENCY_KEY_MISSING",
    "CASE_INPUT_NOT_DEIDENTIFIED",
    "STATE_CONFLICT",
    "WORKFLOW_ALREADY_RUNNING",
    "ILLEGAL_STATE_TRANSITION",
    "MAX_REVIEW_ROUNDS_EXCEEDED",
    "REPORT_NOT_READY",
    "RAG_TIMEOUT",
    "LLM_TIMEOUT",
    "JUDGE_TIMEOUT",
    "COMPLIANCE_BLOCKED",
    "EVAL_DATA_LEAKAGE_DETECTED",
    "TASK_LEASE_EXPIRED",
    "TASK_LEASE_LOST",
    "OPTIMISTIC_LOCK_CONFLICT",
    "WORKFLOW_RETRY_EXCEEDED",
])
def test_key_error_codes_exist(expected_code: str) -> None:
    codes = all_error_codes()
    assert expected_code in codes, f"missing error code: {expected_code}"


def test_error_spec_fields_complete() -> None:
    spec = get_error_spec("STATE_CONFLICT")
    assert spec.code == "STATE_CONFLICT"
    assert spec.http_status == 409
    assert spec.category == ErrorCategory.BUSINESS
    assert spec.retryable is True
    assert spec.requires_human_escalation is False
    assert spec.alert is True
    assert spec.description
    assert spec.default_action


def test_error_category_distribution() -> None:
    """五类错误码都有代表。"""
    codes = all_error_codes()
    categories = {spec.category for spec in codes.values()}
    assert ErrorCategory.USER_INPUT in categories
    assert ErrorCategory.BUSINESS in categories
    assert ErrorCategory.DEPENDENCY in categories
    assert ErrorCategory.DATA_QUALITY in categories
    assert ErrorCategory.SYSTEM in categories


def test_alertable_codes_filter() -> None:
    alertable = alertable_codes()
    assert "STATE_CONFLICT" in alertable
    assert "EVAL_DATA_LEAKAGE_DETECTED" in alertable
    assert "TASK_LEASE_LOST" in alertable
    # 用户输入错误一般不告警
    assert "CASE_INVALID_INPUT" not in alertable


def test_retryable_codes_filter() -> None:
    retryable = retryable_codes()
    assert "RAG_TIMEOUT" in retryable
    assert "LLM_TIMEOUT" in retryable
    assert "OPTIMISTIC_LOCK_CONFLICT" in retryable
    # 不可重试的
    assert "CASE_INVALID_INPUT" not in retryable
    assert "TASK_LEASE_LOST" not in retryable  # 脑裂防护，丢弃而非重试


def test_medidiag_error_carries_spec() -> None:
    with pytest.raises(MediDiagError) as exc_info:
        raise MediDiagError("STATE_CONFLICT", detail="test conflict", context={"case_id": "c1"})
    err = exc_info.value
    assert err.code == "STATE_CONFLICT"
    assert err.spec.http_status == 409
    assert err.detail == "test conflict"
    assert err.context["case_id"] == "c1"


def test_medidiag_error_unknown_code_raises() -> None:
    with pytest.raises(KeyError):
        MediDiagError("NONEXISTENT_CODE")


def test_leakage_error_spec() -> None:
    """数据泄露校验错误码规格正确。"""
    spec = get_error_spec("EVAL_DATA_LEAKAGE_DETECTED")
    assert spec.category == ErrorCategory.DATA_QUALITY
    assert spec.retryable is False
    assert spec.requires_human_escalation is True
    assert spec.alert is True


# ===== state_machine 模块测试 =====

def test_case_state_count_is_14() -> None:
    """PLAN.md 定义 14 个状态。"""
    states = list(CaseState)
    assert len(states) == 14, f"expected 14 states, got {len(states)}: {[s.value for s in states]}"


def test_all_expected_states_present() -> None:
    expected = {
        "CREATED", "NORMALIZED", "EVIDENCE_RETRIEVED", "PLAN_GENERATED",
        "SPECIALIST_REVIEWING", "ARBITRATION_REVIEWING",
        "REVISION_REQUIRED", "APPROVED", "REPORT_GENERATED",
        "ESCALATED",
        "CLOSED_SUCCESS", "CLOSED_CANCELLED", "CLOSED_ESCALATED", "CLOSED_FAILED",
    }
    actual = {s.value for s in CaseState}
    assert actual == expected, f"state mismatch: missing={expected - actual}, extra={actual - expected}"


def test_terminal_states_count_is_4() -> None:
    assert len(TERMINAL_STATES) == 4


def test_terminal_states_set() -> None:
    assert is_terminal(CaseState.CLOSED_SUCCESS)
    assert is_terminal(CaseState.CLOSED_CANCELLED)
    assert is_terminal(CaseState.CLOSED_ESCALATED)
    assert is_terminal(CaseState.CLOSED_FAILED)


def test_non_terminal_states() -> None:
    assert not is_terminal(CaseState.CREATED)
    assert not is_terminal(CaseState.NORMALIZED)
    assert not is_terminal(CaseState.EVIDENCE_RETRIEVED)
    assert not is_terminal(CaseState.PLAN_GENERATED)
    assert not is_terminal(CaseState.SPECIALIST_REVIEWING)
    assert not is_terminal(CaseState.ARBITRATION_REVIEWING)
    assert not is_terminal(CaseState.REVISION_REQUIRED)
    assert not is_terminal(CaseState.APPROVED)
    assert not is_terminal(CaseState.REPORT_GENERATED)


def test_escalated_is_not_terminal() -> None:
    """PLAN.md 核心约束：ESCALATED 是中间等待态，不是终态。"""
    assert not is_terminal(CaseState.ESCALATED)


def test_trigger_subject_count_is_6() -> None:
    """PLAN.md 定义 6 个触发主体：API/worker/Agent worker/reviewer worker/human/system。"""
    subjects = list(TriggerSubject)
    assert len(subjects) == 6
    assert TriggerSubject.API in subjects
    assert TriggerSubject.WORKER in subjects
    assert TriggerSubject.AGENT_WORKER in subjects
    assert TriggerSubject.REVIEWER_WORKER in subjects
    assert TriggerSubject.HUMAN in subjects
    assert TriggerSubject.SYSTEM in subjects
