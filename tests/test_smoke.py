"""阶段 0 冒烟测试。

验证：
1. errors 模块可导入、错误码注册表完整、关键错误码存在
2. state_machine 状态枚举完整（14 状态）、终态判断正确、ESCALATED 非终态
3. eval/config.yaml 可加载且锁定字段齐全
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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


# ===== errors 模块测试 =====

def test_error_registry_non_empty() -> None:
    codes = all_error_codes()
    assert len(codes) >= 15, f"expected >=15 error codes, got {len(codes)}"


@pytest.mark.parametrize("expected_code", [
    "CASE_INVALID_INPUT",
    "CASE_NOT_FOUND",
    "CASE_ALREADY_CLOSED",
    "CASE_CANCELLED",
    "IDEMPOTENCY_KEY_MISSING",
    "CASE_INPUT_NOT_DEIDENTIFIED",
    "STATE_CONFLICT",
    "WORKFLOW_ALREADY_RUNNING",
    "ILLEGAL_STATE_TRANSITION",
    "MAX_REVIEW_ROUNDS_EXCEEDED",
    "REPORT_NOT_READY",
    "RAG_TIMEOUT",
    "LLM_TIMEOUT",
    "LLM_JSON_INVALID",
    "JUDGE_TIMEOUT",
    "EMBEDDING_TIMEOUT",
    "RAG_EMPTY_RESULT",
    "REVIEW_UNSUPPORTED_CLAIM",
    "CITATION_VERIFICATION_FAILED",
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
    assert "CASE_CANCELLED" not in retryable
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


# ===== eval/config.yaml 测试 =====

@pytest.fixture
def eval_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "eval" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_eval_config_locked_models(eval_config: dict) -> None:
    """验证锁定模型版本。换模型必须重跑全部评测。"""
    assert eval_config["generation"]["model"] == "deepseek-v4-flash-free"
    assert eval_config["embedding"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert eval_config["rerank"]["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert eval_config["judge"]["model"] == "cross-encoder/nli-MiniLM2-L6-H768"
    assert eval_config["embedding"]["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert eval_config["rerank"]["revision"] == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    assert eval_config["judge"]["revision"] == "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"


def test_eval_config_reproducibility_params(eval_config: dict) -> None:
    """温度、seed、数据集版本必须锁定。"""
    assert eval_config["generation"]["temperature"] == 0
    assert eval_config["generation"]["seed"] == 42
    assert eval_config["dataset"]["version"] == "v1"


def test_eval_config_public_ratio(eval_config: dict) -> None:
    """公开题占比必须 >= 40%。"""
    assert eval_config["dataset"]["public_ratio_min"] >= 0.40


def test_eval_config_experiment_namespaces_complete(eval_config: dict) -> None:
    """RAG 与 Agent 实验必须使用互不耦合的命名空间。"""
    assert set(eval_config["experiments"]["rag"]) == {
        "rag_embedding",
        "rag_bm25",
        "rag_evidence_weight",
        "rag_term_norm",
        "rag_citation_review",
        "rag_full",
    }
    assert set(eval_config["experiments"]["agent"]) == {
        "agent_single",
        "agent_fixed_pair",
        "agent_dynamic_pair",
    }


def test_eval_config_retrieval_weights_locked(eval_config: dict) -> None:
    """检索权重必须写入 config，不得只在代码硬编码。"""
    weights = eval_config["retrieval"]["weights"]
    for w in ("w1_bm25", "w2_embedding", "w3_evidence_level", "w4_term_overlap"):
        assert w in weights
        assert isinstance(weights[w], (int, float))


def test_eval_config_metrics_defined(eval_config: dict) -> None:
    """PLAN.md 要求的 7 个核心指标必须定义。"""
    metric_names = {m["name"] for m in eval_config["metrics"]}
    required = {
        "evidence_recall_at_5",
        "gold_evidence_coverage",
        "citation_precision",
        "judge_agreement",
        "unsupported_claim_rate",
        "terminology_normalization_gain",
        "workflow_success_rate",
        "p95_latency_ms",
    }
    assert required.issubset(metric_names), f"missing metrics: {required - metric_names}"


def test_eval_config_leakage_check_fields(eval_config: dict) -> None:
    """泄露校验字段配置完整。"""
    leak = eval_config["leakage_check"]
    assert "source" in leak["chunk_fields_to_check"]
    assert "source_id" in leak["chunk_fields_to_check"]
    assert "metadata.raw_id" in leak["chunk_fields_to_check"]
    assert leak["check_question_text"] is True
    assert leak["check_answer_key"] is True


def test_eval_config_compliance_rules(eval_config: dict) -> None:
    """合规配置包含绝对化措辞拦截和强制风险提示。"""
    comp = eval_config["compliance"]
    assert len(comp["block_absolute_terms"]) > 0
    assert "确诊" in comp["block_absolute_terms"]
    assert comp["mandatory_disclaimer"]


def test_eval_config_reproduction_commands(eval_config: dict) -> None:
    """可复现命令必须定义。"""
    repro = eval_config["reproduction"]
    assert "validate_command" in repro
    assert "rag_command" in repro
    assert "agent_command" in repro
    assert "leakage_check_command" in repro
