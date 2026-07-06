"""错误码分级定义。

按 PLAN.md 要求，错误码分五类，每个错误码必须包含：
- 是否可重试
- 默认处理策略
- 是否需要人工升级
- 是否计入告警

分类：
    4xx 用户/输入错误       — 调用方问题，不重试
    42x 业务流程错误        — 状态/并发冲突，部分可重试
    52x 外部依赖错误        — RAG/LLM/judge 超时或不可用，可重试
    53x 数据质量错误        — 检索空、引用不成立、合规拦截，一般不重试
    55x 系统错误            — 租约丢失、重试耗尽，需人工升级
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class ErrorCategory(str, Enum):
    """错误码大类。对应 PLAN.md 的五级分类。"""

    USER_INPUT = "4xx"        # 用户/输入错误
    BUSINESS = "42x"          # 业务流程错误
    DEPENDENCY = "52x"        # 外部依赖错误
    DATA_QUALITY = "53x"      # 数据质量错误
    SYSTEM = "55x"            # 系统错误


@dataclass(frozen=True)
class ErrorSpec:
    """单个错误码的完整规格。"""

    code: str
    """错误码字符串，如 ``STATE_CONFLICT``。"""

    http_status: int
    """对应的 HTTP 状态码。"""

    category: ErrorCategory
    """错误大类。"""

    retryable: bool
    """是否可重试。"""

    default_action: str
    """默认处理策略（人类可读）。"""

    requires_human_escalation: bool
    """是否需要人工升级。"""

    alert: bool
    """是否计入告警。"""

    description: str
    """错误说明。"""

    @property
    def numeric_prefix(self) -> int:
        """返回数字前缀，如 ``42x`` -> ``420``。用于日志分组。"""
        mapping = {
            ErrorCategory.USER_INPUT: 400,
            ErrorCategory.BUSINESS: 420,
            ErrorCategory.DEPENDENCY: 520,
            ErrorCategory.DATA_QUALITY: 530,
            ErrorCategory.SYSTEM: 550,
        }
        return mapping[self.category]


# ===== 错误码注册表 =====
# 所有错误码集中定义，便于审计与告警配置。
_REGISTRY: Final[dict[str, ErrorSpec]] = {
    # ----- 4xx 用户/输入错误 -----
    "CASE_INVALID_INPUT": ErrorSpec(
        code="CASE_INVALID_INPUT",
        http_status=400,
        category=ErrorCategory.USER_INPUT,
        retryable=False,
        default_action="返回 400，要求调用方修正输入。",
        requires_human_escalation=False,
        alert=False,
        description="病例输入字段缺失、格式错误或非法值。",
    ),
    "CASE_NOT_FOUND": ErrorSpec(
        code="CASE_NOT_FOUND",
        http_status=404,
        category=ErrorCategory.USER_INPUT,
        retryable=False,
        default_action="返回 404。",
        requires_human_escalation=False,
        alert=False,
        description="病例 ID 不存在。",
    ),
    "CASE_ALREADY_CLOSED": ErrorSpec(
        code="CASE_ALREADY_CLOSED",
        http_status=409,
        category=ErrorCategory.USER_INPUT,
        retryable=False,
        default_action="返回 409，告知终态病例不可变更。",
        requires_human_escalation=False,
        alert=False,
        description="病例已进入 CLOSED_* 终态，拒绝进一步操作。",
    ),
    "CASE_CANCELLED": ErrorSpec(
        code="CASE_CANCELLED",
        http_status=409,
        category=ErrorCategory.USER_INPUT,
        retryable=False,
        default_action="返回 409，记录 CLOSED_CANCELLED。",
        requires_human_escalation=False,
        alert=False,
        description="用户主动取消病例。",
    ),
    "IDEMPOTENCY_KEY_MISSING": ErrorSpec(
        code="IDEMPOTENCY_KEY_MISSING",
        http_status=400,
        category=ErrorCategory.USER_INPUT,
        retryable=False,
        default_action="返回 400，要求调用方提供 Idempotency-Key。",
        requires_human_escalation=False,
        alert=False,
        description="创建病例或启动工作流时缺少幂等键。",
    ),

    # ----- 42x 业务流程错误 -----
    "STATE_CONFLICT": ErrorSpec(
        code="STATE_CONFLICT",
        http_status=409,
        category=ErrorCategory.BUSINESS,
        retryable=True,
        default_action="乐观锁冲突，自动重试 3 次（退避 50/100/200ms）；"
        "重试前重读 case 状态，若目标状态已被推进到等价或更后状态则返回成功，"
        "否则写入 STATE_CONFLICT 并进入人工升级或失败处理。",
        requires_human_escalation=False,
        alert=True,
        description="乐观锁冲突或状态跳转非法。",
    ),
    "WORKFLOW_ALREADY_RUNNING": ErrorSpec(
        code="WORKFLOW_ALREADY_RUNNING",
        http_status=409,
        category=ErrorCategory.BUSINESS,
        retryable=False,
        default_action="返回 409；若同幂等键则返回已有 task_id，"
        "若不同幂等键但同一 case 正在运行则拒绝新建 worker。",
        requires_human_escalation=False,
        alert=False,
        description="同一 case 已有 RUNNING 任务，拒绝重复启动。",
    ),
    "ILLEGAL_STATE_TRANSITION": ErrorSpec(
        code="ILLEGAL_STATE_TRANSITION",
        http_status=409,
        category=ErrorCategory.BUSINESS,
        retryable=False,
        default_action="返回 409，记录非法跳转尝试到 event_log。",
        requires_human_escalation=False,
        alert=True,
        description="状态机非法跳转（不在合法跳转表内）。",
    ),
    "MAX_REVIEW_ROUNDS_EXCEEDED": ErrorSpec(
        code="MAX_REVIEW_ROUNDS_EXCEEDED",
        http_status=422,
        category=ErrorCategory.BUSINESS,
        retryable=False,
        default_action="进入 ESCALATED 等待人工介入，不进死状态。",
        requires_human_escalation=True,
        alert=True,
        description="审核连续驳回超过最大轮次。",
    ),

    # ----- 52x 外部依赖错误 -----
    "RAG_TIMEOUT": ErrorSpec(
        code="RAG_TIMEOUT",
        http_status=504,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="指数退避重试；连续失败后降级为纯 BM25 检索并记录降级事件。",
        requires_human_escalation=False,
        alert=True,
        description="RAG 检索超时。",
    ),
    "LLM_TIMEOUT": ErrorSpec(
        code="LLM_TIMEOUT",
        http_status=504,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="指数退避重试；重试次数耗尽后进入 ESCALATED。",
        requires_human_escalation=False,
        alert=True,
        description="LLM 调用超时。",
    ),
    "LLM_JSON_INVALID": ErrorSpec(
        code="LLM_JSON_INVALID",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="重试时强化 schema 提示；比例超过 5% 触发告警。",
        requires_human_escalation=False,
        alert=True,
        description="LLM 返回的 JSON 不符合 schema。",
    ),
    "JUDGE_TIMEOUT": ErrorSpec(
        code="JUDGE_TIMEOUT",
        http_status=504,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="指数退避重试；judge 不可用时降级为规则校验并标记 JUDGE_DEGRADED。",
        requires_human_escalation=False,
        alert=True,
        description="NLI/cross-encoder judge 模型调用超时。",
    ),
    "EMBEDDING_TIMEOUT": ErrorSpec(
        code="EMBEDDING_TIMEOUT",
        http_status=504,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="指数退避重试；连续失败后降级为纯 BM25。",
        requires_human_escalation=False,
        alert=True,
        description="embedding 模型调用超时。",
    ),

    # ----- 53x 数据质量错误 -----
    "RAG_EMPTY_RESULT": ErrorSpec(
        code="RAG_EMPTY_RESULT",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="放宽检索阈值重试一次；仍空则进入 REVISION_REQUIRED 并标记证据不足。",
        requires_human_escalation=False,
        alert=True,
        description="RAG 检索返回空结果。",
    ),
    "REVIEW_UNSUPPORTED_CLAIM": ErrorSpec(
        code="REVIEW_UNSUPPORTED_CLAIM",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="驳回至 REVISION_REQUIRED，要求 Agent 重新生成带证据 claim。",
        requires_human_escalation=False,
        alert=False,
        description="审核发现 UNSUPPORTED claim 比例超阈值。",
    ),
    "CITATION_VERIFICATION_FAILED": ErrorSpec(
        code="CITATION_VERIFICATION_FAILED",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="驳回至 REVISION_REQUIRED；记录不支持 claim 的证据 ID。",
        requires_human_escalation=False,
        alert=False,
        description="引用校验失败，claim 无法被 evidence 支持。",
    ),
    "COMPLIANCE_BLOCKED": ErrorSpec(
        code="COMPLIANCE_BLOCKED",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="拦截输出，写入合规命中日志；超范围问题返回拒答模板。",
        requires_human_escalation=True,
        alert=True,
        description="合规管控拦截（绝对化措辞/超范围/不当输出）。",
    ),
    "EVAL_DATA_LEAKAGE_DETECTED": ErrorSpec(
        code="EVAL_DATA_LEAKAGE_DETECTED",
        http_status=500,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="评测立即失败，禁止继续；要求修复知识库 chunk metadata。",
        requires_human_escalation=True,
        alert=True,
        description="数据泄露校验命中：测试样本 ID 出现在知识库 chunk source/source_id/metadata.raw_id。",
    ),

    # ----- 55x 系统错误 -----
    "TASK_LEASE_EXPIRED": ErrorSpec(
        code="TASK_LEASE_EXPIRED",
        http_status=500,
        category=ErrorCategory.SYSTEM,
        retryable=True,
        default_action="扫描器接管任务，重入校验后重新执行；旧 worker 写入被丢弃。",
        requires_human_escalation=False,
        alert=True,
        description="worker 租约过期，任务被扫描器接管。",
    ),
    "TASK_LEASE_LOST": ErrorSpec(
        code="TASK_LEASE_LOST",
        http_status=500,
        category=ErrorCategory.SYSTEM,
        retryable=False,
        default_action="丢弃旧 worker 的迟到写入，记录事件，不覆盖新 worker 结果（防脑裂双写）。",
        requires_human_escalation=False,
        alert=True,
        description="旧 worker 恢复后租约已失效，写入被拒绝。",
    ),
    "OPTIMISTIC_LOCK_CONFLICT": ErrorSpec(
        code="OPTIMISTIC_LOCK_CONFLICT",
        http_status=409,
        category=ErrorCategory.SYSTEM,
        retryable=True,
        default_action="自动重试 3 次（退避 50/100/200ms）；重试前重读 case 状态判断兼容性。",
        requires_human_escalation=False,
        alert=True,
        description="乐观锁 version 冲突。",
    ),
    "WORKFLOW_RETRY_EXCEEDED": ErrorSpec(
        code="WORKFLOW_RETRY_EXCEEDED",
        http_status=500,
        category=ErrorCategory.SYSTEM,
        retryable=False,
        default_action="进入 ESCALATED 或 CLOSED_FAILED，记录失败链路。",
        requires_human_escalation=True,
        alert=True,
        description="工作流重试次数耗尽。",
    ),
    "AST_PARSE_FAILED": ErrorSpec(
        code="AST_PARSE_FAILED",
        http_status=500,
        category=ErrorCategory.SYSTEM,
        retryable=False,
        default_action="降级到文本窗口定位（MiniCoder 用，此处预留）。",
        requires_human_escalation=False,
        alert=False,
        description="AST 解析失败（预留，MediDiag 不使用）。",
    ),
}


def get_error_spec(code: str) -> ErrorSpec:
    """根据错误码字符串返回其规格。

    Raises:
        KeyError: 错误码未注册。
    """
    return _REGISTRY[code]


def all_error_codes() -> dict[str, ErrorSpec]:
    """返回全部错误码注册表（只读视图）。"""
    return dict(_REGISTRY)


def alertable_codes() -> list[str]:
    """返回所有需要告警的错误码。"""
    return [code for code, spec in _REGISTRY.items() if spec.alert]


def retryable_codes() -> list[str]:
    """返回所有可重试的错误码。"""
    return [code for code, spec in _REGISTRY.items() if spec.retryable]


class MediDiagError(Exception):
    """MediDiag 业务异常基类。

    所有业务错误应携带错误码，便于日志、告警和客户端处理。
    """

    def __init__(self, code: str, *, detail: str | None = None, context: dict | None = None) -> None:
        self.spec = get_error_spec(code)
        self.code = code
        self.detail = detail or self.spec.description
        self.context = context or {}
        super().__init__(f"[{code}] {self.detail}")
