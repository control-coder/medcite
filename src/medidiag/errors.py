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
    53x 数据质量错误        — 合规拦截、评测数据泄露，一般不重试
    55x 系统错误            — 租约丢失、重试耗尽，需人工升级

注册表只登记真实会被抛出或写入事件的错误码。一个描述了未实现 ``default_action``
的错误码比更小的注册表更糟：它承诺了一条实际不存在的处理路径，会误导告警配置和
事故排查。``tests/test_smoke.py::test_every_registered_code_is_actually_raised``
扫描 ``src/`` 与 ``eval/`` 强制这条约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final


class ErrorCategory(str, Enum):
    """错误码大类。对应 PLAN.md 的五级分类。"""

    USER_INPUT = "4xx"  # 用户/输入错误
    BUSINESS = "42x"  # 业务流程错误
    DEPENDENCY = "52x"  # 外部依赖错误
    DATA_QUALITY = "53x"  # 数据质量错误
    SYSTEM = "55x"  # 系统错误


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
    "CASE_INPUT_NOT_DEIDENTIFIED": ErrorSpec(
        code="CASE_INPUT_NOT_DEIDENTIFIED",
        http_status=400,
        category=ErrorCategory.USER_INPUT,
        retryable=False,
        default_action="拒绝输入，要求移除直接身份标识后重试。",
        requires_human_escalation=False,
        alert=False,
        description="模拟病例包含明显邮箱、电话或身份证格式，未通过脱敏门禁。",
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
    "REPORT_NOT_READY": ErrorSpec(
        code="REPORT_NOT_READY",
        http_status=409,
        category=ErrorCategory.BUSINESS,
        retryable=True,
        default_action="等待工作流生成报告后重试。",
        requires_human_escalation=False,
        alert=False,
        description="病例尚未生成结构化报告。",
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
        default_action="指数退避重试；耗尽后进入 ESCALATED，禁止降级为规则校验。",
        requires_human_escalation=False,
        alert=True,
        description="NLI/cross-encoder judge 模型调用超时。",
    ),
    "JUDGE_UNAVAILABLE": ErrorSpec(
        code="JUDGE_UNAVAILABLE",
        http_status=503,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="有限重试固定 NLI judge；耗尽后进入 ESCALATED，禁止规则 fallback。",
        requires_human_escalation=False,
        alert=True,
        description="固定 NLI judge 初始化、标签校验或推理失败。",
    ),
    "PROVIDER_RATE_LIMITED": ErrorSpec(
        code="PROVIDER_RATE_LIMITED",
        http_status=429,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="按有限退避重试；耗尽后保留任务供租约恢复，不无限重试。",
        requires_human_escalation=False,
        alert=True,
        description="外部 provider 返回 429 限流。",
    ),
    "PROVIDER_NETWORK_ERROR": ErrorSpec(
        code="PROVIDER_NETWORK_ERROR",
        http_status=503,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="按有限退避重试；耗尽后保留任务供租约恢复，不无限重试。",
        requires_human_escalation=False,
        alert=True,
        description="调用外部 provider 时发生传输层故障（DNS 解析失败、连接被拒、连接中断等），未收到 HTTP 响应。",
    ),
    "PROVIDER_RESPONSE_ID_MISSING": ErrorSpec(
        code="PROVIDER_RESPONSE_ID_MISSING",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="正式 response-id 溯源模式下进行有限退避重试；仍缺失时终止本次正式运行，不生成可报告结果。",
        requires_human_escalation=False,
        alert=True,
        description="provider 成功响应未返回可用于调用关联的 request ID。",
    ),
    "PROVIDER_UNAVAILABLE": ErrorSpec(
        code="PROVIDER_UNAVAILABLE",
        http_status=503,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="仅对瞬时 5xx 做有限退避；耗尽后等待租约恢复。",
        requires_human_escalation=False,
        alert=True,
        description="外部 provider 返回瞬时 5xx。",
    ),
    "PROVIDER_REQUEST_REJECTED": ErrorSpec(
        code="PROVIDER_REQUEST_REJECTED",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=False,
        default_action="记录响应状态并停止自动重试，修正请求或配置后再处理。",
        requires_human_escalation=False,
        alert=True,
        description="外部 provider 拒绝请求且错误不属于限流或瞬时 5xx。",
    ),
    "PROVIDER_SCHEMA_INVALID": ErrorSpec(
        code="PROVIDER_SCHEMA_INVALID",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=False,
        default_action="记录 schema 校验错误并停止自动重试，禁止写入阶段产物。",
        requires_human_escalation=False,
        alert=True,
        description="外部 provider 返回值不满足阶段结构化 schema。",
    ),
    "PROVIDER_AUTH_FAILED": ErrorSpec(
        code="PROVIDER_AUTH_FAILED",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=False,
        default_action="停止自动重试，检查 profile 的 api_key_env 与本地密钥配置。",
        requires_human_escalation=False,
        alert=True,
        description="外部 provider 认证或授权失败。",
    ),
    "PROVIDER_CONTEXT_LIMIT": ErrorSpec(
        code="PROVIDER_CONTEXT_LIMIT",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=False,
        default_action="进入证据压缩或 revision 路径，不对相同请求盲目重试。",
        requires_human_escalation=False,
        alert=True,
        description="请求超过 provider 上下文或 token 限制。",
    ),
    "PROVIDER_CONTENT_BLOCKED": ErrorSpec(
        code="PROVIDER_CONTENT_BLOCKED",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=False,
        default_action="记录策略拒绝并进入合规或人工升级流程。",
        requires_human_escalation=True,
        alert=True,
        description="provider 内容策略拒绝请求或响应。",
    ),
    "STRUCTURED_OUTPUT_INVALID": ErrorSpec(
        code="STRUCTURED_OUTPUT_INVALID",
        http_status=502,
        category=ErrorCategory.DEPENDENCY,
        retryable=False,
        default_action="最多执行一次受控修复；仍失败则进入 revision，禁止强转为成功结果。",
        requires_human_escalation=False,
        alert=True,
        description="provider 返回的结构化输出无法解析为约定 JSON object。",
    ),
    "AGENT_RUNTIME_INVALID": ErrorSpec(
        code="AGENT_RUNTIME_INVALID",
        http_status=502,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="停止 Agent 阶段，保留 EvidenceBundle 与失败 trace，转人工检查。",
        requires_human_escalation=True,
        alert=True,
        description="Agent topology、引用、结构化输出或 provenance 不满足运行时契约。",
    ),
    "CLAIM_LANGUAGE_INVALID": ErrorSpec(
        code="CLAIM_LANGUAGE_INVALID",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="拒绝进入固定英文 NLI judge，要求重新生成 canonical English claim。",
        requires_human_escalation=False,
        alert=True,
        description="claim 不是固定英文 NLI judge 可接受的 canonical English 文本。",
    ),
    "CITATION_REVIEW_INVALID": ErrorSpec(
        code="CITATION_REVIEW_INVALID",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="停止普通报告链路，保留 claim/evidence artifact 并转人工检查。",
        requires_human_escalation=True,
        alert=True,
        description="citation 审核缺少固定 NLI、claim、证据或唯一标识。",
    ),
    "REPORT_PROVENANCE_INVALID": ErrorSpec(
        code="REPORT_PROVENANCE_INVALID",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="拒绝生成普通报告，检查 claim 过滤、引用和展示层 provenance。",
        requires_human_escalation=True,
        alert=True,
        description="用户报告引入、改写或放行了未经审核的 claim。",
    ),
    "RAG_CORPUS_INVALID": ErrorSpec(
        code="RAG_CORPUS_INVALID",
        http_status=500,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="停止检索并修复 corpus schema、版本或配置。",
        requires_human_escalation=True,
        alert=True,
        description="运行时医学 corpus 缺失、为空或不符合固定 schema。",
    ),
    "RAG_INDEX_BUILD_FAILED": ErrorSpec(
        code="RAG_INDEX_BUILD_FAILED",
        http_status=503,
        category=ErrorCategory.DEPENDENCY,
        retryable=True,
        default_action="检查固定模型 revision 与本地缓存后重建索引。",
        requires_human_escalation=False,
        alert=True,
        description="BM25、embedding 或 FAISS 医学检索索引构建失败。",
    ),
    "RAG_NO_EVIDENCE": ErrorSpec(
        code="RAG_NO_EVIDENCE",
        http_status=422,
        category=ErrorCategory.DATA_QUALITY,
        retryable=False,
        default_action="禁止进入普通报告链路，记录失败并转人工复核。",
        requires_human_escalation=True,
        alert=False,
        description="检索没有返回可追溯证据。",
    ),
    # ----- 53x 数据质量错误 -----
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

    def __init__(
        self, code: str, *, detail: str | None = None, context: dict[str, Any] | None = None
    ) -> None:
        self.spec = get_error_spec(code)
        self.code = code
        self.detail = detail or self.spec.description
        self.context = context or {}
        super().__init__(f"[{code}] {self.detail}")
