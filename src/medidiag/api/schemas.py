"""MVP API 使用的 Pydantic 数据结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CaseCreateRequest(BaseModel):
    question: str = Field(min_length=10, max_length=12000)
    input_kind: Literal["public_dataset", "deidentified_simulation"]
    source_ref: str | None = Field(default=None, max_length=128)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class WorkflowStartResponse(BaseModel):
    task_id: str
    case_id: str
    status: str
    attempt: int


class CaseResponse(BaseModel):
    case_id: str
    trace_id: str
    status: str
    version: int
    input_kind: str
    source_ref: str | None
    normalized_query: str | None
    active_task: WorkflowStartResponse | None
    available_human_actions: list[str]


class EventResponse(BaseModel):
    event_id: int
    event_type: str
    from_status: str | None
    to_status: str | None
    trigger_subject: str
    trigger_entity: str | None
    detail: dict[str, Any] | None
    created_at: str


class EventPage(BaseModel):
    items: list[EventResponse]
    next_cursor: int | None


class HumanDecisionRequest(BaseModel):
    decision: Literal["REVISION_REQUIRED", "APPROVED", "CLOSED_ESCALATED"]
    reason: str = Field(min_length=3, max_length=2000)

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reason must not be blank")
        return value


class ReportResponse(BaseModel):
    report_id: str
    case_id: str
    version: int
    structured_report: dict[str, Any]
    risk_warnings: list[str]
    compliance_status: str
    generation_version: str
    created_at: str


class ErrorResponse(BaseModel):
    code: str
    detail: str
    retryable: bool
    action: str


class ConsultationCreateRequest(BaseModel):
    """用户入口收集必要背景并要求非敏感数据确认。"""

    symptoms: str = Field(min_length=10, max_length=8000)
    duration: str = Field(min_length=1, max_length=200)
    background: str = Field(default="", max_length=2000)
    input_kind: Literal["public_dataset", "deidentified_simulation"]
    source_ref: str | None = Field(default=None, max_length=128)
    non_sensitive_confirmed: Literal[True]

    @field_validator("symptoms", "duration", "background", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    def as_case(self) -> CaseCreateRequest:
        return CaseCreateRequest(
            question=f"症状：{self.symptoms}\n持续时间：{self.duration}\n背景：{self.background or '未提供'}",
            input_kind=self.input_kind, source_ref=self.source_ref,
        )


class EvidenceResponse(BaseModel):
    chunk_id: str
    text: str
    source: str
    source_id: str
    source_url: str | None = None
    evidence_level: str


class ClaimResponse(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str]


class ObservationResponse(BaseModel):
    """只展示实际落库的观测，缺失账单不推定为零费用。"""

    recorded_stage_latency_ms: int = 0
    provider_attempts: int = 0
    recorded_input_tokens: int | None = None
    recorded_output_tokens: int | None = None
    usage_status: Literal["not_recorded", "partial"] = "not_recorded"
    cost_usd: float | None = None
    note: str = "耗时为最新任务已保存阶段之和，不含排队和未提交阶段；Token 为部分已记录用量，费用未计价。"


class AnalysisResponse(BaseModel):
    schema_version: Literal["consultation-v1"] = "consultation-v1"
    case_id: str
    status: str
    outcome: Literal["processing", "ready", "insufficient_evidence", "failed", "cancelled"]
    message: str
    summary: str | None = None
    observation: ObservationResponse = Field(default_factory=ObservationResponse)
    claims: list[ClaimResponse] = Field(default_factory=list)
    evidence: list[EvidenceResponse] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    retry_action: Literal["none", "resubmit"] = "none"
    disclaimer: str = "仅供公开资料与模拟数据的工程演示，不构成诊断、处方或治疗建议。"
