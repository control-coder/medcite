"""MVP API 使用的 Pydantic 数据结构。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CaseCreateRequest(BaseModel):
    question: str = Field(min_length=10, max_length=12000)
    input_kind: Literal["public_dataset", "deidentified_simulation"]
    source_ref: str | None = Field(default=None, max_length=128)

    @field_validator("question")
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
