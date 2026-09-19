"""SQLAlchemy 2.0 数据模型。

PLAN.md 规定的最小字段集，不新增额外字段。

事务边界：
- 状态推进 + 事件追加必须在同一事务内
- 外部 IO（RAG/LLM/judge）不进事务

表清单：
    cases            - 病例（含 version 乐观锁）
    workflow_tasks   - 工作流任务（含 lease 字段）
    case_event_log   - 状态变更与事件日志
    agent_runs       - Agent 执行记录（含 input_hash 幂等）
    citations        - claim 与 evidence 绑定
    reviews          - 审核记录
    stage_artifacts  - worker 阶段结构化产物
    case_reports     - 最终结构化报告
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """UTC 当前时间（naive，便于 SQLite 存储）。"""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。"""

    pass


# ===== 1. cases：病例表（含 version 乐观锁） =====


class Case(Base):
    """病例表。

    ``version`` 字段用于乐观锁并发控制：
    状态更新使用 ``WHERE id = ? AND version = ?``，冲突时重试。
    """

    __tablename__ = "cases"
    __table_args__ = (
        UniqueConstraint(
            "idempotency_user_scope",
            "idempotency_key",
            name="uq_cases_scope_idempotency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="CREATED", index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)  # 乐观锁
    question: Mapped[str] = mapped_column(Text)
    gold_answer: Mapped[str | None] = mapped_column(String(64), nullable=True)
    normalized_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_kind: Mapped[str] = mapped_column(
        String(32), default="deidentified_simulation"
    )
    source_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    trace_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True,
        default=lambda: f"trace_{uuid.uuid4().hex}",
    )
    review_round: Mapped[int] = mapped_column(Integer, default=0)
    # 指向 workflow_tasks.task_id 的逻辑引用。这里刻意不设置外键，
    # 因为 workflow_tasks 已引用 cases，而 SQLite 无法在不重建两张表的情况下
    # 添加由此产生的循环外键。
    active_task_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), index=True)
    idempotency_user_scope: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<Case {self.case_id} status={self.status} v{self.version}>"


# ===== 2. workflow_tasks：工作流任务表（含 lease 字段） =====


class WorkflowTask(Base):
    """工作流任务表。

    租约字段：``lease_owner`` / ``lease_until`` / ``heartbeat_at`` / ``attempt``。
    幂等字段：``input_hash`` / ``idempotency_key``。
    """

    __tablename__ = "workflow_tasks"
    __table_args__ = (
        UniqueConstraint(
            "case_id",
            "task_type",
            "idempotency_key",
            name="uq_workflow_tasks_case_type_idempotency",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    task_type: Mapped[str] = mapped_column(String(32), index=True)
    # normalize / retrieve / plan / review / arbitrate / report：标准阶段名
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    # PENDING / RUNNING / SUCCEEDED / FAILED / STALE：任务状态
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    input_hash: Mapped[str] = mapped_column(String(64))  # 幂等
    idempotency_key: Mapped[str] = mapped_column(String(128))
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    def __repr__(self) -> str:
        return f"<WorkflowTask {self.task_id} type={self.task_type} status={self.status}>"


# ===== 3. case_event_log：状态变更与事件日志 =====


class CaseEventLog(Base):
    """病例事件日志。

    状态变更与事件追加必须在同一事务内完成。
    """

    __tablename__ = "case_event_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    # state_transition / lease_acquire / lease_release / lease_lost / error / ...：事件类型
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    trigger_subject: Mapped[str] = mapped_column(String(32))
    # api / worker / agent_worker / reviewer_worker / human / system：事件主体
    trigger_entity: Mapped[str | None] = mapped_column(String(128), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"<CaseEventLog {self.case_id} {self.event_type} "
            f"{self.from_status}->{self.to_status}>"
        )


# ===== 4. agent_runs：Agent 执行记录（含 input_hash 幂等） =====


class AgentRun(Base):
    """Agent 执行记录。

    ``input_hash`` + ``attempt_group`` 用于幂等：非幂等外部调用前查是否已有成功结果。
    """

    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint(
            "case_id",
            "agent_name",
            "input_hash",
            "attempt_group",
            name="uq_agent_runs_execution_identity",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    # normalize / retrieve / diagnosis / diagnosis_cardiology / diagnosis_respiratory / arbitration：产物阶段
    input_hash: Mapped[str] = mapped_column(String(64))  # 幂等
    attempt_group: Mapped[str] = mapped_column(String(64))
    input_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    output_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="PENDING")
    # PENDING / RUNNING / SUCCEEDED / FAILED：产物状态
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<AgentRun {self.run_id} agent={self.agent_name} status={self.status}>"


# ===== 5. citations：claim 与 evidence 绑定 =====


class Citation(Base):
    """引用记录。claim 与 evidence chunk 绑定，含 verdict 判定。"""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    claim_text: Mapped[str] = mapped_column(Text)
    chunk_id: Mapped[str] = mapped_column(String(64), index=True)
    verdict: Mapped[str] = mapped_column(String(16))
    # SUPPORTED / PARTIAL / UNSUPPORTED：引用审核结果
    verifier_model: Mapped[str] = mapped_column(String(128))
    verifier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    human_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return f"<Citation {self.case_id} chunk={self.chunk_id} verdict={self.verdict}>"


# ===== 6. reviews：审核记录 =====


class Review(Base):
    """审核记录。含审核轮次和结果。"""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    review_type: Mapped[str] = mapped_column(String(32), index=True)
    # citation / logic / compliance / arbitration：审核类型
    reviewer: Mapped[str] = mapped_column(String(128))
    result: Mapped[str] = mapped_column(String(32))
    # APPROVED / REVISION_REQUIRED / ESCALATED：审核状态
    round: Mapped[int] = mapped_column(Integer, default=1)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<Review {self.case_id} type={self.review_type} "
            f"result={self.result} round={self.round}>"
        )


class StageArtifact(Base):
    """单个 worker 阶段的持久化输出。"""

    __tablename__ = "stage_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "case_id", "task_id", "stage", "attempt",
            name="uq_stage_artifacts_execution_stage",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    task_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("workflow_tasks.task_id"), index=True
    )
    stage: Mapped[str] = mapped_column(String(32), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    input_hash: Mapped[str] = mapped_column(String(64))
    output_hash: Mapped[str] = mapped_column(String(64))
    component_version: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class CaseReport(Base):
    """审批后生成的版本化结构化报告。"""

    __tablename__ = "case_reports"
    __table_args__ = (
        UniqueConstraint("case_id", "version", name="uq_case_reports_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cases.case_id"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    structured_report: Mapped[dict[str, Any]] = mapped_column(JSON)
    risk_warnings: Mapped[list[Any]] = mapped_column(JSON)
    compliance_status: Mapped[str] = mapped_column(String(32))
    generation_version: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


# 表清单常量（便于测试与文档引用）
ALL_TABLES: tuple[str, ...] = (
    "cases",
    "workflow_tasks",
    "case_event_log",
    "agent_runs",
    "citations",
    "reviews",
    "stage_artifacts",
    "case_reports",
)
