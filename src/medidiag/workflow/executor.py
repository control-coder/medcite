"""任务执行器：状态推进、幂等控制、乐观锁重试、RUNNING 重复请求处理。

PLAN.md 事务边界:
- 数据库事务只负责记录"要做什么"和"状态如何变化"
- 外部 IO（RAG/LLM/judge）在事务外执行
- 状态变更和事件日志写入必须在同一事务内完成
- 外部调用成功后，再开启事务写入结果、推进状态、追加事件

乐观锁策略（PLAN.md）:
- cases 表包含 version 字段
- 状态更新使用 WHERE id = ? AND version = ?
- 冲突后最多自动重试 3 次，退避 50ms -> 100ms -> 200ms
- 重试前重新读取 case 状态，如果目标状态已被推进到等价或更后状态，则直接返回成功
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from medidiag.db.models import Case, CaseEventLog, WorkflowTask
from medidiag.errors import MediDiagError
from medidiag.workflow.lease import LeaseManager
from medidiag.workflow.state_machine import (
    CaseState,
    IllegalTransitionError,
    TriggerSubject,
    validate_transition,
)

# 乐观锁重试参数（PLAN.md: 3 次退避 50/100/200ms）
OPTIMISTIC_LOCK_MAX_RETRIES = 3
OPTIMISTIC_LOCK_BACKOFF_MS: list[int] = [50, 100, 200]


class WorkflowExecutor:
    """工作流执行器。

    职责:
    - 幂等创建病例（Idempotency-Key + user_scope）
    - 幂等启动工作流（处理 RUNNING 重复请求）
    - 状态推进（乐观锁重试 + 状态+事件同事务）
    - 外部结果写入（原子 lease/result CAS，防脑裂双写）
    """

    def __init__(self, lease_manager: LeaseManager | None = None) -> None:
        self.lease = lease_manager or LeaseManager()

    # ===== 创建病例（幂等）=====

    def create_case(
        self,
        session: Session,
        question: str,
        idempotency_key: str,
        user_scope: str,
        gold_answer: str | None = None,
        input_kind: str = "deidentified_simulation",
        source_ref: str | None = None,
    ) -> Case:
        """幂等创建病例。

        幂等键: Idempotency-Key + user_scope。
        重复请求返回已有病例，不新建。

        Args:
            session: 数据库会话。
            question: 病例问题。
            idempotency_key: 调用方提供的幂等键。
            user_scope: 用户作用域（配合幂等键去重）。
            gold_answer: 标准答案（评测用，可选）。

        Returns:
            Case 对象（新建或已有）。
        """
        # 查找同幂等键的现有病例
        existing = session.execute(
            select(Case).where(
                Case.idempotency_key == idempotency_key,
                Case.idempotency_user_scope == user_scope,
            )
        ).scalar_one_or_none()
        if existing:
            return existing  # 幂等返回

        # 创建新病例 + 同事务写事件日志
        case = Case(
            case_id=f"case_{uuid.uuid4().hex[:12]}",
            status=CaseState.CREATED.value,
            version=1,
            question=question,
            gold_answer=gold_answer,
            input_kind=input_kind,
            source_ref=source_ref,
            idempotency_key=idempotency_key,
            idempotency_user_scope=user_scope,
        )
        session.add(case)
        session.add(
            CaseEventLog(
                case_id=case.case_id,
                event_type="case_created",
                to_status=CaseState.CREATED.value,
                trigger_subject=TriggerSubject.API.value,
                detail={"idempotency_key": idempotency_key},
            )
        )
        try:
            session.commit()
            return case
        except IntegrityError:
            # The database constraint is the final arbiter for concurrent
            # requests that both missed the initial read.
            session.rollback()
            existing = session.execute(
                select(Case).where(
                    Case.idempotency_key == idempotency_key,
                    Case.idempotency_user_scope == user_scope,
                )
            ).scalar_one_or_none()
            if existing is not None:
                return existing
            raise

    # ===== 启动工作流（幂等 + RUNNING 重复请求处理）=====

    def start_workflow(
        self,
        session: Session,
        case_id: str,
        task_type: str,
        idempotency_key: str,
        input_hash: str,
    ) -> WorkflowTask:
        """幂等启动工作流任务。

        幂等键: case_id + workflow_type（通过 idempotency_key 参数传入）。

        RUNNING 重复请求处理（PLAN.md）:
        - 同幂等键: 返回已有任务，不新建 worker
        - 不同幂等键但同 case 有 RUNNING: 抛 WORKFLOW_ALREADY_RUNNING
        - 已成功完成: 返回已有结果

        Args:
            session: 数据库会话。
            case_id: 病例 ID。
            task_type: 任务类型（normalize/retrieve/plan/review/arbitrate/report）。
            idempotency_key: 工作流幂等键。
            input_hash: 输入哈希。

        Returns:
            WorkflowTask 对象。

        Raises:
            MediDiagError: WORKFLOW_ALREADY_RUNNING（不同幂等键冲突）。
        """
        case = session.execute(
            select(Case).where(Case.case_id == case_id)
        ).scalar_one_or_none()
        if case is None:
            raise MediDiagError("CASE_NOT_FOUND", detail=f"case {case_id} not found")

        # 1. 查找同一病例/任务类型/幂等键的逻辑任务
        existing = session.execute(
            select(WorkflowTask).where(
                WorkflowTask.case_id == case_id,
                WorkflowTask.task_type == task_type,
                WorkflowTask.idempotency_key == idempotency_key,
            )
        ).scalar_one_or_none()

        if existing:
            if existing.status in ("SUCCEEDED", "RUNNING", "PENDING"):
                return existing  # 返回已有任务（幂等）

        # active_task_id covers both PENDING and RUNNING tasks.
        expected_active_task_id: str | None = None
        if case.active_task_id:
            active = session.execute(
                select(WorkflowTask).where(
                    WorkflowTask.task_id == case.active_task_id
                )
            ).scalar_one_or_none()
            if active and active.idempotency_key == idempotency_key:
                if active.status in ("SUCCEEDED", "RUNNING", "PENDING"):
                    return active
                expected_active_task_id = active.task_id
            else:
                raise MediDiagError(
                    "WORKFLOW_ALREADY_RUNNING",
                    detail=f"case {case_id} already has active task {case.active_task_id}",
                    context={"existing_task_id": case.active_task_id},
                )

        active_predicate = (
            Case.active_task_id.is_(None)
            if expected_active_task_id is None
            else Case.active_task_id == expected_active_task_id
        )
        if expected_active_task_id and (
            existing is None or existing.task_id != expected_active_task_id
        ):
            raise MediDiagError(
                "WORKFLOW_ALREADY_RUNNING",
                detail=f"case {case_id} active task identity mismatch",
                context={"existing_task_id": expected_active_task_id},
            )

        task_id = existing.task_id if existing else f"task_{uuid.uuid4().hex[:12]}"
        claimed = session.execute(
            update(Case)
            .where(
                Case.case_id == case_id,
                Case.version == case.version,
                active_predicate,
            )
            .values(active_task_id=task_id, version=case.version + 1)
        )
        if claimed.rowcount != 1:
            session.rollback()
            refreshed = session.execute(
                select(Case).where(Case.case_id == case_id)
            ).scalar_one()
            active = (
                session.execute(
                    select(WorkflowTask).where(
                        WorkflowTask.task_id == refreshed.active_task_id
                    )
                ).scalar_one_or_none()
                if refreshed.active_task_id
                else None
            )
            if (
                active
                and active.task_type == task_type
                and active.idempotency_key == idempotency_key
            ):
                return active
            raise MediDiagError(
                "WORKFLOW_ALREADY_RUNNING",
                detail=f"case {case_id} active task CAS conflict",
                context={"existing_task_id": refreshed.active_task_id},
            )

        if existing:
            existing.status = "PENDING"
            existing.input_hash = input_hash
            existing.attempt += 1
            existing.lease_owner = None
            existing.lease_until = None
            existing.heartbeat_at = None
            existing.result = None
            existing.error_code = None
            existing.error_message = None
            task = existing
        else:
            task = WorkflowTask(
                task_id=task_id,
                case_id=case_id,
                task_type=task_type,
                status="PENDING",
                input_hash=input_hash,
                idempotency_key=idempotency_key,
            )
            session.add(task)
        session.add(
            CaseEventLog(
                case_id=case_id,
                event_type="workflow_started",
                from_status=case.status,
                to_status=case.status,
                trigger_subject=TriggerSubject.API.value,
                detail={
                    "task_id": task_id,
                    "task_type": task_type,
                    "attempt": task.attempt,
                    "idempotency_key": idempotency_key,
                },
            )
        )
        try:
            session.commit()
            return task
        except IntegrityError:
            session.rollback()
            winner = session.execute(
                select(WorkflowTask).where(
                    WorkflowTask.case_id == case_id,
                    WorkflowTask.task_type == task_type,
                    WorkflowTask.idempotency_key == idempotency_key,
                )
            ).scalar_one_or_none()
            if winner is not None:
                return winner
            raise

    # ===== 状态推进（乐观锁重试 + 状态+事件同事务）=====

    def advance_state(
        self,
        session: Session,
        case_id: str,
        to_state: CaseState,
        subject: TriggerSubject,
        trigger_entity: str | None = None,
        event_type: str = "state_transition",
        detail: dict | None = None,
    ) -> None:
        """推进病例状态。

        乐观锁冲突自动重试（3 次退避 50/100/200ms）。
        状态变更 + 事件日志在同一事务内。
        重试前重读 case 状态，如果已推进到目标状态则直接返回（幂等）。

        Args:
            session: 数据库会话。
            case_id: 病例 ID。
            to_state: 目标状态。
            subject: 触发主体。
            trigger_entity: 触发实体标识（worker_id 等）。
            event_type: 事件类型。
            detail: 事件详情。

        Raises:
            MediDiagError: CASE_NOT_FOUND / ILLEGAL_STATE_TRANSITION / OPTIMISTIC_LOCK_CONFLICT。
        """
        last_error: MediDiagError | None = None

        for attempt in range(OPTIMISTIC_LOCK_MAX_RETRIES):
            # 重读 case 状态
            case = session.execute(
                select(Case).where(Case.case_id == case_id)
            ).scalar_one_or_none()

            if case is None:
                raise MediDiagError(
                    "CASE_NOT_FOUND", detail=f"case {case_id} not found"
                )

            current_state = CaseState(case.status)

            # 幂等：已推进到目标状态
            if current_state == to_state:
                return

            # 校验跳转合法性
            try:
                validate_transition(current_state, to_state, subject)
            except IllegalTransitionError as e:
                raise MediDiagError(
                    "ILLEGAL_STATE_TRANSITION",
                    detail=str(e),
                    context={"from": current_state.value, "to": to_state.value},
                ) from e

            # 尝试乐观锁更新（WHERE case_id=? AND version=?）
            result = session.execute(
                update(Case)
                .where(Case.case_id == case_id, Case.version == case.version)
                .values(status=to_state.value, version=case.version + 1)
            )

            if result.rowcount == 1:
                # 成功，同事务写事件日志
                session.add(
                    CaseEventLog(
                        case_id=case_id,
                        event_type=event_type,
                        from_status=current_state.value,
                        to_status=to_state.value,
                        trigger_subject=subject.value,
                        trigger_entity=trigger_entity,
                        detail=detail,
                    )
                )
                session.commit()
                return

            # 冲突，回滚并退避重试
            session.rollback()
            last_error = MediDiagError(
                "OPTIMISTIC_LOCK_CONFLICT",
                detail=f"version conflict on attempt {attempt + 1}",
                context={"case_id": case_id, "attempt": attempt + 1},
            )
            if attempt < OPTIMISTIC_LOCK_MAX_RETRIES - 1:
                time.sleep(OPTIMISTIC_LOCK_BACKOFF_MS[attempt] / 1000.0)

        # 重试耗尽
        raise last_error or MediDiagError(
            "WORKFLOW_RETRY_EXCEEDED",
            detail=f"optimistic lock retry exhausted for case {case_id}",
        )

    # ===== 外部结果写入（原子 lease/result CAS）=====

    def write_external_result(
        self,
        session: Session,
        task_id: str,
        worker_id: str,
        attempt: int,
        result: dict,
    ) -> bool:
        """写入外部 IO 结果。

        结果写入与 lease 校验合并为单条条件 UPDATE（防脑裂双写）。
        如果租约已失效，丢弃写入并抛 TASK_LEASE_LOST。

        Args:
            session: 数据库会话。
            task_id: 任务 ID。
            worker_id: worker 标识。
            attempt: 任务尝试次数（用于租约校验）。
            result: 外部 IO 结果。

        Returns:
            True 如果写入成功。

        Raises:
            MediDiagError: TASK_LEASE_LOST（租约已失效，写入被丢弃）。
        """
        now = self.lease.now()
        write = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.lease_owner == worker_id,
                WorkflowTask.attempt == attempt,
                WorkflowTask.status == "RUNNING",
                WorkflowTask.lease_until > now,
            )
            .values(status="SUCCEEDED", result=result)
        )
        if write.rowcount == 1:
            task = session.execute(
                select(WorkflowTask).where(WorkflowTask.task_id == task_id)
            ).scalar_one()
            cleared = session.execute(
                update(Case)
                .where(
                    Case.case_id == task.case_id,
                    Case.active_task_id == task_id,
                )
                .values(active_task_id=None, version=Case.version + 1)
            )
            if cleared.rowcount == 1:
                session.commit()
                return True

        session.rollback()
        # Record the rejected stale write in a separate transaction.
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one_or_none()
        if task:
            session.add(
                CaseEventLog(
                    case_id=task.case_id,
                    event_type="lease_lost",
                    trigger_subject=TriggerSubject.SYSTEM.value,
                    trigger_entity=worker_id,
                    detail={
                        "task_id": task_id,
                        "attempt": attempt,
                        "reason": "TASK_LEASE_LOST",
                    },
                )
            )
            session.commit()
        raise MediDiagError(
            "TASK_LEASE_LOST",
            detail=f"worker {worker_id} lost lease on task {task_id}",
            context={"task_id": task_id, "attempt": attempt},
        )

    def commit_stage(
        self,
        session: Session,
        *,
        task_id: str,
        worker_id: str,
        attempt: int,
        to_state: CaseState,
        subject: TriggerSubject,
        stage: str,
        records: list[Any] | None = None,
        case_values: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
        complete_task: bool = False,
        task_result: dict[str, Any] | None = None,
    ) -> None:
        """Atomically fence a worker and persist one workflow stage.

        The task lease predicate, case optimistic-lock predicate, stage records,
        state transition, and event append share one transaction. External IO
        must complete before this method is called.
        """
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one_or_none()
        if task is None:
            raise MediDiagError("TASK_LEASE_LOST", detail=f"task {task_id} not found")

        now = self.lease.now()
        task_updates: dict[str, Any] = {"heartbeat_at": now}
        if complete_task:
            task_updates.update(status="SUCCEEDED", result=task_result or {})
        fenced = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.lease_owner == worker_id,
                WorkflowTask.attempt == attempt,
                WorkflowTask.status == "RUNNING",
                WorkflowTask.lease_until > now,
            )
            .values(**task_updates)
        )
        if fenced.rowcount != 1:
            session.rollback()
            self._record_lease_lost(session, task, worker_id, attempt, stage)
            raise MediDiagError(
                "TASK_LEASE_LOST",
                detail=f"worker {worker_id} lost lease on task {task_id}",
                context={"task_id": task_id, "attempt": attempt, "stage": stage},
            )

        case = session.execute(
            select(Case).where(Case.case_id == task.case_id)
        ).scalar_one_or_none()
        if case is None:
            session.rollback()
            raise MediDiagError("CASE_NOT_FOUND", detail=f"case {task.case_id} not found")
        current_state = CaseState(case.status)
        try:
            validate_transition(current_state, to_state, subject)
        except IllegalTransitionError as exc:
            session.rollback()
            raise MediDiagError(
                "ILLEGAL_STATE_TRANSITION",
                detail=str(exc),
                context={"from": current_state.value, "to": to_state.value},
            ) from exc

        values: dict[str, Any] = {
            "status": to_state.value,
            "version": case.version + 1,
            **(case_values or {}),
        }
        if complete_task:
            values["active_task_id"] = None
        advanced = session.execute(
            update(Case)
            .where(
                Case.case_id == case.case_id,
                Case.version == case.version,
                Case.active_task_id == task_id,
            )
            .values(**values)
        )
        if advanced.rowcount != 1:
            session.rollback()
            raise MediDiagError(
                "OPTIMISTIC_LOCK_CONFLICT",
                detail=f"stage {stage} case CAS conflict",
                context={"case_id": case.case_id, "task_id": task_id},
            )

        for record in records or []:
            session.add(record)
        session.add(
            CaseEventLog(
                case_id=case.case_id,
                event_type="stage_completed",
                from_status=current_state.value,
                to_status=to_state.value,
                trigger_subject=subject.value,
                trigger_entity=worker_id,
                detail={"stage": stage, "attempt": attempt, **(detail or {})},
            )
        )
        session.commit()

    @staticmethod
    def _record_lease_lost(
        session: Session,
        task: WorkflowTask,
        worker_id: str,
        attempt: int,
        stage: str,
    ) -> None:
        session.add(
            CaseEventLog(
                case_id=task.case_id,
                event_type="lease_lost",
                trigger_subject=TriggerSubject.SYSTEM.value,
                trigger_entity=worker_id,
                detail={
                    "task_id": task.task_id,
                    "attempt": attempt,
                    "stage": stage,
                    "reason": "TASK_LEASE_LOST",
                },
            )
        )
        session.commit()
