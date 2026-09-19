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
from medidiag.db.session import rowcount
from medidiag.errors import MediDiagError
from medidiag.observability.logging import get_logger
from medidiag.workflow.lease import LeaseManager
from medidiag.workflow.state_machine import (
    CaseState,
    IllegalTransitionError,
    TriggerSubject,
    validate_transition,
)

# 乐观锁最多重试 3 次；连同首次尝试共 4 次，退避 50/100/200ms。
OPTIMISTIC_LOCK_MAX_RETRIES = 3
OPTIMISTIC_LOCK_BACKOFF_MS: list[int] = [50, 100, 200]

_log = get_logger(__name__)


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
            # 当并发请求都错过首次读取时，
            # 数据库约束作为最终裁决依据。
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

        # active_task_id 同时覆盖 PENDING 和 RUNNING 任务。
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
        if rowcount(claimed) != 1:
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
        detail: dict[str, Any] | None = None,
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

        for attempt in range(OPTIMISTIC_LOCK_MAX_RETRIES + 1):
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

            if rowcount(result) == 1:
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
            if attempt < OPTIMISTIC_LOCK_MAX_RETRIES:
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
        result: dict[str, Any],
    ) -> bool:
        """写入外部 IO 结果。

        结果写入与 lease 校验合并为单条条件 UPDATE（防脑裂双写）。

        两种失败被刻意区分开：

        - 条件 UPDATE 未命中 → 租约确实失效（被接管或已过期）。丢弃写入，
          记录 ``lease_lost``，抛 ``TASK_LEASE_LOST``。
        - 条件 UPDATE 命中、但 ``Case.active_task_id`` 已不指向本任务 →
          租约是有效的，失配的是 case 与 task 的关联。把它报成
          ``TASK_LEASE_LOST`` 会写下一条与事实相反的脑裂记录，并连带丢弃一个
          已经完成的外部结果。这里改为保留任务结果、记录 ``case_task_desync``
          并抛 ``STATE_CONFLICT``，交由调用方升级人工处理；case 状态本身不被
          本方法改写，因此保留结果不会覆盖另一个任务的进展。

        Args:
            session: 数据库会话。
            task_id: 任务 ID。
            worker_id: worker 标识。
            attempt: 任务尝试次数（用于租约校验）。
            result: 外部 IO 结果。

        Returns:
            True 如果写入成功。

        Raises:
            MediDiagError: ``TASK_LEASE_LOST``（租约已失效，写入被丢弃）或
                ``STATE_CONFLICT``（租约有效但 case/task 失配）。
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
        if rowcount(write) != 1:
            session.rollback()
            self._record_rejected_stale_write(session, task_id, worker_id, attempt)
            _log.warning(
                "lease.lost",
                task_id=task_id,
                worker_id=worker_id,
                attempt=attempt,
                error_code="TASK_LEASE_LOST",
                stale_write_discarded=True,
            )
            raise MediDiagError(
                "TASK_LEASE_LOST",
                detail=f"worker {worker_id} lost lease on task {task_id}",
                context={"task_id": task_id, "attempt": attempt},
            )

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
        if rowcount(cleared) == 1:
            session.commit()
            return True

        # 清理未命中：case 已不指向本任务。上面的 Case UPDATE 匹配 0 行，
        # 提交只会持久化任务结果本身，不触碰 case 状态。
        session.commit()
        session.add(
            CaseEventLog(
                case_id=task.case_id,
                event_type="case_task_desync",
                trigger_subject=TriggerSubject.SYSTEM.value,
                trigger_entity=worker_id,
                detail={
                    "task_id": task_id,
                    "attempt": attempt,
                    "error_code": "STATE_CONFLICT",
                    "reason": "CASE_ACTIVE_TASK_MISMATCH",
                    "result_preserved": True,
                },
            )
        )
        session.commit()
        _log.error(
            "case.task_desync",
            task_id=task_id,
            case_id=task.case_id,
            worker_id=worker_id,
            attempt=attempt,
            error_code="STATE_CONFLICT",
            result_preserved=True,
        )
        raise MediDiagError(
            "STATE_CONFLICT",
            detail=(
                f"task {task_id} completed under a valid lease but case "
                f"{task.case_id} no longer points to it"
            ),
            context={"task_id": task_id, "attempt": attempt},
        )

    @staticmethod
    def _record_rejected_stale_write(
        session: Session, task_id: str, worker_id: str, attempt: int
    ) -> None:
        """记录被拒绝的过期写入，并使用独立事务。"""
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one_or_none()
        if task is None:
            return
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

    def fail_stage(
        self,
        session: Session,
        *,
        task_id: str,
        worker_id: str,
        attempt: int,
        to_state: CaseState,
        subject: TriggerSubject,
        stage: str,
        error_code: str,
        error_message: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """原子地停止一个带租约任务，并保留可供人工审核的失败结果。

        这是一个有意设计为终态的任务写入，不是重试机制。它用于 Provider 运行时耗尽有界重试后；不会持久化任何外部结果。
        """
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one_or_none()
        if task is None:
            raise MediDiagError("TASK_LEASE_LOST", detail=f"task {task_id} not found")

        now = self.lease.now()
        fenced = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.lease_owner == worker_id,
                WorkflowTask.attempt == attempt,
                WorkflowTask.status == "RUNNING",
                WorkflowTask.lease_until > now,
            )
            .values(
                status="FAILED",
                heartbeat_at=now,
                lease_until=now,
                error_code=error_code,
                error_message=error_message,
            )
        )
        if rowcount(fenced) != 1:
            session.rollback()
            self._record_lease_lost(session, task, worker_id, attempt, stage)
            _log.warning(
                "lease.lost",
                task_id=task_id,
                case_id=task.case_id,
                worker_id=worker_id,
                attempt=attempt,
                stage=stage,
                error_code="TASK_LEASE_LOST",
            )
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

        advanced = session.execute(
            update(Case)
            .where(
                Case.case_id == case.case_id,
                Case.version == case.version,
                Case.active_task_id == task_id,
            )
            .values(
                status=to_state.value,
                version=case.version + 1,
                active_task_id=None,
            )
        )
        if rowcount(advanced) != 1:
            session.rollback()
            raise MediDiagError(
                "OPTIMISTIC_LOCK_CONFLICT",
                detail=f"failed stage {stage} case CAS conflict",
                context={"case_id": case.case_id, "task_id": task_id},
            )

        session.add(
            CaseEventLog(
                case_id=case.case_id,
                event_type="stage_failed",
                from_status=current_state.value,
                to_status=to_state.value,
                trigger_subject=subject.value,
                trigger_entity=worker_id,
                detail={
                    "task_id": task_id,
                    "stage": stage,
                    "attempt": attempt,
                    "error_code": error_code,
                    **(detail or {}),
                },
            )
        )
        session.commit()
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
        """在同一事务中完成租约 fence、病例 CAS、产物和事件写入。

        外部 IO 必须在调用前完成。病例 version 冲突最多重试 3 次；每次重试
        都重新校验任务租约和当前状态，失败事务不会写入 artifact 或事件。
        """
        last_error: MediDiagError | None = None
        retry_count = 0
        case_id: str | None = None

        for retry_count in range(OPTIMISTIC_LOCK_MAX_RETRIES + 1):
            task = session.execute(
                select(WorkflowTask).where(WorkflowTask.task_id == task_id)
            ).scalar_one_or_none()
            if task is None:
                raise MediDiagError("TASK_LEASE_LOST", detail=f"task {task_id} not found")
            case_id = task.case_id

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
            if rowcount(fenced) != 1:
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
            if rowcount(advanced) != 1:
                session.rollback()
                last_error = MediDiagError(
                    "OPTIMISTIC_LOCK_CONFLICT",
                    detail=f"stage {stage} case CAS conflict on retry {retry_count}",
                    context={
                        "case_id": case.case_id,
                        "task_id": task_id,
                        "retry_count": retry_count,
                    },
                )
                _log.warning(
                    "workflow.stage_cas_conflict",
                    case_id=case.case_id,
                    task_id=task_id,
                    worker_id=worker_id,
                    stage=stage,
                    retry_count=retry_count,
                    error_code="OPTIMISTIC_LOCK_CONFLICT",
                )
                if retry_count < OPTIMISTIC_LOCK_MAX_RETRIES:
                    time.sleep(OPTIMISTIC_LOCK_BACKOFF_MS[retry_count] / 1000.0)
                    continue
                break

            for record in records or []:
                session.add(record)
            event_detail = {
                "task_id": task_id,
                "stage": stage,
                "attempt": attempt,
                "optimistic_lock_retry_count": retry_count,
                **(detail or {}),
            }
            session.add(
                CaseEventLog(
                    case_id=case.case_id,
                    event_type="stage_completed",
                    from_status=current_state.value,
                    to_status=to_state.value,
                    trigger_subject=subject.value,
                    trigger_entity=worker_id,
                    detail=event_detail,
                )
            )
            session.commit()
            _log.info(
                "workflow.stage_committed",
                case_id=case.case_id,
                task_id=task_id,
                worker_id=worker_id,
                attempt=attempt,
                stage=stage,
                from_state=current_state.value,
                to_state=to_state.value,
                optimistic_lock_retry_count=retry_count,
            )
            return

        raise last_error or MediDiagError(
            "WORKFLOW_RETRY_EXCEEDED",
            detail=f"stage {stage} optimistic lock retry exhausted for case {case_id}",
        )

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
