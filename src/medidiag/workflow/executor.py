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

from sqlalchemy import select, update
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
    - 外部结果写入（二次校验租约，防脑裂双写）
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
        session.commit()
        return case

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
        # 1. 查找同幂等键的现有任务
        existing = session.execute(
            select(WorkflowTask).where(
                WorkflowTask.idempotency_key == idempotency_key
            )
        ).scalar_one_or_none()

        if existing:
            if existing.status in ("SUCCEEDED", "RUNNING", "PENDING"):
                return existing  # 返回已有任务（幂等）
            # FAILED/STALE 可以重新启动，继续往下

        # 2. 检查同 case 是否有其他 RUNNING 任务（不同幂等键）
        running = session.execute(
            select(WorkflowTask).where(
                WorkflowTask.case_id == case_id,
                WorkflowTask.status == "RUNNING",
            )
        ).scalar_one_or_none()

        if running and running.idempotency_key != idempotency_key:
            raise MediDiagError(
                "WORKFLOW_ALREADY_RUNNING",
                detail=f"case {case_id} already has RUNNING task {running.task_id}",
                context={"existing_task_id": running.task_id},
            )

        # 3. 创建新任务
        task = WorkflowTask(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            case_id=case_id,
            task_type=task_type,
            status="PENDING",
            input_hash=input_hash,
            idempotency_key=idempotency_key,
        )
        session.add(task)
        session.commit()
        return task

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

    # ===== 外部结果写入（二次校验租约）=====

    def write_external_result(
        self,
        session: Session,
        task_id: str,
        worker_id: str,
        attempt: int,
        result: dict,
    ) -> bool:
        """写入外部 IO 结果。

        写入前必须二次校验租约仍有效（防脑裂双写）。
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
        # 二次校验租约
        if self.lease.check_lease_lost(session, task_id, worker_id, attempt):
            # 租约丢失，记录事件（不写入结果）
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

        # 租约有效，写入结果
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task_id)
            .values(status="SUCCEEDED", result=result)
        )
        session.commit()
        return True
