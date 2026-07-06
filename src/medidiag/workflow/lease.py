"""worker 租约管理。

PLAN.md 租约策略:
- 默认租约 60s，worker 每 20s 续期一次
- 扫描器每 30s 查找 RUNNING 且 lease_until < now() 的任务
- 接管前校验任务对应 case 当前状态仍允许重入（非终态）
- worker 写入外部结果前必须二次校验租约
- 旧 worker 迟到写入被丢弃并记录 TASK_LEASE_LOST（防脑裂双写）

租约参数沿用 eval/config.yaml:
    lease_seconds: 60
    heartbeat_seconds: 20
    scan_interval_seconds: 30
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from medidiag.db.models import Case, WorkflowTask
from medidiag.workflow.state_machine import CaseState, is_terminal


def _utcnow() -> datetime:
    """UTC 当前时间（naive）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class LeaseManager:
    """租约管理器。

    处理租约的领取、续期、超时扫描、重入校验、脑裂防护。
    """

    def __init__(
        self,
        lease_seconds: int = 60,
        heartbeat_seconds: int = 20,
        scan_interval_seconds: int = 30,
    ) -> None:
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.scan_interval_seconds = scan_interval_seconds

    def acquire(
        self, session: Session, task_id: str, worker_id: str
    ) -> bool:
        """领取租约：将 PENDING 任务改为 RUNNING，设置 lease 字段。

        Returns:
            True 如果领取成功（任务从 PENDING 变为 RUNNING）。
        """
        now = _utcnow()
        result = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.status == "PENDING",
            )
            .values(
                status="RUNNING",
                lease_owner=worker_id,
                lease_until=now + timedelta(seconds=self.lease_seconds),
                heartbeat_at=now,
            )
        )
        session.commit()
        return result.rowcount == 1

    def renew(
        self, session: Session, task_id: str, worker_id: str
    ) -> bool:
        """续期租约：更新 lease_until 和 heartbeat_at。

        只有 lease_owner 匹配且任务仍 RUNNING 才能续期。

        Returns:
            True 如果续期成功。
        """
        now = _utcnow()
        result = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.lease_owner == worker_id,
                WorkflowTask.status == "RUNNING",
            )
            .values(
                lease_until=now + timedelta(seconds=self.lease_seconds),
                heartbeat_at=now,
            )
        )
        session.commit()
        return result.rowcount == 1

    def find_expired(self, session: Session) -> list[WorkflowTask]:
        """查找过期任务：RUNNING 且 lease_until < now()。"""
        now = _utcnow()
        result = session.execute(
            select(WorkflowTask).where(
                WorkflowTask.status == "RUNNING",
                WorkflowTask.lease_until < now,
            )
        )
        return list(result.scalars().all())

    def reclaim(
        self, session: Session, task_id: str, new_worker_id: str
    ) -> bool:
        """接管过期任务。

        接管前校验 case 状态仍允许重入（非终态）。
        旧任务标记为 STALE，创建新任务（attempt + 1）。

        Returns:
            True 如果接管成功。False 如果 case 已终态或任务不存在。
        """
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one_or_none()
        if task is None:
            return False

        # 校验 case 仍可重入
        case = session.execute(
            select(Case).where(Case.case_id == task.case_id)
        ).scalar_one_or_none()
        if case is None:
            return False
        if is_terminal(CaseState(case.status)):
            # case 已终态，旧任务标 STALE，不接管
            session.execute(
                update(WorkflowTask)
                .where(WorkflowTask.task_id == task_id)
                .values(status="STALE")
            )
            session.commit()
            return False

        # 标记旧任务为 STALE
        old_attempt = task.attempt
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task_id)
            .values(status="STALE")
        )

        # 创建新任务（attempt + 1）
        now = _utcnow()
        new_task = WorkflowTask(
            task_id=f"{task_id}_retry{old_attempt + 1}",
            case_id=task.case_id,
            task_type=task.task_type,
            status="RUNNING",
            lease_owner=new_worker_id,
            lease_until=now + timedelta(seconds=self.lease_seconds),
            heartbeat_at=now,
            attempt=old_attempt + 1,
            input_hash=task.input_hash,
            idempotency_key=f"{task.idempotency_key}_retry{old_attempt + 1}",
        )
        session.add(new_task)
        session.commit()
        return True

    def validate_lease(
        self,
        session: Session,
        task_id: str,
        worker_id: str,
        attempt: int,
    ) -> bool:
        """二次校验租约是否仍有效。

        worker 写入外部结果前必须调用此方法。
        条件: task_id + lease_owner + status=RUNNING + lease_until > now() + attempt 匹配。

        Returns:
            True 如果租约有效。
        """
        now = _utcnow()
        result = session.execute(
            select(WorkflowTask).where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.lease_owner == worker_id,
                WorkflowTask.status == "RUNNING",
                WorkflowTask.lease_until > now,
                WorkflowTask.attempt == attempt,
            )
        )
        return result.scalar_one_or_none() is not None

    def check_lease_lost(
        self,
        session: Session,
        task_id: str,
        worker_id: str,
        attempt: int,
    ) -> bool:
        """检查租约是否已丢失（旧 worker 迟到写入防护）。

        如果租约已失效，返回 True。
        调用方应丢弃写入并记录 TASK_LEASE_LOST。

        Returns:
            True 如果租约已丢失（写入应被丢弃）。
        """
        return not self.validate_lease(session, task_id, worker_id, attempt)
