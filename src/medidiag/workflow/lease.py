"""worker 租约管理。

PLAN.md 租约策略:
- 默认租约 60s，worker 每 20s 续期一次
- 扫描器每 30s 查找 RUNNING 且 lease_until < now() 的任务
- 接管前校验任务对应 case 当前状态仍允许重入（非终态）
- worker 写入外部结果时必须把 lease 条件合并进同一条 UPDATE
- 旧 worker 迟到写入被丢弃并记录 TASK_LEASE_LOST（防脑裂双写）

租约参数沿用 eval/config.yaml:
    lease_seconds: 60
    heartbeat_seconds: 20
    scan_interval_seconds: 30
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from medidiag.db.models import Case, CaseEventLog, WorkflowTask
from medidiag.observability.logging import get_logger
from medidiag.workflow.state_machine import CaseState, is_terminal

_log = get_logger(__name__)


def _utcnow() -> datetime:
    """UTC 当前时间（naive）。"""
    return datetime.now(UTC).replace(tzinfo=None)


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

    def now(self) -> datetime:
        """Return the database-comparable UTC timestamp used by CAS predicates."""
        return _utcnow()

    def acquire(
        self, session: Session, task_id: str, worker_id: str
    ) -> bool:
        """领取租约：将 PENDING 任务改为 RUNNING，设置 lease 字段。

        Returns:
            True 如果领取成功（任务从 PENDING 变为 RUNNING）。
        """
        now = self.now()
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
        if result.rowcount == 1:
            task = session.execute(
                select(WorkflowTask).where(WorkflowTask.task_id == task_id)
            ).scalar_one()
            session.add(
                CaseEventLog(
                    case_id=task.case_id,
                    event_type="lease_acquired",
                    trigger_subject="worker",
                    trigger_entity=worker_id,
                    detail={"task_id": task_id, "attempt": task.attempt},
                )
            )
            _log.info(
                "lease.acquired",
                task_id=task_id,
                case_id=task.case_id,
                worker_id=worker_id,
                attempt=task.attempt,
                lease_seconds=self.lease_seconds,
            )
        else:
            _log.debug("lease.acquire_rejected", task_id=task_id, worker_id=worker_id)
        session.commit()
        return result.rowcount == 1

    def renew(
        self,
        session: Session,
        task_id: str,
        worker_id: str,
        attempt: int,
    ) -> bool:
        """续期租约：更新 lease_until 和 heartbeat_at。

        只有 task/owner/attempt 匹配、任务仍 RUNNING 且原租约未过期才能续期。

        Returns:
            True 如果续期成功。
        """
        now = self.now()
        result = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.lease_owner == worker_id,
                WorkflowTask.attempt == attempt,
                WorkflowTask.status == "RUNNING",
                WorkflowTask.lease_until > now,
            )
            .values(
                lease_until=now + timedelta(seconds=self.lease_seconds),
                heartbeat_at=now,
            )
        )
        session.commit()
        renewed = result.rowcount == 1
        if renewed:
            _log.debug(
                "lease.renewed",
                task_id=task_id,
                worker_id=worker_id,
                attempt=attempt,
                lease_seconds=self.lease_seconds,
            )
        else:
            # 续期失败等于本 worker 已不再持有租约，调用方必须停止写入。
            _log.warning(
                "lease.renew_rejected",
                task_id=task_id,
                worker_id=worker_id,
                attempt=attempt,
                error_code="TASK_LEASE_LOST",
            )
        return renewed

    def find_expired(self, session: Session) -> list[WorkflowTask]:
        """查找过期任务：RUNNING 且 lease_until < now()。"""
        now = self.now()
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

        接管前校验 case 状态仍允许重入（非终态），然后用一条条件
        UPDATE 原子更新 owner/lease 并递增 attempt。并发扫描器只有一个
        能命中旧 attempt。

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
        now = self.now()
        if is_terminal(CaseState(case.status)):
            stale = session.execute(
                update(WorkflowTask)
                .where(
                    WorkflowTask.task_id == task_id,
                    WorkflowTask.status == "RUNNING",
                    WorkflowTask.attempt == task.attempt,
                    WorkflowTask.lease_until <= now,
                )
                .values(status="STALE")
            )
            if stale.rowcount == 1:
                _log.info(
                    "lease.stale_marked",
                    task_id=task_id,
                    case_id=case.case_id,
                    case_status=case.status,
                    reason="case_terminal",
                )
                session.execute(
                    update(Case)
                    .where(
                        Case.case_id == case.case_id,
                        Case.active_task_id == task_id,
                    )
                    .values(active_task_id=None, version=Case.version + 1)
                )
            session.commit()
            return False

        old_owner = task.lease_owner
        old_attempt = task.attempt
        eligible_case = select(Case.case_id).where(
            Case.case_id == task.case_id,
            Case.version == case.version,
            Case.active_task_id == task_id,
            Case.status.not_in([state.value for state in CaseState if is_terminal(state)]),
        )
        reclaimed = session.execute(
            update(WorkflowTask)
            .where(
                WorkflowTask.task_id == task_id,
                WorkflowTask.case_id.in_(eligible_case),
                WorkflowTask.status == "RUNNING",
                WorkflowTask.attempt == old_attempt,
                WorkflowTask.lease_until <= now,
            )
            .values(
                lease_owner=new_worker_id,
                lease_until=now + timedelta(seconds=self.lease_seconds),
                heartbeat_at=now,
                attempt=old_attempt + 1,
            )
        )
        if reclaimed.rowcount == 1:
            session.add(
                CaseEventLog(
                    case_id=task.case_id,
                    event_type="lease_reclaimed",
                    trigger_subject="system",
                    trigger_entity=new_worker_id,
                    detail={
                        "task_id": task_id,
                        "old_owner": old_owner,
                        "new_owner": new_worker_id,
                        "old_attempt": old_attempt,
                        "new_attempt": old_attempt + 1,
                    },
                )
            )
            _log.warning(
                "lease.reclaimed",
                task_id=task_id,
                case_id=task.case_id,
                old_owner=old_owner,
                new_owner=new_worker_id,
                old_attempt=old_attempt,
                new_attempt=old_attempt + 1,
                error_code="TASK_LEASE_EXPIRED",
            )
        session.commit()
        return reclaimed.rowcount == 1

    def validate_lease(
        self,
        session: Session,
        task_id: str,
        worker_id: str,
        attempt: int,
    ) -> bool:
        """只读检查租约是否仍有效（诊断/监控用途）。

        结果写入不能依赖该先查后写方法，必须使用原子条件 UPDATE。
        条件: task_id + lease_owner + status=RUNNING + lease_until > now() + attempt 匹配。

        Returns:
            True 如果租约有效。
        """
        now = self.now()
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
