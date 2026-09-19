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

import threading
import time
from datetime import UTC, datetime, timedelta
from types import TracebackType

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from medidiag.config import get_settings
from medidiag.db.models import Case, CaseEventLog, WorkflowTask
from medidiag.db.session import rowcount
from medidiag.observability.logging import get_logger
from medidiag.workflow.state_machine import CaseState, is_terminal

_log = get_logger(__name__)


def _utcnow() -> datetime:
    """UTC 当前时间（naive）。"""
    return datetime.now(UTC).replace(tzinfo=None)


class LeaseManager:
    """租约管理器。

    处理租约的领取、续期、超时扫描、重入校验、脑裂防护。

    参数默认值取自 ``Settings``（``MEDIDIAG_LEASE_SECONDS`` /
    ``MEDIDIAG_HEARTBEAT_SECONDS`` / ``MEDIDIAG_LEASE_SCAN_SECONDS``），
    显式传参优先。此前三个环境变量中只有 scan 被真正读取，另外两个是声明了
    但不起作用的配置项。

    外部 IO 期间的续期由 :class:`LeaseHeartbeat` 承担（见 DD-020）：阶段之间
    仍由 ``SingleMachineWorker._renew`` 续期，单次 provider 调用期间则由心跳
    线程用独立会话续期，因此「租约必须长于最慢的单个阶段」这一运行边界不再成立。
    """

    def __init__(
        self,
        lease_seconds: int | None = None,
        heartbeat_seconds: int | None = None,
        scan_interval_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self.lease_seconds = int(
            lease_seconds
            if lease_seconds is not None
            else settings.medidiag_lease_seconds
        )
        self.heartbeat_seconds = int(
            heartbeat_seconds
            if heartbeat_seconds is not None
            else settings.medidiag_heartbeat_seconds
        )
        self.scan_interval_seconds = int(
            scan_interval_seconds
            if scan_interval_seconds is not None
            else settings.medidiag_lease_scan_seconds
        )
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be at least 1")
        if not 1 <= self.heartbeat_seconds < self.lease_seconds:
            # 续期节奏不短于租约时长时，租约必然在续期之前过期。
            raise ValueError(
                "heartbeat_seconds must be at least 1 and shorter than lease_seconds"
            )
        if self.scan_interval_seconds < 1:
            raise ValueError("scan_interval_seconds must be at least 1")

    def now(self) -> datetime:
        """返回 CAS 条件使用的、可与数据库比较的 UTC 时间戳。"""
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
        if rowcount(result) == 1:
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
        return rowcount(result) == 1

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
        renewed = rowcount(result) == 1
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
            if rowcount(stale) == 1:
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
        if rowcount(reclaimed) == 1:
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
        return rowcount(reclaimed) == 1

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


# 心跳最多续期到 `HEARTBEAT_MAX_LEASE_PERIODS * lease_seconds`。这个上界是刻意
# 保留的：无上界的心跳会让一个卡死的 provider 调用永远续期下去，任务再也不会被
# 扫描器接管——那是把「浪费一次调用」换成了「永久卡住」，比原来的缺口更糟。
HEARTBEAT_MAX_LEASE_PERIODS = 10


class LeaseHeartbeat:
    """provider 调用期间用后台线程续期租约（DD-020）。

    此前租约只在阶段之间续期，单个阶段的 provider 调用超过 ``lease_seconds``
    时，扫描器会在该阶段仍在执行时接管任务；旧 worker 的写入随后被
    ``commit_stage`` 的条件 UPDATE 拒绝——不会脑裂双写，但会白白浪费一次调用。
    默认配置的边际很窄：60s 租约对 60s provider 超时，加上有界重试后必然越界。

    **使用独立会话**：worker 的 ``Session`` 归 worker 线程所有，SQLAlchemy 的
    Session 不是线程安全的，而且外部 IO 期间 worker 事务的状态不该被另一个线程
    改写。心跳从 ``session_factory`` 自取会话并在每次续期后提交。跨线程写同一行
    由 DD-021 的 ``journal_mode=WAL`` 与 ``busy_timeout`` 兜底。

    用法::

        with LeaseHeartbeat(factory, lease, task_id=..., worker_id=..., attempt=...) as hb:
            outcome = call_runner.call(...)
        if hb.lease_lost:
            ...  # 本 worker 已不再持有租约，不得写入
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        lease: LeaseManager,
        *,
        task_id: str,
        worker_id: str,
        attempt: int,
        max_seconds: float | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.lease = lease
        self.task_id = task_id
        self.worker_id = worker_id
        self.attempt = attempt
        self.max_seconds = (
            float(max_seconds)
            if max_seconds is not None
            else float(HEARTBEAT_MAX_LEASE_PERIODS * lease.lease_seconds)
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease_lost = False
        self._gave_up = False
        self._renewals = 0

    @property
    def lease_lost(self) -> bool:
        """续期被拒（另一个 worker 已接管），本 worker 不得再写入。"""
        return self._lease_lost

    @property
    def gave_up(self) -> bool:
        """超过 ``max_seconds`` 后主动停止续期，任务将重新变为可接管。"""
        return self._gave_up

    @property
    def renewals(self) -> int:
        """成功续期次数（测试与诊断用）。"""
        return self._renewals

    def __enter__(self) -> LeaseHeartbeat:
        self._thread = threading.Thread(
            target=self._loop,
            name=f"lease-heartbeat-{self.task_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            # 心跳只在 wait 上阻塞，join 不会等到一次完整的续期间隔。
            self._thread.join(timeout=self.lease.lease_seconds)
            self._thread = None

    def _loop(self) -> None:
        started = time.monotonic()
        interval = float(self.lease.heartbeat_seconds)
        while not self._stop.wait(interval):
            if time.monotonic() - started >= self.max_seconds:
                self._gave_up = True
                _log.error(
                    "lease.heartbeat_gave_up",
                    task_id=self.task_id,
                    worker_id=self.worker_id,
                    attempt=self.attempt,
                    max_seconds=self.max_seconds,
                    renewals=self._renewals,
                    error_code="TASK_LEASE_LOST",
                )
                return
            try:
                with self.session_factory() as session:
                    renewed = self.lease.renew(
                        session, self.task_id, self.worker_id, self.attempt
                    )
            except Exception:  # noqa: BLE001 - 心跳线程不得让 worker 崩溃
                # 续期出错与续期被拒同等对待：本 worker 不能再假定持有租约。
                _log.exception(
                    "lease.heartbeat_failed",
                    task_id=self.task_id,
                    worker_id=self.worker_id,
                    attempt=self.attempt,
                )
                self._lease_lost = True
                return
            if not renewed:
                self._lease_lost = True
                return
            self._renewals += 1
