"""阶段 3 一致性测试。

覆盖:
1. 幂等创建病例（重复请求返回已有）
2. 启动工作流幂等 + RUNNING 重复请求处理
3. 状态推进 + 乐观锁重试 + 状态/事件同事务
4. 租约管理（领取/续期/超时扫描/校验）
5. 旧 worker 迟到写入防护（TASK_LEASE_LOST）
6. 租约接管（reclaim，case 终态时不接管）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from medidiag.db.models import Case, CaseEventLog, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.idempotency import compute_input_hash, make_workflow_key
from medidiag.workflow.lease import LeaseManager
from medidiag.workflow.state_machine import CaseState, TriggerSubject


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@pytest.fixture
def engine():
    eng = create_db_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    factory = get_session_factory(engine)
    s = factory()
    yield s
    s.close()


@pytest.fixture
def executor():
    return WorkflowExecutor(LeaseManager(lease_seconds=60, heartbeat_seconds=20))


@pytest.fixture
def case(executor, session):
    return executor.create_case(session, "question", "ik1", "user1")


# ===== 幂等创建病例 =====


class TestCreateCaseIdempotency:
    def test_create_returns_new_case(self, executor, session) -> None:
        c = executor.create_case(session, "q1", "ik1", "u1")
        assert c.case_id.startswith("case_")
        assert c.status == "CREATED"
        assert c.version == 1

    def test_duplicate_returns_existing(self, executor, session) -> None:
        c1 = executor.create_case(session, "q1", "ik1", "u1")
        c2 = executor.create_case(session, "different", "ik1", "u1")
        assert c1.case_id == c2.case_id

    def test_different_scope_creates_new(self, executor, session) -> None:
        c1 = executor.create_case(session, "q1", "ik1", "u1")
        c2 = executor.create_case(session, "q1", "ik1", "u2")
        assert c1.case_id != c2.case_id

    def test_case_created_event_logged(self, executor, session) -> None:
        c = executor.create_case(session, "q1", "ik1", "u1")
        events = (
            session.query(CaseEventLog)
            .filter_by(case_id=c.case_id)
            .all()
        )
        assert len(events) == 1
        assert events[0].event_type == "case_created"


# ===== 启动工作流幂等 + RUNNING 重复请求 =====


class TestStartWorkflowIdempotency:
    def test_start_creates_pending_task(self, executor, session, case) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        assert task.status == "PENDING"
        assert task.task_type == "normalize"

    def test_duplicate_key_returns_existing(self, executor, session, case) -> None:
        t1 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        t2 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        assert t1.task_id == t2.task_id

    def test_different_key_running_raises_409(
        self, executor, session, case
    ) -> None:
        t1 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == t1.task_id)
            .values(status="RUNNING")
        )
        session.commit()
        with pytest.raises(MediDiagError) as exc:
            executor.start_workflow(
                session, case.case_id, "normalize", "wk2", "h2"
            )
        assert exc.value.code == "WORKFLOW_ALREADY_RUNNING"

    def test_different_key_pending_also_conflicts(
        self, executor, session, case
    ) -> None:
        executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        with pytest.raises(MediDiagError) as exc:
            executor.start_workflow(
                session, case.case_id, "normalize", "wk2", "h2"
            )
        assert exc.value.code == "WORKFLOW_ALREADY_RUNNING"
        assert session.query(WorkflowTask).count() == 1

    def test_succeeded_returns_existing(self, executor, session, case) -> None:
        t1 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == t1.task_id)
            .values(status="SUCCEEDED", result={"ok": True})
        )
        session.commit()
        t2 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        assert t1.task_id == t2.task_id

    def test_failed_task_can_restart(self, executor, session, case) -> None:
        """FAILED/STALE 任务可以重新启动。"""
        t1 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == t1.task_id)
            .values(status="FAILED")
        )
        session.commit()
        # 同一逻辑任务复用 task_id，并递增 attempt。
        t2 = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        assert t2.task_id == t1.task_id
        assert t2.status == "PENDING"
        assert t2.attempt == 1

    def test_stale_versions_only_one_can_claim_active_task(
        self, tmp_path
    ) -> None:
        database_path = tmp_path / "active-task-cas.db"
        engine = create_db_engine(f"sqlite:///{database_path.as_posix()}")
        init_db(engine)
        factory = get_session_factory(engine)
        creator = factory()
        case = WorkflowExecutor().create_case(
            creator, "question", "case-key", "scope"
        )
        case_id = case.case_id
        creator.close()

        session_a = factory()
        session_b = factory()
        stale_a = session_a.execute(
            select(Case).where(Case.case_id == case_id)
        ).scalar_one()
        stale_b = session_b.execute(
            select(Case).where(Case.case_id == case_id)
        ).scalar_one()
        assert stale_a.version == stale_b.version == 1

        winner = session_a.execute(
            update(Case)
            .where(
                Case.case_id == case_id,
                Case.version == stale_a.version,
                Case.active_task_id.is_(None),
            )
            .values(active_task_id="task-a", version=2)
        )
        session_a.commit()
        loser = session_b.execute(
            update(Case)
            .where(
                Case.case_id == case_id,
                Case.version == stale_b.version,
                Case.active_task_id.is_(None),
            )
            .values(active_task_id="task-b", version=2)
        )
        session_b.commit()

        assert winner.rowcount == 1
        assert loser.rowcount == 0
        persisted = session_b.execute(
            select(Case).where(Case.case_id == case_id)
        ).scalar_one()
        session_b.refresh(persisted)
        assert persisted.active_task_id == "task-a"
        session_a.close()
        session_b.close()
        engine.dispose()


# ===== 状态推进 + 乐观锁重试 =====


class TestAdvanceState:
    def test_legal_transition(self, executor, session, case) -> None:
        executor.advance_state(
            session, case.case_id, CaseState.NORMALIZED,
            TriggerSubject.API, "api",
        )
        session.refresh(case)
        assert case.status == "NORMALIZED"
        assert case.version == 2

    def test_state_and_event_same_transaction(
        self, executor, session, case
    ) -> None:
        executor.advance_state(
            session, case.case_id, CaseState.NORMALIZED,
            TriggerSubject.API, "api",
        )
        events = (
            session.query(CaseEventLog)
            .filter_by(case_id=case.case_id, event_type="state_transition")
            .all()
        )
        assert len(events) == 1
        assert events[0].from_status == "CREATED"
        assert events[0].to_status == "NORMALIZED"

    def test_illegal_transition_raises(self, executor, session, case) -> None:
        with pytest.raises(MediDiagError) as exc:
            executor.advance_state(
                session, case.case_id, CaseState.PLAN_GENERATED,
                TriggerSubject.WORKER,
            )
        assert exc.value.code == "ILLEGAL_STATE_TRANSITION"

    def test_terminal_cannot_advance(self, executor, session, case) -> None:
        executor.advance_state(
            session, case.case_id, CaseState.CLOSED_CANCELLED, TriggerSubject.API
        )
        with pytest.raises(MediDiagError) as exc:
            executor.advance_state(
                session, case.case_id, CaseState.NORMALIZED, TriggerSubject.API
            )
        assert exc.value.code == "ILLEGAL_STATE_TRANSITION"

    def test_idempotent_advance(self, executor, session, case) -> None:
        """重复推进到同一状态是幂等的（version 只增一次）。"""
        executor.advance_state(
            session, case.case_id, CaseState.NORMALIZED, TriggerSubject.API
        )
        executor.advance_state(
            session, case.case_id, CaseState.NORMALIZED, TriggerSubject.API
        )
        session.refresh(case)
        assert case.status == "NORMALIZED"
        assert case.version == 2

    def test_full_happy_path(self, executor, session, case) -> None:
        """完整主链路状态推进。"""
        steps = [
            (CaseState.NORMALIZED, TriggerSubject.API),
            (CaseState.EVIDENCE_RETRIEVED, TriggerSubject.WORKER),
            (CaseState.PLAN_GENERATED, TriggerSubject.WORKER),
            (CaseState.SPECIALIST_REVIEWING, TriggerSubject.AGENT_WORKER),
            (CaseState.ARBITRATION_REVIEWING, TriggerSubject.AGENT_WORKER),
            (CaseState.APPROVED, TriggerSubject.REVIEWER_WORKER),
            (CaseState.REPORT_GENERATED, TriggerSubject.REVIEWER_WORKER),
            (CaseState.CLOSED_SUCCESS, TriggerSubject.WORKER),
        ]
        for to_state, subject in steps:
            executor.advance_state(
                session, case.case_id, to_state, subject, "entity"
            )
        session.refresh(case)
        assert case.status == "CLOSED_SUCCESS"
        assert case.version == 9  # 1 初始 + 8 次推进


# ===== 租约管理 =====


class TestLeaseManager:
    def _create_pending_task(self, session, case) -> WorkflowTask:
        task = WorkflowTask(
            task_id="t1", case_id=case.case_id, task_type="normalize",
            status="PENDING", input_hash="h1", idempotency_key="ik1",
        )
        session.add(task)
        session.commit()
        return task

    def test_acquire_lease(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        assert lm.acquire(session, "t1", "worker-1") is True
        task = session.get(WorkflowTask, 1)
        assert task.status == "RUNNING"
        assert task.lease_owner == "worker-1"
        assert task.lease_until is not None

    def test_acquire_already_running_fails(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        lm.acquire(session, "t1", "worker-1")
        assert lm.acquire(session, "t1", "worker-2") is False

    def test_renew_lease(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        lm.acquire(session, "t1", "worker-1")
        assert lm.renew(session, "t1", "worker-1", 0) is True

    def test_renew_wrong_owner_fails(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        lm.acquire(session, "t1", "worker-1")
        assert lm.renew(session, "t1", "worker-2", 0) is False

    def test_renew_wrong_attempt_fails(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        lm.acquire(session, "t1", "worker-1")
        assert lm.renew(session, "t1", "worker-1", 1) is False

    def test_expired_lease_cannot_be_revived(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        lm.acquire(session, "t1", "worker-1")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == "t1")
            .values(lease_until=_utcnow() - timedelta(seconds=1))
        )
        session.commit()
        assert lm.renew(session, "t1", "worker-1", 0) is False

    def test_find_expired(self, session, case) -> None:
        now = _utcnow()
        task = WorkflowTask(
            task_id="t1", case_id=case.case_id, task_type="normalize",
            status="RUNNING", lease_owner="worker-1",
            lease_until=now - timedelta(seconds=10),
            heartbeat_at=now - timedelta(seconds=30),
            input_hash="h1", idempotency_key="ik1",
        )
        session.add(task)
        session.commit()
        lm = LeaseManager()
        expired = lm.find_expired(session)
        assert len(expired) == 1
        assert expired[0].task_id == "t1"

    def test_validate_lease(self, session, case) -> None:
        self._create_pending_task(session, case)
        lm = LeaseManager()
        lm.acquire(session, "t1", "worker-1")
        assert lm.validate_lease(session, "t1", "worker-1", 0) is True
        assert lm.validate_lease(session, "t1", "worker-2", 0) is False
        assert lm.validate_lease(session, "t1", "worker-1", 1) is False


# ===== 旧 worker 迟到写入防护 =====


class TestLeaseLostProtection:
    def test_write_with_valid_lease(self, executor, session, case) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        result = executor.write_external_result(
            session, task.task_id, "worker-1", 0, {"data": "ok"}
        )
        assert result is True
        session.refresh(task)
        assert task.status == "SUCCEEDED"
        session.refresh(case)
        assert case.active_task_id is None

    def test_write_with_lost_lease_raises(
        self, executor, session, case
    ) -> None:
        """旧 worker 租约被接管后，迟到写入抛 TASK_LEASE_LOST。"""
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        # 模拟 worker-2 接管（lease_owner 变更）
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task.task_id)
            .values(lease_owner="worker-2")
        )
        session.commit()
        # worker-1 迟到写入应被拒绝
        with pytest.raises(MediDiagError) as exc:
            executor.write_external_result(
                session, task.task_id, "worker-1", 0, {"data": "stale"}
            )
        assert exc.value.code == "TASK_LEASE_LOST"
        # 结果不应被写入
        session.refresh(task)
        assert task.status == "RUNNING"
        # 应记录 lease_lost 事件
        events = (
            session.query(CaseEventLog)
            .filter_by(case_id=case.case_id, event_type="lease_lost")
            .all()
        )
        assert len(events) == 1

    def test_expired_worker_result_is_atomically_rejected(
        self, executor, session, case
    ) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task.task_id)
            .values(lease_until=_utcnow() - timedelta(seconds=1))
        )
        session.commit()
        with pytest.raises(MediDiagError) as exc:
            executor.write_external_result(
                session, task.task_id, "worker-1", 0, {"stale": True}
            )
        assert exc.value.code == "TASK_LEASE_LOST"
        session.refresh(task)
        assert task.status == "RUNNING"
        assert task.result is None

    def test_result_rolls_back_when_case_no_longer_points_to_task(
        self, executor, session, case
    ) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        session.execute(
            update(Case)
            .where(Case.case_id == case.case_id)
            .values(active_task_id="other-task", version=Case.version + 1)
        )
        session.commit()

        with pytest.raises(MediDiagError) as exc:
            executor.write_external_result(
                session, task.task_id, "worker-1", 0, {"should": "rollback"}
            )
        assert exc.value.code == "TASK_LEASE_LOST"
        session.refresh(task)
        assert task.status == "RUNNING"
        assert task.result is None


# ===== 租约接管 =====


class TestLeaseReclaim:
    def test_reclaim_expired_task(self, executor, session, case) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task.task_id)
            .values(lease_until=_utcnow() - timedelta(seconds=10))
        )
        session.commit()
        lm = LeaseManager()
        assert lm.reclaim(session, task.task_id, "worker-2") is True
        session.refresh(task)
        assert task.status == "RUNNING"
        assert task.lease_owner == "worker-2"
        assert task.attempt == 1
        assert session.query(WorkflowTask).count() == 1

    def test_only_one_reclaim_wins(self, executor, session, case) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task.task_id)
            .values(lease_until=_utcnow() - timedelta(seconds=10))
        )
        session.commit()
        lm = LeaseManager()
        assert lm.reclaim(session, task.task_id, "worker-2") is True
        assert lm.reclaim(session, task.task_id, "worker-3") is False
        session.refresh(task)
        assert task.lease_owner == "worker-2"
        assert task.attempt == 1

    def test_old_worker_result_is_rejected_after_reclaim(
        self, executor, session, case
    ) -> None:
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task.task_id)
            .values(lease_until=_utcnow() - timedelta(seconds=10))
        )
        session.commit()
        assert executor.lease.reclaim(session, task.task_id, "worker-2") is True

        with pytest.raises(MediDiagError) as exc:
            executor.write_external_result(
                session, task.task_id, "worker-1", 0, {"stale": True}
            )
        assert exc.value.code == "TASK_LEASE_LOST"
        session.refresh(task)
        assert task.status == "RUNNING"
        assert task.lease_owner == "worker-2"
        assert task.attempt == 1
        assert task.result is None

    def test_reclaim_terminal_case_not_reclaimed(
        self, executor, session, case
    ) -> None:
        """case 已终态时，过期任务标 STALE 但不接管。"""
        task = executor.start_workflow(
            session, case.case_id, "normalize", "wk1", "h1"
        )
        executor.lease.acquire(session, task.task_id, "worker-1")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task.task_id)
            .values(lease_until=_utcnow() - timedelta(seconds=10))
        )
        session.commit()
        executor.advance_state(
            session, case.case_id, CaseState.CLOSED_CANCELLED, TriggerSubject.API
        )
        lm = LeaseManager()
        assert lm.reclaim(session, task.task_id, "worker-2") is False
        session.refresh(task)
        assert task.status == "STALE"
        session.refresh(case)
        assert case.active_task_id is None


# ===== 幂等键工具 =====


class TestIdempotencyUtils:
    def test_compute_input_hash_stable(self) -> None:
        """相同内容（不同 key 顺序）产生相同哈希。"""
        h1 = compute_input_hash({"a": 1, "b": 2})
        h2 = compute_input_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_compute_input_hash_different(self) -> None:
        h1 = compute_input_hash({"a": 1})
        h2 = compute_input_hash({"a": 2})
        assert h1 != h2

    def test_make_workflow_key(self) -> None:
        key = make_workflow_key("case_001", "normalize")
        assert key == "workflow:case_001:normalize"
