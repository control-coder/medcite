"""P0-C 单机 worker 和恢复测试。"""

from __future__ import annotations

import time
import types
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select, update

from medidiag.compliance.status import ComplianceStatus
from medidiag.db.models import (
    Case,
    CaseEventLog,
    CaseReport,
    StageArtifact,
    WorkflowTask,
)
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.lease import LeaseHeartbeat, LeaseManager
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.provider_runtime import ProviderCallRunner, ProviderResponse
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker


@pytest.fixture
def runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'worker.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    yield engine, factory
    engine.dispose()


def _create_task(factory, key: str = "workflow-1") -> tuple[str, str]:
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session,
            "Deidentified simulated case for workflow engineering tests.",
            "case-1",
            "test-scope",
        )
        task = executor.start_workflow(
            session, case.case_id, "case_workflow", key, "input-hash"
        )
        return case.case_id, task.task_id


def test_worker_persists_full_success_path(runtime) -> None:
    _, factory = runtime
    case_id, task_id = _create_task(factory)
    result = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="worker-1"
    ).run_once()
    assert result.processed is True
    assert result.final_state == "CLOSED_SUCCESS"

    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert case.status == "CLOSED_SUCCESS"
        assert case.active_task_id is None
        assert task.status == "SUCCEEDED"
        stages = {
            item.stage
            for item in session.execute(
                select(StageArtifact).where(StageArtifact.case_id == case_id)
            ).scalars()
        }
        assert stages == {
            "normalize", "retrieval", "plan", "generation",
            "arbitration", "review", "report",
        }
        report = session.execute(
            select(CaseReport).where(CaseReport.case_id == case_id)
        ).scalar_one()
        assert "不构成医疗建议" in report.structured_report["disclaimer"]
        assert report.risk_warnings == ["qualified_clinician_review_required"]


def test_worker_escalates_and_waits_for_human(runtime) -> None:
    _, factory = runtime
    case_id, task_id = _create_task(factory)
    result = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(review_verdict="ESCALATED"),
        worker_id="worker-1",
    ).run_once()
    assert result.final_state == "ESCALATED"
    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert case.status == "ESCALATED"
        assert case.active_task_id is None
        assert task.status == "SUCCEEDED"
        assert session.query(CaseReport).count() == 0


def test_scanner_reclaims_and_worker_resumes(runtime) -> None:
    _, factory = runtime
    case_id, task_id = _create_task(factory)
    executor = WorkflowExecutor()
    with factory() as session:
        assert executor.lease.acquire(session, task_id, "crashed-worker") is True
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task_id)
            .values(lease_until=executor.lease.now() - timedelta(seconds=1))
        )
        session.commit()

    scanner = LeaseScanner(factory, recovery_worker_id="local-worker")
    assert scanner.scan_once() == [task_id]
    result = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="local-worker"
    ).run_once()
    assert result.final_state == "CLOSED_SUCCESS"
    with factory() as session:
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert task.attempt == 1
        assert task.status == "SUCCEEDED"
        assert session.execute(
            select(Case).where(Case.case_id == case_id)
        ).scalar_one().status == "CLOSED_SUCCESS"
        reclaimed_event = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "lease_reclaimed",
            )
        ).scalar_one()
        assert reclaimed_event.detail["old_owner"] == "crashed-worker"
        assert reclaimed_event.detail["new_attempt"] == 1


def test_stage_events_include_worker_and_attempt(runtime) -> None:
    _, factory = runtime
    case_id, _ = _create_task(factory)
    SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="worker-audit"
    ).run_once()
    with factory() as session:
        events = list(session.execute(
            select(CaseEventLog)
            .where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_completed",
            )
            .order_by(CaseEventLog.id)
        ).scalars())
        assert len(events) == 8
        assert all(event.trigger_entity == "worker-audit" for event in events)
        assert all("attempt" in event.detail for event in events)


def test_worker_audits_provider_retry_and_request_id(runtime) -> None:
    _, factory = runtime
    case_id, _ = _create_task(factory, "provider-retry")

    class RateLimitedOnceProvider(DeterministicWorkflowProvider):
        calls = 0

        def retrieve(self, normalized_query: str):
            self.calls += 1
            if self.calls == 1:
                request = httpx.Request("POST", "https://provider.invalid/retrieve")
                response = httpx.Response(
                    429,
                    request=request,
                    headers={"x-request-id": "req-rate-limited"},
                )
                raise httpx.HTTPStatusError(
                    "rate limited", request=request, response=response
                )
            return ProviderResponse(
                super().retrieve(normalized_query), request_id="req-retrieval-ok"
            )

    worker = SingleMachineWorker(
        factory,
        RateLimitedOnceProvider(),
        worker_id="provider-audit-worker",
        call_runner=ProviderCallRunner(sleep=lambda _: None),
    )
    assert worker.run_once().final_state == "CLOSED_SUCCESS"

    with factory() as session:
        events = session.execute(
            select(CaseEventLog)
            .where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "provider_call",
            )
            .order_by(CaseEventLog.id)
        ).scalars().all()
        retrieval = [
            event.detail for event in events if event.detail["stage"] == "retrieval"
        ]
        assert [item["retry_decision"] for item in retrieval] == [
            "retry", "not_needed"
        ]
        assert [item["provider_request_id"] for item in retrieval] == [
            "req-rate-limited", "req-retrieval-ok"
        ]
        assert all(item["trace_id"] for item in retrieval)

        completed = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_completed",
            )
        ).scalars().all()
        retrieval_completed = next(
            event for event in completed if event.detail["stage"] == "retrieval"
        )
        assert retrieval_completed.detail["provider_request_id"] == "req-retrieval-ok"
        assert retrieval_completed.detail["provider_retry_count"] == 1


def test_provider_crash_after_normalize_recovers_from_persisted_stage(runtime) -> None:
    class FailingRetrieveProvider(DeterministicWorkflowProvider):
        def retrieve(self, normalized_query: str) -> dict:
            raise RuntimeError("simulated retrieval crash")

    _, factory = runtime
    case_id, task_id = _create_task(factory)
    with pytest.raises(RuntimeError, match="simulated retrieval crash"):
        SingleMachineWorker(
            factory, FailingRetrieveProvider(), worker_id="crashed-worker"
        ).run_once()

    executor = WorkflowExecutor()
    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert case.status == "NORMALIZED"
        assert task.status == "RUNNING"
        assert [
            item.stage for item in session.execute(
                select(StageArtifact).where(StageArtifact.case_id == case_id)
            ).scalars()
        ] == ["normalize"]
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task_id)
            .values(lease_until=executor.lease.now() - timedelta(seconds=1))
        )
        session.commit()

    assert LeaseScanner(
        factory, recovery_worker_id="recovery-worker"
    ).scan_once() == [task_id]
    result = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="recovery-worker"
    ).run_once()
    assert result.final_state == "CLOSED_SUCCESS"
    with factory() as session:
        normalize_artifacts = list(session.execute(
            select(StageArtifact).where(
                StageArtifact.case_id == case_id,
                StageArtifact.stage == "normalize",
            )
        ).scalars())
        assert len(normalize_artifacts) == 1


def test_generation_transport_error_escalates_with_network_error_code(runtime) -> None:
    """DNS/连接失败必须被分类，不能作为原始 httpx 错误直接传播。"""

    class UnreachableGenerationProvider(DeterministicWorkflowProvider):
        def generate(self, question: str, retrieval: dict, plan: dict):
            raise httpx.ConnectError(
                "getaddrinfo failed",
                request=httpx.Request("POST", "https://provider.invalid/v1/chat"),
            )

    _, factory = runtime
    case_id, task_id = _create_task(factory, "generation-network-error")
    worker = SingleMachineWorker(
        factory,
        UnreachableGenerationProvider(),
        worker_id="network-worker",
        call_runner=ProviderCallRunner(sleep=lambda _: None),
    )

    result = worker.run_once()
    assert result.processed is True
    assert result.final_state == "ESCALATED"

    with factory() as session:
        attempts = [
            event.detail
            for event in session.execute(
                select(CaseEventLog)
                .where(
                    CaseEventLog.case_id == case_id,
                    CaseEventLog.event_type == "provider_call",
                )
                .order_by(CaseEventLog.id)
            ).scalars()
            if event.detail["stage"] == "generation"
        ]
        assert len(attempts) == 3
        assert all(item["error_code"] == "PROVIDER_NETWORK_ERROR" for item in attempts)
        assert [item["retry_decision"] for item in attempts] == [
            "retry", "retry", "exhausted"
        ]

        failed = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_failed",
            )
        ).scalars().all()
        assert [event.detail["stage"] for event in failed] == ["generation"]
        assert failed[0].detail["error_code"] == "PROVIDER_NETWORK_ERROR"

        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == "ESCALATED"
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert task.status == "FAILED"


_STAGE_METHODS = {
    "normalize": "normalize",
    "retrieval": "retrieve",
    "plan": "plan",
    "generation": "generate",
    "arbitration": "arbitrate",
    "review": "review",
    "report": "report",
}


@pytest.mark.parametrize("stage", sorted(_STAGE_METHODS))
def test_every_stage_failure_escalates_without_killing_the_worker(runtime, stage) -> None:
    """任何阶段都不能让 Provider 失败从 run_once() 直接传播出去。"""
    method = _STAGE_METHODS[stage]

    class FailingStageProvider(DeterministicWorkflowProvider):
        pass

    def _fail(*args, **kwargs):
        raise httpx.ConnectError(
            "provider unreachable",
            request=httpx.Request("POST", "https://provider.invalid/v1/call"),
        )

    setattr(FailingStageProvider, method, _fail)

    _, factory = runtime
    case_id, task_id = _create_task(factory, f"stage-failure-{stage}")
    worker = SingleMachineWorker(
        factory,
        FailingStageProvider(),
        worker_id=f"{stage}-worker",
        call_runner=ProviderCallRunner(sleep=lambda _: None),
    )

    result = worker.run_once()
    assert result.processed is True
    assert result.final_state == "ESCALATED"

    with factory() as session:
        failed = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_failed",
            )
        ).scalars().all()
        assert [event.detail["stage"] for event in failed] == [stage]
        assert failed[0].detail["error_code"] == "PROVIDER_NETWORK_ERROR"

        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == "ESCALATED"
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert task.status == "FAILED"


def test_lease_loss_is_not_swallowed_as_a_stage_failure(runtime) -> None:
    """租约丢失意味着另一个 worker 已拥有任务；当前路径不能在此处升级任务。"""
    from medidiag.errors import MediDiagError

    _, factory = runtime
    _create_task(factory, "lease-loss")
    worker = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="lease-worker"
    )
    worker.executor.lease.renew = lambda *args, **kwargs: False

    with pytest.raises(MediDiagError) as caught:
        worker.run_once()
    assert caught.value.code == "TASK_LEASE_LOST"


def test_review_rounds_are_capped_and_escalate(runtime) -> None:
    """卡在 REVISION_REQUIRED 的 Provider 必须升级，不能无限循环。"""
    _, factory = runtime
    case_id, task_id = _create_task(factory, "review-round-cap")
    executor = WorkflowExecutor()

    def _run() -> str:
        return SingleMachineWorker(
            factory,
            DeterministicWorkflowProvider(review_verdict="REVISION_REQUIRED"),
            worker_id="round-cap-worker",
            max_review_rounds=3,
        ).run_once().final_state

    # 第 1、2 轮将病例暂停为待修订；每轮都需要新的工作流任务。
    for round_number in (1, 2):
        assert _run() == "REVISION_REQUIRED"
        with factory() as session:
            case = session.execute(
                select(Case).where(Case.case_id == case_id)
            ).scalar_one()
            assert case.review_round == round_number
            executor.start_workflow(
                session, case_id, "case_workflow",
                f"review-round-cap-{round_number}", "input-hash",
            )

    # 第 3 轮达到上限，升级而不是再次请求修订。
    assert _run() == "ESCALATED"

    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == "ESCALATED"
        assert case.review_round == 3

        reviews = session.execute(
            select(CaseEventLog)
            .where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_completed",
            )
            .order_by(CaseEventLog.id)
        ).scalars().all()
        capped = [
            event.detail for event in reviews
            if event.detail.get("error_code") == "MAX_REVIEW_ROUNDS_EXCEEDED"
        ]
        assert len(capped) == 1
        assert capped[0]["review_round"] == 3
        assert capped[0]["max_review_rounds"] == 3
        # Provider 仍然返回 REVISION_REQUIRED；这里只重定向目标主体。
        assert capped[0]["verdict"] == "REVISION_REQUIRED"


def test_review_round_cap_is_configurable(runtime) -> None:
    """max_review_rounds=1 时，第一次修订请求就应升级。"""
    _, factory = runtime
    case_id, _ = _create_task(factory, "review-round-cap-1")
    result = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(review_verdict="REVISION_REQUIRED"),
        worker_id="round-cap-1-worker",
        max_review_rounds=1,
    ).run_once()
    assert result.final_state == "ESCALATED"
    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == "ESCALATED"
        assert case.review_round == 1


def test_compliance_block_escalation_records_its_own_error_code(runtime) -> None:
    """合规拦截驱动的升级要带 COMPLIANCE_BLOCKED，与普通复核升级区分开。"""
    _, factory = runtime
    case_id, _ = _create_task(factory, "compliance-block")

    class BlockingReviewProvider(DeterministicWorkflowProvider):
        def review(
            self, generation: dict, arbitration: dict, retrieval: dict
        ) -> dict:
            payload = super().review(generation, arbitration, retrieval)
            payload["verdict"] = "ESCALATED"
            payload["compliance_status"] = ComplianceStatus.BLOCKED.value
            payload["issues"] = ["absolute_term_blocked"]
            return payload

    result = SingleMachineWorker(
        factory, BlockingReviewProvider(), worker_id="compliance-worker"
    ).run_once()
    assert result.final_state == "ESCALATED"

    with factory() as session:
        events = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_completed",
            )
        ).scalars().all()
        blocked = [
            event.detail
            for event in events
            if event.detail.get("error_code") == "COMPLIANCE_BLOCKED"
        ]
        assert len(blocked) == 1
        assert blocked[0]["stage"] == "review"

        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.case_id == case_id)
        ).scalar_one()
        assert task.result == {"outcome": "COMPLIANCE_BLOCKED"}


# ===== 租约心跳（DD-020）=====


class _SlowNormalizeProvider(DeterministicWorkflowProvider):
    """normalize 阶段刻意慢于 lease_seconds，用于验证心跳续期。"""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds

    def normalize(self, question: str):
        time.sleep(self.delay_seconds)
        return super().normalize(question)


def _short_lease_worker(factory, provider, worker_id: str) -> SingleMachineWorker:
    # lease 2s / heartbeat 1s 是构造校验允许的最短可用组合
    # （heartbeat 必须 >=1 且 < lease）。
    return SingleMachineWorker(
        factory,
        provider,
        worker_id=worker_id,
        executor=WorkflowExecutor(
            LeaseManager(lease_seconds=2, heartbeat_seconds=1, scan_interval_seconds=1)
        ),
    )


def test_stage_longer_than_the_lease_survives(runtime) -> None:
    """单个阶段长于 lease_seconds 时，心跳必须让它活下来。

    这是 DD-020 记录的缺口：租约此前只在阶段之间续期，因此一次超过
    ``lease_seconds`` 的 provider 调用会在仍在执行时被扫描器接管，随后
    ``commit_stage`` 的条件 UPDATE 拒绝写入并记 ``TASK_LEASE_LOST``。
    """
    _, factory = runtime
    case_id, task_id = _create_task(factory, "heartbeat-long-stage")
    # 3s 阶段对 2s 租约：没有心跳时租约必然在阶段结束前过期。
    result = _short_lease_worker(
        factory, _SlowNormalizeProvider(3.0), "heartbeat-worker"
    ).run_once()

    assert result.processed is True
    assert result.final_state == "CLOSED_SUCCESS"

    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == "CLOSED_SUCCESS"
        lease_lost = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "lease_lost",
            )
        ).scalars().all()
        assert lease_lost == []
        task = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one()
        assert task.status == "SUCCEEDED"


def test_without_the_heartbeat_the_same_stage_loses_the_lease(runtime) -> None:
    """对照组：证明上一个测试测的是心跳，而不是宽松的租约判定。"""
    _, factory = runtime
    _create_task(factory, "heartbeat-control")
    worker = _short_lease_worker(
        factory, _SlowNormalizeProvider(3.0), "control-worker"
    )
    # 停掉心跳线程，其余完全一致。
    worker._invoke = types.MethodType(
        lambda self, session, case, task, stage, operation: self.call_runner.call(
            stage,
            operation,
            on_attempt=lambda attempt: self._record_provider_attempt(
                session, case, task, attempt
            ),
        ),
        worker,
    )

    with pytest.raises(MediDiagError) as caught:
        worker.run_once()
    assert caught.value.code == "TASK_LEASE_LOST"


def test_heartbeat_stops_renewing_after_its_budget(runtime) -> None:
    """心跳有上界：卡死的调用不能被无限续期，否则任务永远无法被接管。"""
    _, factory = runtime
    _create_task(factory, "heartbeat-budget")
    worker = _short_lease_worker(
        factory, _SlowNormalizeProvider(3.0), "budget-worker"
    )
    # 预算短于阶段时长：心跳应放弃续期，租约随之过期。
    worker.heartbeat_max_seconds = 1.0

    with pytest.raises(MediDiagError) as caught:
        worker.run_once()
    assert caught.value.code == "TASK_LEASE_LOST"


def test_heartbeat_reports_a_rejected_renewal(runtime) -> None:
    """续期被拒（另一 worker 已接管）时，心跳必须把租约标记为已丢失。"""
    _, factory = runtime
    case_id, task_id = _create_task(factory, "heartbeat-rejected")
    lease = LeaseManager(lease_seconds=2, heartbeat_seconds=1, scan_interval_seconds=1)
    with factory() as session:
        assert lease.acquire(session, task_id, "owner-worker") is True
        attempt = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one().attempt

    # owner 不匹配：续期条件 UPDATE 命中 0 行。
    heartbeat = LeaseHeartbeat(
        factory, lease, task_id=task_id, worker_id="other-worker", attempt=attempt
    )
    with heartbeat:
        time.sleep(1.5)
    assert heartbeat.lease_lost is True
    assert heartbeat.renewals == 0
    assert case_id


def test_heartbeat_uses_its_own_session(runtime) -> None:
    """心跳不得复用 worker 的会话：Session 非线程安全，且外部 IO 不进事务。"""
    _, factory = runtime
    _, task_id = _create_task(factory, "heartbeat-session")
    lease = LeaseManager(lease_seconds=2, heartbeat_seconds=1, scan_interval_seconds=1)
    with factory() as session:
        assert lease.acquire(session, task_id, "owner-worker") is True
        attempt = session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == task_id)
        ).scalar_one().attempt

    opened: list[int] = []
    original = factory

    def _tracking_factory():
        opened.append(1)
        return original()

    heartbeat = LeaseHeartbeat(
        _tracking_factory, lease, task_id=task_id, worker_id="owner-worker",
        attempt=attempt,
    )
    with heartbeat:
        time.sleep(1.5)
    assert heartbeat.lease_lost is False
    assert heartbeat.renewals >= 1
    # 每次续期都自取会话，没有共用调用方的 Session。
    assert len(opened) >= 1
