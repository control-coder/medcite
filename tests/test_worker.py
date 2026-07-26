"""P0-C single-machine worker and recovery tests."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from sqlalchemy import select, update

from medidiag.db.models import (
    Case,
    CaseEventLog,
    CaseReport,
    StageArtifact,
    WorkflowTask,
)
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.executor import WorkflowExecutor
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
    """A DNS/connection failure must be classified, not propagated as a raw httpx error."""

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
