"""P7 可靠性故障注入与恢复闭环测试。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import select, update

from medidiag.db.models import Case, CaseEventLog, CaseReport, StageArtifact, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.observability.trace_exporter import TraceExporter
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.provider_runtime import ProviderCallRunner, ProviderResponse
from medidiag.workflow.state_machine import CaseState, TriggerSubject
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker


def _http_error(status: int, request_id: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request, headers={"x-request-id": request_id})
    return httpx.HTTPStatusError("注入的 provider 故障", request=request, response=response)


@pytest.fixture
def runtime(tmp_path: Path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'reliability.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    yield engine, factory
    engine.dispose()


def _create_task(factory, key: str) -> tuple[str, str]:
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session,
            "公开脱敏模拟输入，仅用于可靠性工程测试。",
            f"case-{key}",
            "p7-reliability",
        )
        task = executor.start_workflow(
            session, case.case_id, "case_workflow", key, f"hash-{key}"
        )
        return case.case_id, task.task_id


class _RetryThenRecoverProvider(DeterministicWorkflowProvider):
    """normalize 前两次返回 429，第三次恢复。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def normalize(self, question: str) -> ProviderResponse:
        self.calls += 1
        if self.calls < 3:
            raise _http_error(429, f"req-rate-{self.calls}")
        return ProviderResponse(
            super().normalize(question), request_id="req-rate-recovered"
        )


def test_429_retry_recovers_without_duplicate_business_writes(runtime) -> None:
    _, factory = runtime
    case_id, _ = _create_task(factory, "rate-limit-recovery")
    provider = _RetryThenRecoverProvider()
    result = SingleMachineWorker(
        factory,
        provider,
        worker_id="p7-rate-worker",
        call_runner=ProviderCallRunner(backoff_seconds=(0, 0), sleep=lambda _: None),
    ).run_once()

    assert result.final_state == CaseState.CLOSED_SUCCESS.value
    assert provider.calls == 3
    with factory() as session:
        attempts = session.execute(
            select(CaseEventLog)
            .where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "provider_call",
            )
            .order_by(CaseEventLog.id)
        ).scalars().all()
        normalize_attempts = [item for item in attempts if item.detail["stage"] == "normalize"]
        assert [item.detail["retry_decision"] for item in normalize_attempts] == [
            "retry",
            "retry",
            "not_needed",
        ]
        assert [item.detail["provider_request_id"] for item in normalize_attempts] == [
            "req-rate-1",
            "req-rate-2",
            "req-rate-recovered",
        ]
        artifacts = session.execute(
            select(StageArtifact).where(StageArtifact.case_id == case_id)
        ).scalars().all()
        assert len(artifacts) == 7
        assert len({item.stage for item in artifacts}) == 7
        assert session.query(CaseReport).filter_by(case_id=case_id).count() == 1


@pytest.mark.parametrize(
    ("fault", "expected_code", "expected_attempts"),
    [
        (httpx.ReadTimeout("注入超时"), "LLM_TIMEOUT", 2),
        (_http_error(503, "req-5xx"), "PROVIDER_UNAVAILABLE", 2),
        ({}, "PROVIDER_SCHEMA_INVALID", 1),
        (MediDiagError("STRUCTURED_OUTPUT_INVALID"), "STRUCTURED_OUTPUT_INVALID", 1),
        (MediDiagError("PROVIDER_CONTEXT_LIMIT"), "PROVIDER_CONTEXT_LIMIT", 1),
    ],
    ids=["timeout", "5xx", "empty-response", "json-invalid", "context-limit"],
)
def test_generation_faults_fail_closed_with_stable_trace(
    runtime, fault: object, expected_code: str, expected_attempts: int
) -> None:
    _, factory = runtime
    case_id, _ = _create_task(factory, f"generation-{expected_code}")

    class FaultProvider(DeterministicWorkflowProvider):
        def generate(self, question: str, retrieval: dict, plan: dict) -> dict:
            if isinstance(fault, BaseException):
                raise fault
            return fault  # type: ignore[return-value]

    result = SingleMachineWorker(
        factory,
        FaultProvider(),
        worker_id="p7-fault-worker",
        call_runner=ProviderCallRunner(
            max_attempts=2, backoff_seconds=(0,), sleep=lambda _: None
        ),
    ).run_once()

    assert result.final_state == CaseState.ESCALATED.value
    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == CaseState.ESCALATED.value
        assert session.query(CaseReport).filter_by(case_id=case_id).count() == 0
        assert (
            session.query(StageArtifact)
            .filter_by(case_id=case_id, stage="generation")
            .count()
            == 0
        )
        provider_events = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "provider_call",
            )
        ).scalars().all()
        generation_events = [item for item in provider_events if item.detail["stage"] == "generation"]
        assert len(generation_events) == expected_attempts
        assert generation_events[-1].detail["error_code"] == expected_code
        failure = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_failed",
            )
        ).scalar_one()
        assert failure.detail["error_code"] == expected_code


def test_review_judge_timeout_fails_closed_without_report(runtime) -> None:
    _, factory = runtime
    case_id, _ = _create_task(factory, "judge-timeout")

    class JudgeTimeoutProvider(DeterministicWorkflowProvider):
        def review(self, generation: dict, arbitration: dict, retrieval: dict) -> dict:
            raise httpx.ReadTimeout("注入的固定 judge timeout")

    result = SingleMachineWorker(
        factory,
        JudgeTimeoutProvider(),
        worker_id="p7-judge-worker",
        call_runner=ProviderCallRunner(
            max_attempts=2, backoff_seconds=(0,), sleep=lambda _: None
        ),
    ).run_once()

    assert result.final_state == CaseState.ESCALATED.value
    with factory() as session:
        assert session.query(CaseReport).filter_by(case_id=case_id).count() == 0
        failure = session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_failed",
            )
        ).scalar_one()
        assert failure.detail["stage"] == "review"
        assert failure.detail["error_code"] == "JUDGE_TIMEOUT"
        assert (
            session.query(StageArtifact)
            .filter_by(case_id=case_id, stage="review")
            .count()
            == 0
        )


def test_crash_reclaim_and_late_write_fence_preserve_new_worker_artifact(
    runtime, tmp_path: Path
) -> None:
    _, factory = runtime
    case_id, task_id = _create_task(factory, "crash-reclaim")
    executor = WorkflowExecutor()
    with factory() as session:
        assert executor.lease.acquire(session, task_id, "worker-old") is True
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task_id)
            .values(lease_until=executor.lease.now() - timedelta(seconds=1))
        )
        session.commit()

    assert LeaseScanner(factory, recovery_worker_id="worker-new").scan_once() == [task_id]
    recovered = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="worker-new"
    ).run_once()
    assert recovered.final_state == CaseState.CLOSED_SUCCESS.value

    stale_record = StageArtifact(
        artifact_id="artifact-stale-p7",
        case_id=case_id,
        task_id=task_id,
        stage="normalize",
        attempt=0,
        payload={"normalized_query": "过期结果"},
        input_hash="stale-input",
        output_hash="stale-output",
        component_version="stale-worker",
        latency_ms=1,
    )
    with factory() as session:
        with pytest.raises(MediDiagError) as caught:
            executor.commit_stage(
                session,
                task_id=task_id,
                worker_id="worker-old",
                attempt=0,
                to_state=CaseState.NORMALIZED,
                subject=TriggerSubject.WORKER,
                stage="normalize",
                records=[stale_record],
            )
        assert caught.value.code == "TASK_LEASE_LOST"
        normalize = session.execute(
            select(StageArtifact).where(
                StageArtifact.case_id == case_id,
                StageArtifact.stage == "normalize",
            )
        ).scalar_one()
        assert normalize.attempt == 1
        assert normalize.component_version != "stale-worker"
        assert session.query(CaseReport).filter_by(case_id=case_id).count() == 1
        exported = TraceExporter().export_case(
            session,
            case_id,
            raw_dir=tmp_path / "raw",
            summary_dir=tmp_path / "summary",
        )
        summary = exported.summary_path.read_text(encoding="utf-8")
        assert "TASK_LEASE_LOST" in summary
        assert "过期结果" not in summary


def test_duplicate_start_keeps_one_active_task_and_one_start_event(runtime) -> None:
    _, factory = runtime
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(session, "公开脱敏输入。", "duplicate-case", "p7")
        first = executor.start_workflow(
            session, case.case_id, "case_workflow", "same-key", "same-hash"
        )
        second = executor.start_workflow(
            session, case.case_id, "case_workflow", "same-key", "same-hash"
        )
        assert first.task_id == second.task_id
        with pytest.raises(MediDiagError) as caught:
            executor.start_workflow(
                session, case.case_id, "case_workflow", "different-key", "other-hash"
            )
        assert caught.value.code == "WORKFLOW_ALREADY_RUNNING"
        assert session.query(WorkflowTask).filter_by(case_id=case.case_id).count() == 1
        assert (
            session.query(CaseEventLog)
            .filter_by(case_id=case.case_id, event_type="workflow_started")
            .count()
            == 1
        )


class _ZeroRowcount:
    rowcount = 0


class _CasFaultOnceSession:
    """仅跳过第一次病例 UPDATE，用于确定性注入 CAS conflict。"""

    def __init__(self, session) -> None:
        self._session = session
        self.injected = False

    def execute(self, statement, *args, **kwargs):
        text = str(statement)
        if not self.injected and text.startswith("UPDATE cases SET") and "active_task_id" in text:
            self.injected = True
            return _ZeroRowcount()
        return self._session.execute(statement, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


def test_commit_stage_retries_cas_without_duplicate_event_or_artifact(
    runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, factory = runtime
    case_id, task_id = _create_task(factory, "cas-retry")
    executor = WorkflowExecutor()
    sleeps: list[float] = []
    monkeypatch.setattr("medidiag.workflow.executor.time.sleep", sleeps.append)
    with factory() as real_session:
        assert executor.lease.acquire(real_session, task_id, "cas-worker") is True
        session = _CasFaultOnceSession(real_session)
        executor.commit_stage(
            session,  # type: ignore[arg-type]
            task_id=task_id,
            worker_id="cas-worker",
            attempt=0,
            to_state=CaseState.NORMALIZED,
            subject=TriggerSubject.WORKER,
            stage="normalize",
            records=[],
        )
        assert session.injected is True
        assert sleeps == [0.05]
        completed = real_session.execute(
            select(CaseEventLog).where(
                CaseEventLog.case_id == case_id,
                CaseEventLog.event_type == "stage_completed",
            )
        ).scalars().all()
        assert len(completed) == 1
        assert completed[0].detail["optimistic_lock_retry_count"] == 1
        case = real_session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        assert case.status == CaseState.NORMALIZED.value
