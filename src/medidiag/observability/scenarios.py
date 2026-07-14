"""Deterministic scenarios used to produce reproducible trace examples."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import update

from medidiag.db.models import Case, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.observability.trace_exporter import TraceExportResult, TraceExporter
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.idempotency import compute_input_hash
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.state_machine import CaseState, TriggerSubject
from medidiag.workflow.worker import LeaseScanner, SingleMachineWorker


@dataclass(frozen=True)
class ScenarioTrace:
    scenario: str
    final_state: str
    export: TraceExportResult


def generate_trace_examples(output_root: str | Path) -> list[ScenarioTrace]:
    """Generate three network-free traces without retaining a scenario database."""
    output_root = Path(output_root)
    with tempfile.TemporaryDirectory(prefix="medidiag-traces-") as directory:
        engine = create_db_engine(
            f"sqlite:///{(Path(directory) / 'scenarios.db').as_posix()}"
        )
        init_db(engine)
        factory = get_session_factory(engine)
        try:
            case_ids = {
                "success": _success_scenario(factory),
                "lease_recovery": _lease_recovery_scenario(factory),
                "review_escalation": _review_escalation_scenario(factory),
            }
            exporter = TraceExporter()
            results = []
            with factory() as session:
                for scenario, case_id in case_ids.items():
                    exported = exporter.export_case(
                        session,
                        case_id,
                        raw_dir=output_root / "raw",
                        summary_dir=output_root / "summary",
                    )
                    case = session.query(Case).filter_by(case_id=case_id).one()
                    results.append(ScenarioTrace(scenario, case.status, exported))
            return results
        finally:
            engine.dispose()


def _create_task(factory, scenario: str) -> tuple[str, str]:
    executor = WorkflowExecutor()
    question = f"Deidentified simulated {scenario} case for trace verification."
    with factory() as session:
        case = executor.create_case(
            session,
            question,
            f"case-{scenario}",
            "trace-example",
        )
        task = executor.start_workflow(
            session,
            case.case_id,
            "case_workflow",
            f"workflow-{scenario}",
            compute_input_hash({"scenario": scenario}),
        )
        return case.case_id, task.task_id


def _success_scenario(factory) -> str:
    case_id, _ = _create_task(factory, "success")
    result = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(),
        worker_id="trace-success-worker",
    ).run_once()
    if result.final_state != CaseState.CLOSED_SUCCESS.value:
        raise RuntimeError("success trace scenario did not close successfully")
    return case_id


def _lease_recovery_scenario(factory) -> str:
    case_id, task_id = _create_task(factory, "lease-recovery")
    executor = WorkflowExecutor()
    with factory() as session:
        if not executor.lease.acquire(session, task_id, "trace-crashed-worker"):
            raise RuntimeError("could not acquire scenario lease")
        session.execute(
            update(WorkflowTask)
            .where(WorkflowTask.task_id == task_id)
            .values(lease_until=executor.lease.now())
        )
        session.commit()

    reclaimed = LeaseScanner(
        factory, recovery_worker_id="trace-recovery-worker"
    ).scan_once()
    if reclaimed != [task_id]:
        raise RuntimeError("lease trace scenario was not reclaimed")

    with factory() as session:
        try:
            executor.commit_stage(
                session,
                task_id=task_id,
                worker_id="trace-crashed-worker",
                attempt=0,
                to_state=CaseState.NORMALIZED,
                subject=TriggerSubject.WORKER,
                stage="normalize",
            )
        except MediDiagError as exc:
            if exc.code != "TASK_LEASE_LOST":
                raise
        else:
            raise RuntimeError("stale worker write unexpectedly succeeded")

    result = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(),
        worker_id="trace-recovery-worker",
    ).run_once()
    if result.final_state != CaseState.CLOSED_SUCCESS.value:
        raise RuntimeError("lease trace scenario did not recover")
    return case_id


def _review_escalation_scenario(factory) -> str:
    case_id, _ = _create_task(factory, "review-escalation")
    escalated = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(review_verdict="ESCALATED"),
        worker_id="trace-review-worker",
    ).run_once()
    if escalated.final_state != CaseState.ESCALATED.value:
        raise RuntimeError("review trace scenario did not escalate")

    executor = WorkflowExecutor()
    with factory() as session:
        executor.advance_state(
            session,
            case_id,
            CaseState.APPROVED,
            TriggerSubject.HUMAN,
            trigger_entity="trace-human-reviewer",
            event_type="human_decision",
            detail={
                "decision": CaseState.APPROVED.value,
                "reason": "Approved deterministic non-diagnostic fixture for trace demo.",
            },
        )
        task = executor.start_workflow(
            session,
            case_id,
            "case_workflow",
            "workflow-review-resume",
            compute_input_hash({"scenario": "review-resume"}),
        )
        if task.status != "PENDING":
            raise RuntimeError("review trace resume task was not created")

    resumed = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(),
        worker_id="trace-resume-worker",
    ).run_once()
    if resumed.final_state != CaseState.CLOSED_SUCCESS.value:
        raise RuntimeError("review trace scenario did not resume")
    return case_id
