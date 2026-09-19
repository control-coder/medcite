"""单机工作流 worker 和租约扫描器。"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from medidiag.compliance.status import is_compliance_hit
from medidiag.db.models import (
    AgentRun,
    Case,
    CaseEventLog,
    CaseReport,
    Citation,
    Review,
    StageArtifact,
    WorkflowTask,
)
from medidiag.errors import MediDiagError
from medidiag.observability.logging import get_logger
from medidiag.review.logic import ClinicalLogicReviewer
from medidiag.workflow.assistant_pipeline import (
    AssistantPipeline,
    StageContext,
    StageExecution,
)
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.idempotency import compute_input_hash
from medidiag.workflow.lease import LeaseHeartbeat
from medidiag.workflow.provider import WorkflowProvider
from medidiag.workflow.provider_runtime import (
    ProviderAttempt,
    ProviderCallOutcome,
    ProviderCallRunner,
    ProviderResponse,
)
from medidiag.workflow.state_machine import CaseState, TriggerSubject, is_terminal


@dataclass
class WorkerRunResult:
    processed: bool
    task_id: str | None = None
    case_id: str | None = None
    final_state: str | None = None


# 每个状态驱动的阶段，以及获准推进该状态的触发主体。
# 成功边与失败边共享这份定义，
# 确保阶段不会使用状态机拒绝的主体升级处理。
_STAGE_BY_STATE: dict[CaseState, tuple[str, TriggerSubject]] = {
    CaseState.CREATED: ("normalize", TriggerSubject.WORKER),
    CaseState.NORMALIZED: ("retrieval", TriggerSubject.WORKER),
    CaseState.EVIDENCE_RETRIEVED: ("plan", TriggerSubject.WORKER),
    CaseState.PLAN_GENERATED: ("generation", TriggerSubject.AGENT_WORKER),
    CaseState.REVISION_REQUIRED: ("revision_restart", TriggerSubject.REVIEWER_WORKER),
    CaseState.SPECIALIST_REVIEWING: ("arbitration", TriggerSubject.AGENT_WORKER),
    CaseState.ARBITRATION_REVIEWING: ("review", TriggerSubject.REVIEWER_WORKER),
    CaseState.APPROVED: ("report", TriggerSubject.REVIEWER_WORKER),
    CaseState.REPORT_GENERATED: ("close", TriggerSubject.WORKER),
}

_STAGE_SUBJECTS: dict[str, TriggerSubject] = {
    stage: subject for stage, subject in _STAGE_BY_STATE.values()
}


_log = get_logger(__name__)


class _StageFailure(Exception):
    """Provider 阶段耗尽 ProviderCallRunner 有界重试后产生的异常。

    该异常只由 ``_invoke`` 抛出；``_renew`` 的租约错误意味着另一个 worker 已经拥有任务，当前 worker 必须继续传播该错误而不能写入，因此不能把它误判为阶段失败。
    """

    def __init__(self, stage: str, error: MediDiagError) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.error = error


class SingleMachineWorker:
    """在保留数据库租约的同时执行一个完整病例工作流。"""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        provider: WorkflowProvider,
        *,
        worker_id: str = "local-worker",
        executor: WorkflowExecutor | None = None,
        call_runner: ProviderCallRunner | None = None,
        pipeline: AssistantPipeline | None = None,
        max_review_rounds: int = 3,
        heartbeat_max_seconds: float | None = None,
    ) -> None:
        if max_review_rounds < 1:
            raise ValueError("max_review_rounds must be at least 1")
        self.session_factory = session_factory
        self.provider = provider
        self.worker_id = worker_id
        self.executor = executor or WorkflowExecutor()
        self.call_runner = call_runner or ProviderCallRunner()
        self.pipeline = pipeline or AssistantPipeline(
            component_version=provider.version,
            provider_profile=str(
                getattr(
                    provider,
                    "profile_id",
                    getattr(getattr(provider, "llm", None), "profile_id", "workflow_default"),
                )
            ),
            call_runner=self.call_runner,
        )
        self.max_review_rounds = max_review_rounds
        # None 表示用 LeaseHeartbeat 的默认上界（HEARTBEAT_MAX_LEASE_PERIODS
        # 个租约周期）。显式传参主要供测试缩短等待。
        self.heartbeat_max_seconds = heartbeat_max_seconds

    def run_once(self) -> WorkerRunResult:
        with self.session_factory() as session:
            task = session.execute(
                select(WorkflowTask)
                .where(WorkflowTask.status == "PENDING")
                .order_by(WorkflowTask.id)
                .limit(1)
            ).scalar_one_or_none()
            if task is not None:
                if not self.executor.lease.acquire(session, task.task_id, self.worker_id):
                    return WorkerRunResult(processed=False)
                session.expire_all()
                task = session.execute(
                    select(WorkflowTask).where(WorkflowTask.task_id == task.task_id)
                ).scalar_one()
            else:
                task = session.execute(
                    select(WorkflowTask)
                    .where(
                        WorkflowTask.status == "RUNNING",
                        WorkflowTask.lease_owner == self.worker_id,
                        WorkflowTask.lease_until > self.executor.lease.now(),
                    )
                    .order_by(WorkflowTask.id)
                    .limit(1)
                ).scalar_one_or_none()
            if task is None:
                return WorkerRunResult(processed=False)
            _log.info(
                "worker.task_claimed",
                worker_id=self.worker_id,
                task_id=task.task_id,
                case_id=task.case_id,
                task_type=task.task_type,
                attempt=task.attempt,
            )
            return self._process(session, task)

    def _process(self, session: Session, task: WorkflowTask) -> WorkerRunResult:
        for _ in range(16):
            session.expire_all()
            task = session.execute(
                select(WorkflowTask).where(WorkflowTask.task_id == task.task_id)
            ).scalar_one()
            case = session.execute(select(Case).where(Case.case_id == task.case_id)).scalar_one()
            state = CaseState(case.status)
            if is_terminal(state) or state == CaseState.ESCALATED:
                return WorkerRunResult(True, task.task_id, case.case_id, state.value)

            artifacts = self._artifacts(session, case.case_id, task.task_id)
            attempt = task.attempt
            try:
                if state == CaseState.CREATED:
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "normalize",
                        lambda: self.provider.normalize(case.question),
                    )
                    payload = outcome.payload
                    artifact = self._artifact(case, task, outcome)
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.NORMALIZED,
                        subject=TriggerSubject.WORKER,
                        stage="normalize",
                        records=[artifact],
                        case_values={"normalized_query": payload["normalized_query"]},
                        detail={
                            "component_version": self.provider.version,
                            **self._provider_detail(outcome),
                        },
                    )
                    continue

                if state == CaseState.NORMALIZED:
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "retrieval",
                        lambda: self.provider.retrieve(case.normalized_query or case.question),
                    )
                    payload = outcome.payload
                    artifact = self._artifact(case, task, outcome)
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.EVIDENCE_RETRIEVED,
                        subject=TriggerSubject.WORKER,
                        stage="retrieval",
                        records=[artifact],
                        detail={
                            "chunk_count": len(payload.get("chunks", [])),
                            **self._provider_detail(outcome),
                        },
                    )
                    continue

                if state == CaseState.EVIDENCE_RETRIEVED:
                    retrieval = artifacts["retrieval"].payload
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "plan",
                        lambda: self.provider.plan(
                            case.normalized_query or case.question, retrieval
                        ),
                    )
                    payload = outcome.payload
                    artifact = self._artifact(case, task, outcome)
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.PLAN_GENERATED,
                        subject=TriggerSubject.WORKER,
                        stage="plan",
                        records=[artifact],
                        detail=self._provider_detail(outcome),
                    )
                    continue

                if state in {CaseState.PLAN_GENERATED, CaseState.REVISION_REQUIRED}:
                    if state == CaseState.REVISION_REQUIRED:
                        self.executor.commit_stage(
                            session,
                            task_id=task.task_id,
                            worker_id=self.worker_id,
                            attempt=attempt,
                            to_state=CaseState.PLAN_GENERATED,
                            subject=TriggerSubject.REVIEWER_WORKER,
                            stage="revision_restart",
                        )
                        continue
                    retrieval = artifacts["retrieval"].payload
                    plan = artifacts["plan"].payload
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "generation",
                        lambda: self.provider.generate(case.question, retrieval, plan),
                    )
                    payload = outcome.payload
                    records: list[Any] = [self._artifact(case, task, outcome)]
                    default_input = {
                        "question": case.question,
                        "retrieval": retrieval,
                        "plan": plan,
                    }
                    default_input_hash = compute_input_hash(default_input)
                    for agent in payload.get("agents", []):
                        # P4 runtime 会提供每个 Agent 独立的输入、耗时与 provenance；
                        # deterministic fixture 未提供时仍保留历史兼容字段。
                        agent_input = agent.get("input", {"plan": plan})
                        records.append(
                            AgentRun(
                                run_id=agent.get("agent_run_id") or f"run_{uuid.uuid4().hex}",
                                case_id=case.case_id,
                                agent_name=agent["agent_name"],
                                input_hash=agent.get("input_hash") or default_input_hash,
                                attempt_group=f"{task.task_id}:{attempt}",
                                input_payload=agent_input,
                                output_payload=agent,
                                status=agent.get("status", "SUCCEEDED"),
                                latency_ms=int(agent.get("latency_ms", outcome.elapsed_ms)),
                            )
                        )
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.SPECIALIST_REVIEWING,
                        subject=TriggerSubject.AGENT_WORKER,
                        stage="generation",
                        records=records,
                        detail={
                            "agent_count": len(payload.get("agents", [])),
                            **self._provider_detail(outcome),
                        },
                    )
                    continue

                if state == CaseState.SPECIALIST_REVIEWING:
                    generation = artifacts["generation"].payload
                    retrieval = artifacts["retrieval"].payload
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "arbitration",
                        lambda: self.provider.arbitrate(generation, retrieval),
                    )
                    payload = outcome.payload
                    artifact = self._artifact(case, task, outcome)
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.ARBITRATION_REVIEWING,
                        subject=TriggerSubject.AGENT_WORKER,
                        stage="arbitration",
                        records=[artifact],
                        detail=self._provider_detail(outcome),
                    )
                    continue

                if state == CaseState.ARBITRATION_REVIEWING:
                    generation = artifacts["generation"].payload
                    arbitration = artifacts["arbitration"].payload
                    retrieval = artifacts["retrieval"].payload
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "review",
                        lambda: self.provider.review(generation, arbitration, retrieval),
                    )
                    payload = outcome.payload
                    verdict = payload["verdict"]
                    target = CaseState(verdict)
                    # 如果 Provider 持续要求修订，流程本来会无限
                    # 循环 REVISION_REQUIRED -> PLAN_GENERATED。
                    round_number = case.review_round + 1
                    capped = target == CaseState.REVISION_REQUIRED and (
                        ClinicalLogicReviewer.should_escalate(round_number, self.max_review_rounds)
                    )
                    if capped:
                        target = CaseState.ESCALATED
                    # 合规拦截驱动的升级要带上自己的错误码，否则 trace 上与
                    # 普通复核升级无法区分。
                    compliance_blocked = target == CaseState.ESCALATED and (
                        is_compliance_hit(payload.get("compliance_status"))
                    )
                    records = [
                        self._artifact(case, task, outcome),
                        Review(
                            case_id=case.case_id,
                            review_type="workflow",
                            reviewer=self.provider.version,
                            result=verdict,
                            round=round_number,
                            detail=payload,
                        ),
                    ]
                    claim_map = {item["claim_id"]: item for item in generation.get("claims", [])}
                    for item in payload.get("citation_verdicts", []):
                        model_name = str(item.get("model_name") or item["method"])
                        model_revision = str(item.get("model_revision") or "")
                        verifier_ref = (
                            f"{model_name}@{model_revision}"
                            if model_revision
                            else model_name
                        )
                        records.append(
                            Citation(
                                case_id=case.case_id,
                                claim_text=claim_map[item["claim_id"]]["text"],
                                chunk_id=item["chunk_id"],
                                verdict=item["verdict"],
                                verifier_model=verifier_ref,
                                verifier_score=item.get("confidence"),
                            )
                        )
                    complete = target in {CaseState.REVISION_REQUIRED, CaseState.ESCALATED}
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=target,
                        subject=TriggerSubject.REVIEWER_WORKER,
                        stage="review",
                        records=records,
                        case_values={"review_round": round_number},
                        detail={
                            "verdict": verdict,
                            "review_round": round_number,
                            "max_review_rounds": self.max_review_rounds,
                            **(
                                {"error_code": "MAX_REVIEW_ROUNDS_EXCEEDED"}
                                if capped
                                else {"error_code": "COMPLIANCE_BLOCKED"}
                                if compliance_blocked
                                else {}
                            ),
                            **self._provider_detail(outcome),
                        },
                        complete_task=complete,
                        task_result={
                            "outcome": (
                                "MAX_REVIEW_ROUNDS_EXCEEDED"
                                if capped
                                else "COMPLIANCE_BLOCKED"
                                if compliance_blocked
                                else verdict
                            )
                        }
                        if complete
                        else None,
                    )
                    if complete:
                        return WorkerRunResult(True, task.task_id, case.case_id, target.value)
                    continue

                if state == CaseState.APPROVED:
                    generation = artifacts["generation"].payload
                    review = artifacts["review"].payload
                    self._renew(session, task)
                    outcome = self._invoke(
                        session,
                        case,
                        task,
                        "report",
                        lambda: self.provider.report(case.case_id, generation, review),
                    )
                    payload = outcome.payload
                    report_version = (
                        session.execute(
                            select(func.count(CaseReport.id)).where(
                                CaseReport.case_id == case.case_id
                            )
                        ).scalar_one()
                        + 1
                    )
                    records = [
                        self._artifact(case, task, outcome),
                        CaseReport(
                            report_id=f"report_{uuid.uuid4().hex}",
                            case_id=case.case_id,
                            version=report_version,
                            structured_report=payload,
                            risk_warnings=["qualified_clinician_review_required"],
                            compliance_status=review["compliance_status"],
                            generation_version=self.provider.version,
                        ),
                    ]
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.REPORT_GENERATED,
                        subject=TriggerSubject.REVIEWER_WORKER,
                        stage="report",
                        records=records,
                        detail=self._provider_detail(outcome),
                    )
                    continue

                if state == CaseState.REPORT_GENERATED:
                    self.executor.commit_stage(
                        session,
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        attempt=attempt,
                        to_state=CaseState.CLOSED_SUCCESS,
                        subject=TriggerSubject.WORKER,
                        stage="close",
                        complete_task=True,
                        task_result={"outcome": "CLOSED_SUCCESS"},
                    )
                    return WorkerRunResult(
                        True, task.task_id, case.case_id, CaseState.CLOSED_SUCCESS.value
                    )
            except _StageFailure as failure:
                _log.error(
                    "worker.stage_failed",
                    worker_id=self.worker_id,
                    task_id=task.task_id,
                    case_id=case.case_id,
                    attempt=attempt,
                    stage=failure.stage,
                    from_state=state.value,
                    error_code=failure.error.code,
                )
                self._fail_stage(session, case, task, attempt, failure)
                return WorkerRunResult(True, task.task_id, case.case_id, CaseState.ESCALATED.value)

        # 只有审核/修订循环可能重复，WS3 的轮次上限会限制它；
        # 这里可达的每个状态都有合法的 ESCALATED 边。
        stage, _ = _STAGE_BY_STATE[state]
        _log.error(
            "worker.stage_limit_exceeded",
            worker_id=self.worker_id,
            task_id=task.task_id,
            case_id=task.case_id,
            stage=stage,
            state=state.value,
            error_code="WORKFLOW_RETRY_EXCEEDED",
        )
        self._fail_stage(
            session,
            case,
            task,
            task.attempt,
            _StageFailure(
                stage,
                MediDiagError(
                    "WORKFLOW_RETRY_EXCEEDED",
                    detail=f"workflow exceeded stage safety limit: {task.task_id}",
                ),
            ),
        )
        return WorkerRunResult(True, task.task_id, case.case_id, CaseState.ESCALATED.value)

    def _fail_stage(
        self,
        session: Session,
        case: Case,
        task: WorkflowTask,
        attempt: int,
        failure: _StageFailure,
    ) -> None:
        """升级一个在 ProviderCallRunner 重试后仍失败的阶段。"""
        exc = failure.error
        provider_attempt = exc.context.get("provider_attempt", {})
        self.executor.fail_stage(
            session,
            task_id=task.task_id,
            worker_id=self.worker_id,
            attempt=attempt,
            to_state=CaseState.ESCALATED,
            subject=_STAGE_SUBJECTS[failure.stage],
            stage=failure.stage,
            error_code=exc.code,
            error_message=(
                f"{failure.stage} provider failed after bounded retries; no report generated"
            ),
            detail={
                "component_version": self.provider.version,
                "provider_request_id": provider_attempt.get("provider_request_id"),
                "provider_retry_count": max(0, provider_attempt.get("provider_attempt", 1) - 1),
                "retry_decision": provider_attempt.get("retry_decision"),
                "http_status": provider_attempt.get("http_status"),
            },
        )

    def _renew(self, session: Session, task: WorkflowTask) -> None:
        if not self.executor.lease.renew(session, task.task_id, self.worker_id, task.attempt):
            raise MediDiagError(
                "TASK_LEASE_LOST",
                detail=f"cannot renew task lease: {task.task_id}",
            )

    def _invoke(
        self,
        session: Session,
        case: Case,
        task: WorkflowTask,
        stage: str,
        operation: Callable[[], dict[str, Any] | ProviderResponse],
    ) -> StageExecution:
        # 心跳在整个 provider 调用（含有界重试）期间用独立会话续期，因此单个阶段
        # 长于 lease_seconds 不再导致任务被扫描器接管。DD-020。
        heartbeat = LeaseHeartbeat(
            self.session_factory,
            self.executor.lease,
            task_id=task.task_id,
            worker_id=self.worker_id,
            attempt=task.attempt,
            max_seconds=self.heartbeat_max_seconds,
        )
        try:
            with heartbeat:
                outcome = self.pipeline.run_stage(
                    stage=stage,
                    input_payload=self._stage_input(session, case, task, stage),
                    operation=operation,
                    context=StageContext(
                        run_id=case.trace_id,
                        case_id=case.case_id,
                        task_id=task.task_id,
                        mode="worker",
                    ),
                    on_attempt=lambda attempt: self._record_provider_attempt(
                        session, case, task, attempt
                    ),
                )
        except MediDiagError as exc:
            raise _StageFailure(stage, exc) from exc
        # 租约丢失与阶段失败必须区分：前者意味着任务已被别的 worker 接管，本
        # worker 不得写入，因此不能走 _StageFailure 的升级路径。
        if heartbeat.lease_lost or heartbeat.gave_up:
            raise MediDiagError(
                "TASK_LEASE_LOST",
                detail=(
                    f"lease heartbeat stopped during stage {stage}: "
                    f"{'renewal rejected' if heartbeat.lease_lost else 'exceeded heartbeat budget'}"
                ),
                context={"task_id": task.task_id, "stage": stage},
            )
        return outcome

    def _stage_input(
        self, session: Session, case: Case, task: WorkflowTask, stage: str
    ) -> dict[str, Any]:
        """从已提交 artifact 组装稳定输入，兼容历史 worker 执行边界。"""
        artifacts = self._artifacts(session, case.case_id, task.task_id)
        if stage == "normalize":
            return {"question": case.question}
        if stage == "retrieval":
            return {"normalized_query": case.normalized_query}
        if stage == "plan":
            return artifacts["retrieval"].payload
        if stage == "generation":
            return {
                "question": case.question,
                "retrieval": artifacts["retrieval"].payload,
                "plan": artifacts["plan"].payload,
            }
        if stage == "arbitration":
            return {
                "generation": artifacts["generation"].payload,
                "retrieval": artifacts["retrieval"].payload,
            }
        if stage == "review":
            return {
                "generation": artifacts["generation"].payload,
                "arbitration": artifacts["arbitration"].payload,
                "retrieval": artifacts["retrieval"].payload,
            }
        if stage == "report":
            return {
                "case_id": case.case_id,
                "generation": artifacts["generation"].payload,
                "review": artifacts["review"].payload,
            }
        raise ValueError(f"unsupported pipeline stage: {stage}")

    def _record_provider_attempt(
        self,
        session: Session,
        case: Case,
        task: WorkflowTask,
        attempt: ProviderAttempt,
    ) -> None:
        session.add(
            CaseEventLog(
                case_id=case.case_id,
                event_type="provider_call",
                trigger_subject=TriggerSubject.SYSTEM.value,
                trigger_entity=self.worker_id,
                detail={
                    "trace_id": case.trace_id,
                    "task_id": task.task_id,
                    "task_attempt": task.attempt,
                    "provider_version": self.provider.version,
                    **attempt.to_event_detail(),
                },
            )
        )
        session.commit()

    @staticmethod
    def _provider_detail(outcome: StageExecution) -> dict[str, Any]:
        return {
            "provider_request_id": outcome.request_id,
            "provider_retry_count": outcome.retry_count,
            **outcome.metadata,
        }

    def _artifacts(self, session: Session, case_id: str, task_id: str) -> dict[str, StageArtifact]:
        records = session.execute(
            select(StageArtifact).where(StageArtifact.case_id == case_id).order_by(StageArtifact.id)
        ).scalars()
        return {record.stage: record for record in records}

    def _artifact(
        self,
        case: Case,
        task: WorkflowTask,
        execution: StageExecution | ProviderCallOutcome,
    ) -> StageArtifact:
        if isinstance(execution, StageExecution):
            envelope = execution.artifact
            stage = envelope.stage
            payload = envelope.payload
            input_hash = envelope.input_hash
            output_hash = envelope.output_hash
            component_version = f"{envelope.pipeline_version}:{envelope.component_version}"
            latency_ms = envelope.latency_ms
        else:
            # 保留历史测试替换 ``_invoke`` 的兼容边界；生产路径始终返回
            # StageExecution。该分支不改变租约 CAS 的失败语义。
            stage = _STAGE_BY_STATE[CaseState(case.status)][0]
            payload = execution.payload
            input_payload = (
                {"question": case.question} if stage == "normalize" else {"legacy_stage": stage}
            )
            input_hash = compute_input_hash(input_payload)
            output_hash = compute_input_hash(payload)
            component_version = self.provider.version
            latency_ms = execution.elapsed_ms
        return StageArtifact(
            artifact_id=f"artifact_{uuid.uuid4().hex}",
            case_id=case.case_id,
            task_id=task.task_id,
            stage=stage,
            attempt=task.attempt,
            payload=payload,
            input_hash=input_hash,
            output_hash=output_hash,
            component_version=component_version,
            latency_ms=latency_ms,
        )


class LeaseScanner:
    """以原子方式回收本地恢复 worker 的过期任务。"""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        recovery_worker_id: str = "local-worker",
        executor: WorkflowExecutor | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.recovery_worker_id = recovery_worker_id
        self.executor = executor or WorkflowExecutor()

    def scan_once(self) -> list[str]:
        reclaimed: list[str] = []
        with self.session_factory() as session:
            task_ids = [task.task_id for task in self.executor.lease.find_expired(session)]
        for task_id in task_ids:
            with self.session_factory() as session:
                if self.executor.lease.reclaim(session, task_id, self.recovery_worker_id):
                    reclaimed.append(task_id)
        if task_ids:
            _log.info(
                "lease.scan_completed",
                worker_id=self.recovery_worker_id,
                expired=len(task_ids),
                reclaimed=len(reclaimed),
            )
        return reclaimed
