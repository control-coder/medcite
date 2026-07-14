"""FastAPI MVP for cases, workflow control, events, reports, and human decisions."""

from __future__ import annotations

import re
from collections.abc import Generator

from fastapi import Depends, FastAPI, Header, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from medidiag.api.schemas import (
    CaseCreateRequest,
    CaseResponse,
    ErrorResponse,
    EventPage,
    EventResponse,
    HumanDecisionRequest,
    ReportResponse,
    WorkflowStartResponse,
)
from medidiag.config import get_settings
from medidiag.db.models import Case, CaseEventLog, CaseReport, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.idempotency import compute_input_hash
from medidiag.workflow.state_machine import CaseState, TriggerSubject, is_terminal

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CN_ID = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def _ensure_deidentified(payload: CaseCreateRequest) -> None:
    if payload.input_kind == "public_dataset" and not payload.source_ref:
        raise MediDiagError(
            "CASE_INVALID_INPUT", detail="public_dataset input requires source_ref"
        )
    if any(pattern.search(payload.question) for pattern in (_EMAIL, _PHONE, _CN_ID)):
        raise MediDiagError("CASE_INPUT_NOT_DEIDENTIFIED")


def create_app(
    *,
    database_url: str | None = None,
    session_factory: sessionmaker[Session] | None = None,
    initialize_schema: bool = False,
) -> FastAPI:
    app = FastAPI(
        title="MediDiag-Agent EvidenceFlow",
        version="0.1.0",
        description="Engineering prototype for public/deidentified data; not medical advice.",
    )
    if session_factory is None:
        engine = create_db_engine(database_url or get_settings().database_url)
        if initialize_schema:
            init_db(engine)
        session_factory = get_session_factory(engine)
        app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.executor = WorkflowExecutor()

    def get_session(request: Request) -> Generator[Session, None, None]:
        with request.app.state.session_factory() as session:
            yield session

    @app.exception_handler(MediDiagError)
    async def medidiag_error_handler(
        request: Request, exc: MediDiagError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.spec.http_status,
            content=ErrorResponse(
                code=exc.code,
                detail=exc.detail,
                retryable=exc.spec.retryable,
                action=exc.spec.default_action,
            ).model_dump(),
        )

    @app.post(
        "/api/v1/cases",
        response_model=CaseResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_case(
        payload: CaseCreateRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        user_scope: str = Header(default="local-demo", alias="X-User-Scope"),
        session: Session = Depends(get_session),
    ) -> CaseResponse:
        if not idempotency_key:
            raise MediDiagError("IDEMPOTENCY_KEY_MISSING")
        _ensure_deidentified(payload)
        case = app.state.executor.create_case(
            session,
            payload.question,
            idempotency_key,
            user_scope,
            input_kind=payload.input_kind,
            source_ref=payload.source_ref,
        )
        return _case_response(session, case)

    @app.post(
        "/api/v1/cases/{case_id}/workflow",
        response_model=WorkflowStartResponse,
    )
    def start_workflow(
        case_id: str,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        session: Session = Depends(get_session),
    ) -> WorkflowStartResponse:
        if not idempotency_key:
            raise MediDiagError("IDEMPOTENCY_KEY_MISSING")
        case = _case_or_404(session, case_id)
        if is_terminal(CaseState(case.status)):
            raise MediDiagError("CASE_ALREADY_CLOSED")
        if case.status == CaseState.ESCALATED.value:
            raise MediDiagError(
                "ILLEGAL_STATE_TRANSITION",
                detail="ESCALATED requires a human decision before restart",
            )
        task = app.state.executor.start_workflow(
            session,
            case_id,
            "case_workflow",
            idempotency_key,
            compute_input_hash(
                {"case_id": case_id, "question": case.question, "version": case.version}
            ),
        )
        return _task_response(task)

    @app.get("/api/v1/cases/{case_id}", response_model=CaseResponse)
    def get_case(
        case_id: str, session: Session = Depends(get_session)
    ) -> CaseResponse:
        return _case_response(session, _case_or_404(session, case_id))

    @app.get("/api/v1/cases/{case_id}/events", response_model=EventPage)
    def get_events(
        case_id: str,
        cursor: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> EventPage:
        _case_or_404(session, case_id)
        records = list(
            session.execute(
                select(CaseEventLog)
                .where(
                    CaseEventLog.case_id == case_id,
                    CaseEventLog.id > cursor,
                )
                .order_by(CaseEventLog.id)
                .limit(limit + 1)
            ).scalars()
        )
        has_more = len(records) > limit
        records = records[:limit]
        return EventPage(
            items=[
                EventResponse(
                    event_id=item.id,
                    event_type=item.event_type,
                    from_status=item.from_status,
                    to_status=item.to_status,
                    trigger_subject=item.trigger_subject,
                    trigger_entity=item.trigger_entity,
                    detail=item.detail,
                    created_at=item.created_at.isoformat(),
                )
                for item in records
            ],
            next_cursor=records[-1].id if has_more and records else None,
        )

    @app.get("/api/v1/cases/{case_id}/report", response_model=ReportResponse)
    def get_report(
        case_id: str, session: Session = Depends(get_session)
    ) -> ReportResponse:
        _case_or_404(session, case_id)
        report = session.execute(
            select(CaseReport)
            .where(CaseReport.case_id == case_id)
            .order_by(CaseReport.version.desc())
            .limit(1)
        ).scalar_one_or_none()
        if report is None:
            raise MediDiagError("REPORT_NOT_READY")
        return ReportResponse(
            report_id=report.report_id,
            case_id=report.case_id,
            version=report.version,
            structured_report=report.structured_report,
            risk_warnings=report.risk_warnings,
            compliance_status=report.compliance_status,
            generation_version=report.generation_version,
            created_at=report.created_at.isoformat(),
        )

    @app.post(
        "/api/v1/cases/{case_id}/human-decisions",
        response_model=CaseResponse,
    )
    def human_decision(
        case_id: str,
        payload: HumanDecisionRequest,
        session: Session = Depends(get_session),
    ) -> CaseResponse:
        case = _case_or_404(session, case_id)
        if case.status != CaseState.ESCALATED.value:
            raise MediDiagError(
                "ILLEGAL_STATE_TRANSITION",
                detail="human decisions are only allowed from ESCALATED",
            )
        app.state.executor.advance_state(
            session,
            case_id,
            CaseState(payload.decision),
            TriggerSubject.HUMAN,
            trigger_entity="human-reviewer",
            event_type="human_decision",
            detail={"decision": payload.decision, "reason": payload.reason},
        )
        session.expire_all()
        return _case_response(session, _case_or_404(session, case_id))

    return app


def _case_or_404(session: Session, case_id: str) -> Case:
    case = session.execute(
        select(Case).where(Case.case_id == case_id)
    ).scalar_one_or_none()
    if case is None:
        raise MediDiagError("CASE_NOT_FOUND", detail=f"case {case_id} not found")
    return case


def _task_response(task: WorkflowTask) -> WorkflowStartResponse:
    return WorkflowStartResponse(
        task_id=task.task_id,
        case_id=task.case_id,
        status=task.status,
        attempt=task.attempt,
    )


def _case_response(session: Session, case: Case) -> CaseResponse:
    active = (
        session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == case.active_task_id)
        ).scalar_one_or_none()
        if case.active_task_id
        else None
    )
    actions = (
        ["REVISION_REQUIRED", "APPROVED", "CLOSED_ESCALATED"]
        if case.status == CaseState.ESCALATED.value
        else []
    )
    return CaseResponse(
        case_id=case.case_id,
        trace_id=case.trace_id,
        status=case.status,
        version=case.version,
        input_kind=case.input_kind,
        source_ref=case.source_ref,
        normalized_query=case.normalized_query,
        active_task=_task_response(active) if active else None,
        available_human_actions=actions,
    )


app = create_app()
