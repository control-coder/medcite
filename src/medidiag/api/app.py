"""用于病例、工作流控制、事件、报告和人工决策的 FastAPI MVP 接口。"""

from __future__ import annotations

import re
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, Header, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from medidiag.api.analysis import build_analysis
from medidiag.api.schemas import (
    AnalysisResponse,
    ConsultationCreateRequest,
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
from medidiag.db.models import (
    Case,
    CaseEventLog,
    CaseReport,
    Citation,
    Review,
    StageArtifact,
    WorkflowTask,
)
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.observability.logging import get_logger
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.idempotency import compute_input_hash
from medidiag.workflow.state_machine import CaseState, TriggerSubject, is_terminal

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_CN_ID = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_API_DIR = Path(__file__).resolve().parent
_TEMPLATES = Jinja2Templates(directory=str(_API_DIR / "templates"))
_log = get_logger(__name__)


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
    app.mount(
        "/static",
        StaticFiles(directory=str(_API_DIR / "static")),
        name="static",
    )
    if session_factory is None:
        engine = create_db_engine(database_url or get_settings().database_url)
        if initialize_schema:
            init_db(engine)
        session_factory = get_session_factory(engine)
        app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.executor = WorkflowExecutor()
    # ``medidiag demo`` 在附加进程内 worker 后会覆盖该配置。
    # 普通的 uvicorn 进程也可以与 ``medidiag worker`` 配合运行。
    app.state.demo_runtime = {
        "label": "独立 worker 模式",
        "detail": "当前 Web 进程未附加同进程 worker；请另行启动 medidiag worker。",
    }

    def get_session(request: Request) -> Generator[Session, None, None]:
        with request.app.state.session_factory() as session:
            yield session

    @app.exception_handler(MediDiagError)
    async def medidiag_error_handler(
        request: Request, exc: MediDiagError
    ) -> Response:
        # 只记录错误码与路由，不记录 exc.detail：CASE_INVALID_INPUT 的 detail
        # 来自 pydantic ValidationError，会带上被拒绝的输入值。
        _log.warning(
            "api.error",
            error_code=exc.code,
            http_status=exc.spec.http_status,
            retryable=exc.spec.retryable,
            alert=exc.spec.alert,
            method=request.method,
            route=request.url.path,
        )
        if request.url.path.startswith(("/demo", "/assistant")):
            return _TEMPLATES.TemplateResponse(
                request=request,
                name="partials/error.html",
                context={"error": exc},
                status_code=exc.spec.http_status,
            )
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

    @app.post("/api/v1/consultations", response_model=CaseResponse, status_code=201)
    def create_consultation(
        payload: ConsultationCreateRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        user_scope: str = Header(default="local-demo", alias="X-User-Scope"),
        session: Session = Depends(get_session),
    ) -> CaseResponse:
        return create_case(payload.as_case(), idempotency_key, user_scope, session)

    @app.get("/api/v1/cases/{case_id}/analysis", response_model=AnalysisResponse)
    def get_analysis(case_id: str, session: Session = Depends(get_session)) -> AnalysisResponse:
        return build_analysis(session, _case_or_404(session, case_id))

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

    @app.get("/assistant", response_class=HTMLResponse)
    def assistant_home(
        request: Request, session: Session = Depends(get_session)
    ) -> HTMLResponse:
        recent = list(
            session.execute(select(Case).order_by(Case.id.desc()).limit(10)).scalars()
        )
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="assistant.html",
            context={
                "recent_cases": recent,
                "demo_runtime": request.app.state.demo_runtime,
            },
        )

    @app.post("/assistant/cases", response_class=HTMLResponse)
    def assistant_create_case(
        request: Request,
        question: str = Form(...),
        input_kind: str = Form(...),
        source_ref: str | None = Form(default=None),
        session: Session = Depends(get_session),
    ) -> Response:
        payload = _validated_form_payload(question, input_kind, source_ref)
        _ensure_deidentified(payload)
        nonce = uuid.uuid4().hex
        case = app.state.executor.create_case(
            session,
            payload.question,
            f"assistant-case-{nonce}",
            "local-assistant",
            input_kind=payload.input_kind,
            source_ref=payload.source_ref,
        )
        app.state.executor.start_workflow(
            session,
            case.case_id,
            "case_workflow",
            f"assistant-workflow-{nonce}",
            compute_input_hash(
                {"case_id": case.case_id, "question": case.question, "version": case.version}
            ),
        )
        location = f"/assistant/cases/{case.case_id}"
        if request.headers.get("HX-Request") == "true":
            return Response(
                status_code=status.HTTP_204_NO_CONTENT,
                headers={"HX-Redirect": location},
            )
        return RedirectResponse(location, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/assistant/cases/{case_id}", response_class=HTMLResponse)
    def assistant_case(
        request: Request,
        case_id: str,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="assistant_case.html",
            context=_assistant_case_context(session, case_id),
        )

    @app.get("/assistant/cases/{case_id}/status", response_class=HTMLResponse)
    def assistant_case_status(
        request: Request,
        case_id: str,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="partials/assistant_case_live.html",
            context=_assistant_case_context(session, case_id),
        )

    @app.get("/demo", response_class=HTMLResponse)
    def demo_home(
        request: Request, session: Session = Depends(get_session)
    ) -> HTMLResponse:
        recent = list(
            session.execute(select(Case).order_by(Case.id.desc()).limit(20)).scalars()
        )
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="demo.html",
            context={
                "recent_cases": recent,
                "demo_runtime": request.app.state.demo_runtime,
            },
        )

    @app.post("/demo/cases", response_class=HTMLResponse)
    def demo_create_case(
        request: Request,
        question: str = Form(...),
        input_kind: str = Form(...),
        source_ref: str | None = Form(default=None),
        session: Session = Depends(get_session),
    ) -> Response:
        payload = _validated_form_payload(question, input_kind, source_ref)
        _ensure_deidentified(payload)
        nonce = uuid.uuid4().hex
        case = app.state.executor.create_case(
            session,
            payload.question,
            f"demo-case-{nonce}",
            "local-demo",
            input_kind=payload.input_kind,
            source_ref=payload.source_ref,
        )
        app.state.executor.start_workflow(
            session,
            case.case_id,
            "case_workflow",
            f"demo-workflow-{nonce}",
            compute_input_hash(
                {"case_id": case.case_id, "question": case.question, "version": case.version}
            ),
        )
        location = f"/demo/cases/{case.case_id}"
        if request.headers.get("HX-Request") == "true":
            return Response(
                status_code=status.HTTP_204_NO_CONTENT,
                headers={"HX-Redirect": location},
            )
        return RedirectResponse(location, status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/demo/cases/{case_id}", response_class=HTMLResponse)
    def demo_case(
        request: Request,
        case_id: str,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="case.html",
            context=_demo_case_context(session, case_id),
        )

    @app.get("/demo/cases/{case_id}/status", response_class=HTMLResponse)
    def demo_case_status(
        request: Request,
        case_id: str,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="partials/case_live.html",
            context=_demo_case_context(session, case_id),
        )

    @app.post("/demo/cases/{case_id}/human-decisions", response_class=HTMLResponse)
    def demo_human_decision(
        request: Request,
        case_id: str,
        decision: str = Form(...),
        reason: str = Form(...),
        session: Session = Depends(get_session),
    ) -> Response:
        try:
            # 同上：Literal 的收窄由 pydantic 在运行时完成。
            payload = HumanDecisionRequest.model_validate(
                {"decision": decision, "reason": reason}
            )
        except ValidationError as exc:
            raise MediDiagError("CASE_INVALID_INPUT", detail=str(exc)) from exc
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
            trigger_entity="demo-human-reviewer",
            event_type="human_decision",
            detail={"decision": payload.decision, "reason": payload.reason},
        )
        if request.headers.get("HX-Request") != "true":
            return RedirectResponse(
                f"/demo/cases/{case_id}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="partials/case_live.html",
            context=_demo_case_context(session, case_id),
        )

    @app.post("/demo/cases/{case_id}/workflow", response_class=HTMLResponse)
    def demo_resume_workflow(
        request: Request,
        case_id: str,
        session: Session = Depends(get_session),
    ) -> Response:
        case = _case_or_404(session, case_id)
        if is_terminal(CaseState(case.status)):
            raise MediDiagError("CASE_ALREADY_CLOSED")
        if case.status == CaseState.ESCALATED.value:
            raise MediDiagError(
                "ILLEGAL_STATE_TRANSITION",
                detail="ESCALATED requires a human decision before restart",
            )
        nonce = uuid.uuid4().hex
        app.state.executor.start_workflow(
            session,
            case_id,
            "case_workflow",
            f"demo-resume-{nonce}",
            compute_input_hash(
                {"case_id": case_id, "question": case.question, "version": case.version}
            ),
        )
        if request.headers.get("HX-Request") != "true":
            return RedirectResponse(
                f"/demo/cases/{case_id}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="partials/case_live.html",
            context=_demo_case_context(session, case_id),
        )

    return app


def _validated_form_payload(
    question: str, input_kind: str, source_ref: str | None
) -> CaseCreateRequest:
    """校验服务端 HTML 表单，不在错误页面回显被拒绝的原始输入。"""
    try:
        return CaseCreateRequest.model_validate(
            {
                "question": question,
                "input_kind": input_kind,
                "source_ref": source_ref or None,
            }
        )
    except ValidationError as exc:
        raise MediDiagError("CASE_INVALID_INPUT", detail=str(exc)) from exc


def _assistant_case_context(session: Session, case_id: str) -> dict[str, Any]:
    """构造面向用户的收敛视图，不暴露内部错误详情或完整 reasoning。"""
    case = _case_or_404(session, case_id)
    active = (
        session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == case.active_task_id)
        ).scalar_one_or_none()
        if case.active_task_id
        else None
    )
    events = list(
        session.execute(
            select(CaseEventLog)
            .where(CaseEventLog.case_id == case_id)
            .order_by(CaseEventLog.id.desc())
            .limit(100)
        ).scalars()
    )
    retrieval = session.execute(
        select(StageArtifact)
        .where(StageArtifact.case_id == case_id, StageArtifact.stage == "retrieval")
        .order_by(StageArtifact.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    persisted_report = session.execute(
        select(CaseReport)
        .where(CaseReport.case_id == case_id)
        .order_by(CaseReport.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    report = None
    if (
        case.status == CaseState.CLOSED_SUCCESS.value
        and persisted_report is not None
        and persisted_report.structured_report.get("schema_version")
        == "assistant-report-v1"
    ):
        report = persisted_report
    return {
        "case": case,
        "active_task": active,
        "evidence": retrieval.payload.get("chunks", []) if retrieval else [],
        "report": report,
        "progress": _assistant_progress(case.status),
        "status_message": _assistant_status_message(case.status),
        "event_count": len(events),
        "recent_event_names": [event.event_type for event in events[:5]],
        "should_poll": active is not None and not is_terminal(CaseState(case.status)),
    }


def _assistant_progress(status_value: str) -> list[dict[str, str]]:
    stages = [
        ("已提交", {"CREATED"}),
        ("输入归一化", {"NORMALIZED"}),
        ("证据检索", {"EVIDENCE_RETRIEVED"}),
        ("分析规划", {"PLAN_GENERATED"}),
        ("专科 Agent 与仲裁", {"SPECIALIST_REVIEWING", "ARBITRATION_REVIEWING"}),
        (
            "引用、逻辑与合规审核",
            {"APPROVED", "REVISION_REQUIRED", "ESCALATED"},
        ),
        (
            "报告生成与关闭",
            {
                "REPORT_GENERATED",
                "CLOSED_SUCCESS",
                "CLOSED_FAILED",
                "CLOSED_ESCALATED",
                "CLOSED_CANCELLED",
            },
        ),
    ]
    current_index = next(
        (index for index, (_, values) in enumerate(stages) if status_value in values),
        0,
    )
    return [
        {
            "label": label,
            "state": (
                "done"
                if index < current_index or status_value == CaseState.CLOSED_SUCCESS.value
                else "current"
                if index == current_index
                else "pending"
            ),
        }
        for index, (label, _) in enumerate(stages)
    ]


def _assistant_status_message(status_value: str) -> str:
    messages = {
        "ESCALATED": "当前等待人工审核，系统不会把该状态展示为成功报告。",
        "REVISION_REQUIRED": "审核要求修订，当前没有可向用户交付的报告。",
        "APPROVED": "审核已通过，正在等待报告生成与安全关闭。",
        "REPORT_GENERATED": "报告已生成但工作流尚未安全关闭，暂不展示结果。",
        "CLOSED_SUCCESS": "工程工作流已安全关闭，可查看审核后的辅助报告。",
        "CLOSED_FAILED": "本次工作流失败，未生成可用报告。",
        "CLOSED_ESCALATED": "本次咨询已由人工终止，未生成可用报告。",
        "CLOSED_CANCELLED": "本次咨询已取消，未生成可用报告。",
    }
    return messages.get(status_value, "系统正在处理，请稍后刷新查看阶段进度。")


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


def _demo_case_context(session: Session, case_id: str) -> dict[str, Any]:
    case = _case_or_404(session, case_id)
    active = (
        session.execute(
            select(WorkflowTask).where(WorkflowTask.task_id == case.active_task_id)
        ).scalar_one_or_none()
        if case.active_task_id
        else None
    )
    events = list(
        session.execute(
            select(CaseEventLog)
            .where(CaseEventLog.case_id == case_id)
            .order_by(CaseEventLog.id.desc())
            .limit(100)
        ).scalars()
    )
    artifacts = list(
        session.execute(
            select(StageArtifact)
            .where(StageArtifact.case_id == case_id)
            .order_by(StageArtifact.id)
        ).scalars()
    )
    latest = {artifact.stage: artifact for artifact in artifacts}
    retrieval = latest.get("retrieval")
    generation = latest.get("generation")
    review_artifact = latest.get("review")
    report = session.execute(
        select(CaseReport)
        .where(CaseReport.case_id == case_id)
        .order_by(CaseReport.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    reviews = list(
        session.execute(
            select(Review).where(Review.case_id == case_id).order_by(Review.id.desc())
        ).scalars()
    )
    citations = list(
        session.execute(
            select(Citation).where(Citation.case_id == case_id).order_by(Citation.id)
        ).scalars()
    )
    failed_stage = next(
        (event for event in events if event.event_type == "stage_failed"), None
    )
    provider_error = failed_stage.detail if failed_stage and failed_stage.detail else None
    can_resume = (
        case.active_task_id is None
        and case.status in {
            CaseState.APPROVED.value,
            CaseState.REVISION_REQUIRED.value,
        }
    )
    return {
        "case": case,
        "active_task": active,
        "events": events,
        "evidence": retrieval.payload.get("chunks", []) if retrieval else [],
        "claims": generation.payload.get("claims", []) if generation else [],
        "generation_version": generation.component_version if generation else None,
        "review_payload": review_artifact.payload if review_artifact else None,
        "reviews": reviews,
        "citations": citations,
        "report": report,
        "provider_error": provider_error,
        "should_poll": active is not None and not is_terminal(CaseState(case.status)),
        "can_resume": can_resume,
    }


app = create_app()
