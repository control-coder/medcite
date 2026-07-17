"""Live-provider failure behavior for the server-rendered demo."""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import select

from medidiag.api.app import create_app
from medidiag.db.models import Case, CaseEventLog, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.provider_runtime import ProviderCallRunner
from medidiag.workflow.worker import SingleMachineWorker


class UnavailableGenerationProvider(DeterministicWorkflowProvider):
    """Keep local demo stages deterministic and fail only the live boundary."""

    version = "deepseek-live-demo:test-unavailable"

    def generate(self, question: str, retrieval: dict, plan: dict) -> dict:
        request = httpx.Request("POST", "https://provider.invalid/v1/chat/completions")
        response = httpx.Response(
            502,
            request=request,
            headers={"x-request-id": "req-demo-unavailable"},
        )
        raise httpx.HTTPStatusError("gateway unavailable", request=request, response=response)


def test_demo_stops_after_bounded_generation_failure_and_renders_safe_notice(tmp_path) -> None:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'demo-failure.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    app = create_app(session_factory=factory)

    try:
        with TestClient(app) as client:
            created = client.post(
                "/demo/cases",
                headers={"HX-Request": "true"},
                data={
                    "question": "Deidentified simulated case for provider-failure demo.",
                    "input_kind": "deidentified_simulation",
                    "source_ref": "",
                },
            )
            assert created.status_code == 204
            case_id = created.headers["HX-Redirect"].rsplit("/", 1)[-1]

            result = SingleMachineWorker(
                factory,
                UnavailableGenerationProvider(
                    version="deepseek-live-demo:test-unavailable"
                ),
                worker_id="demo-provider-failure-worker",
                call_runner=ProviderCallRunner(max_attempts=1, sleep=lambda _: None),
            ).run_once()
            assert result.final_state == "ESCALATED"

            with factory() as session:
                case = session.execute(
                    select(Case).where(Case.case_id == case_id)
                ).scalar_one()
                task = session.execute(
                    select(WorkflowTask).where(WorkflowTask.case_id == case_id)
                ).scalar_one()
                failed = session.execute(
                    select(CaseEventLog).where(
                        CaseEventLog.case_id == case_id,
                        CaseEventLog.event_type == "stage_failed",
                    )
                ).scalar_one()
                assert case.status == "ESCALATED"
                assert case.active_task_id is None
                assert task.status == "FAILED"
                assert task.error_code == "PROVIDER_UNAVAILABLE"
                assert "no report generated" in task.error_message
                assert failed.detail == {
                    "task_id": task.task_id,
                    "stage": "generation",
                    "attempt": 0,
                    "error_code": "PROVIDER_UNAVAILABLE",
                    "component_version": "deepseek-live-demo:test-unavailable",
                    "provider_request_id": "req-demo-unavailable",
                    "provider_retry_count": 0,
                    "retry_decision": "exhausted",
                    "http_status": 502,
                }

            page = client.get(f"/demo/cases/{case_id}")
            assert page.status_code == 200
            assert 'hx-trigger="every 2s"' not in page.text
            assert "外部生成组件当前不可用" in page.text
            assert "PROVIDER_UNAVAILABLE" in page.text
            assert "未生成报告或医疗结论" in page.text
            assert "req-demo-unavailable" in page.text
    finally:
        engine.dispose()
