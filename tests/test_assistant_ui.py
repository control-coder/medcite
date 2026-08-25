"""P6 医疗助手用户视图、安全报告门禁与 Provider smoke 测试。"""

from __future__ import annotations

from uuid import uuid4

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from sqlalchemy import select

from medidiag.api.app import create_app
from medidiag.cli import main
from medidiag.db.models import Case, CaseReport
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


@pytest.fixture
def assistant_runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'assistant.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    app = create_app(session_factory=factory)
    with TestClient(app) as client:
        yield client, factory
    engine.dispose()


def _create(client: TestClient, *, htmx: bool = True) -> str:
    response = client.post(
        "/assistant/cases",
        headers={"HX-Request": "true"} if htmx else {},
        data={
            "question": "Deidentified simulated question for assistant UI verification.",
            "input_kind": "deidentified_simulation",
            "source_ref": "",
        },
        follow_redirects=False,
    )
    assert response.status_code == (204 if htmx else 303), response.text
    location = response.headers["HX-Redirect" if htmx else "location"]
    return location.rsplit("/", 1)[-1]


def _insert_report(factory, case_id: str) -> None:
    with factory() as session:
        session.add(
            CaseReport(
                report_id=f"report-{uuid4().hex}",
                case_id=case_id,
                version=99,
                structured_report={
                    "schema_version": "assistant-report-v1",
                    "title": "不应展示的报告",
                    "summary": "SENSITIVE_REPORT_SENTINEL",
                    "claims": [],
                    "risk_warnings": ["测试"],
                    "limitations": ["测试"],
                    "next_steps": ["测试"],
                    "disclaimer": "不构成医疗建议。",
                },
                risk_warnings=["测试"],
                compliance_status="PASSED",
                generation_version="test-only",
            )
        )
        session.commit()


def test_assistant_home_and_no_javascript_fallback(assistant_runtime) -> None:
    client, _ = assistant_runtime
    home = client.get("/assistant")
    assert home.status_code == 200
    assert "医疗信息助手" in home.text
    assert "不提供真实医疗诊断" in home.text
    assert 'method="post" action="/assistant/cases"' in home.text
    assert 'hx-post="/assistant/cases"' in home.text
    assert 'href="/demo"' in home.text

    case_id = _create(client, htmx=False)
    page = client.get(f"/assistant/cases/{case_id}")
    assert page.status_code == 200
    assert 'hx-trigger="every 2s"' in page.text
    assert "浏览器未启用 JavaScript" in page.text
    assert f'href="/assistant/cases/{case_id}"' in page.text


def test_assistant_htmx_redirect_and_deidentification_error_does_not_echo(assistant_runtime) -> None:
    client, _ = assistant_runtime
    case_id = _create(client)
    assert case_id

    secret = "patient-secret@example.com"
    rejected = client.post(
        "/assistant/cases",
        headers={"HX-Request": "true"},
        data={
            "question": f"This simulated case includes {secret} and must be rejected.",
            "input_kind": "deidentified_simulation",
            "source_ref": "",
        },
    )
    assert rejected.status_code == 400
    assert "CASE_INPUT_NOT_DEIDENTIFIED" in rejected.text
    assert secret not in rejected.text


def test_closed_success_shows_safe_report_evidence_and_trace(assistant_runtime) -> None:
    client, factory = assistant_runtime
    case_id = _create(client)
    result = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="assistant-worker"
    ).run_once()
    assert result.final_state == "CLOSED_SUCCESS"

    page = client.get(f"/assistant/cases/{case_id}")
    assert page.status_code == 200
    assert 'hx-trigger="every 2s"' not in page.text
    assert "确定性证据审阅报告" in page.text
    assert "SUPPORTED" in page.text
    assert "证据摘要" in page.text
    assert "Trace 摘要" in page.text
    assert "reasoning_content" not in page.text
    assert "api_key" not in page.text.lower()
    assert "人工处置" not in page.text


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("ESCALATED", "等待人工审核"),
        ("REPORT_GENERATED", "尚未安全关闭"),
        ("CLOSED_FAILED", "工作流失败"),
        ("CLOSED_ESCALATED", "人工终止"),
    ],
)
def test_non_success_states_never_render_persisted_report(
    assistant_runtime, state: str, expected: str
) -> None:
    client, factory = assistant_runtime
    case_id = _create(client)
    with factory() as session:
        case = session.execute(select(Case).where(Case.case_id == case_id)).scalar_one()
        case.status = state
        case.active_task_id = None
        session.commit()
    _insert_report(factory, case_id)

    page = client.get(f"/assistant/cases/{case_id}")
    assert page.status_code == 200
    assert expected in page.text
    assert "SENSITIVE_REPORT_SENTINEL" not in page.text
    assert "当前没有可展示的成功报告" in page.text
    assert "人工处置" not in page.text


def test_demo_remains_available(assistant_runtime) -> None:
    client, _ = assistant_runtime
    page = client.get("/demo")
    assert page.status_code == 200
    assert "病例工作台" in page.text
    assert 'href="/assistant"' in page.text


def test_provider_smoke_fake_is_explicit_and_redacted() -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "provider-smoke" in help_result.output

    result = runner.invoke(main, ["provider-smoke", "--provider", "fake_offline"])
    assert result.exit_code == 0, result.output
    assert '"profile": "fake_offline"' in result.output
    assert '"response_id_present": true' in result.output
    assert "不代表医疗工作流" in result.output
    assert "reasoning" not in result.output.lower()
    assert "api_key" not in result.output.lower()
