"""P1-B 服务端渲染演示和 HTMX 行为测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medidiag.api.app import create_app
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


@pytest.fixture
def demo_runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'demo.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    app = create_app(session_factory=factory)
    with TestClient(app) as client:
        yield client, factory
    engine.dispose()


def _demo_create(client: TestClient) -> str:
    response = client.post(
        "/demo/cases",
        headers={"HX-Request": "true"},
        data={
            "question": "Deidentified simulated case for the demo workflow.",
            "input_kind": "deidentified_simulation",
            "source_ref": "",
        },
    )
    assert response.status_code == 204, response.text
    location = response.headers["HX-Redirect"]
    return location.rsplit("/", 1)[-1]


def test_demo_home_and_non_htmx_create_fallback(demo_runtime) -> None:
    client, _ = demo_runtime
    home = client.get("/demo")
    assert home.status_code == 200
    assert "MediDiag EvidenceFlow" in home.text
    assert 'hx-post="/demo/cases"' in home.text
    assert 'method="post" action="/demo/cases"' in home.text
    assert '/static/htmx.min.js' in home.text
    assert "独立 worker 模式" in home.text
    assert "不会进入工作流" in home.text

    created = client.post(
        "/demo/cases",
        data={
            "question": "Deidentified fallback form submission for demo testing.",
            "input_kind": "deidentified_simulation",
            "source_ref": "",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/demo/cases/")


def test_demo_home_shows_attached_runtime_mode(demo_runtime) -> None:
    client, _ = demo_runtime
    client.app.state.demo_runtime = {
        "label": "DeepSeek 实时起草",
        "detail": "生成阶段仅使用远程提供方。",
    }

    home = client.get("/demo")

    assert home.status_code == 200
    assert "DeepSeek 实时起草" in home.text
    assert "生成阶段仅使用远程提供方" in home.text


def test_active_polling_stops_and_final_report_is_rendered(demo_runtime) -> None:
    client, factory = demo_runtime
    case_id = _demo_create(client)
    active = client.get(f"/demo/cases/{case_id}")
    assert active.status_code == 200
    assert 'hx-trigger="every 2s"' in active.text

    result = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="demo-worker"
    ).run_once()
    assert result.final_state == "CLOSED_SUCCESS"

    final = client.get(f"/demo/cases/{case_id}")
    assert final.status_code == 200
    assert 'hx-trigger="every 2s"' not in final.text
    assert "SUPPORTED" in final.text
    assert "deterministic_fixture" in final.text
    assert "确定性 fixture 未生成明确诊断，仅用于工作流测试。" in final.text
    assert "不构成医疗建议" in final.text


def test_escalated_case_stops_polling_and_accepts_human_action(demo_runtime) -> None:
    client, factory = demo_runtime
    case_id = _demo_create(client)
    result = SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(review_verdict="ESCALATED"),
        worker_id="demo-review-worker",
    ).run_once()
    assert result.final_state == "ESCALATED"

    escalated = client.get(f"/demo/cases/{case_id}")
    assert 'hx-trigger="every 2s"' not in escalated.text
    assert "人工处置" in escalated.text
    assert f'hx-post="/demo/cases/{case_id}/human-decisions"' in escalated.text

    approved = client.post(
        f"/demo/cases/{case_id}/human-decisions",
        headers={"HX-Request": "true"},
        data={"decision": "APPROVED", "reason": "Reviewed demo fixture."},
    )
    assert approved.status_code == 200
    assert "启动恢复任务" in approved.text

    resumed = client.post(
        f"/demo/cases/{case_id}/workflow",
        headers={"HX-Request": "true"},
    )
    assert resumed.status_code == 200
    assert 'hx-trigger="every 2s"' in resumed.text


def test_demo_validation_error_is_html_and_css_is_responsive(demo_runtime) -> None:
    client, _ = demo_runtime
    invalid = client.post(
        "/demo/cases",
        headers={"HX-Request": "true"},
        data={
            "question": "Public benchmark entry without a source reference.",
            "input_kind": "public_dataset",
            "source_ref": "",
        },
    )
    assert invalid.status_code == 400
    assert "CASE_INVALID_INPUT" in invalid.text
    assert "error-banner" in invalid.text

    css = client.get("/static/demo.css")
    assert css.status_code == 200
    assert "@media (max-width: 600px)" in css.text
    assert "overflow-wrap: anywhere" in css.text
    htmx = client.get("/static/htmx.min.js")
    assert htmx.status_code == 200
    assert "htmx" in htmx.text[:500].lower()


def test_demo_rejects_identifier_input_without_echoing_identifier(demo_runtime) -> None:
    client, _ = demo_runtime
    identifier = "patient@example.com"

    rejected = client.post(
        "/demo/cases",
        headers={"HX-Request": "true"},
        data={
            "question": f"Deidentified simulation contact: {identifier}",
            "input_kind": "deidentified_simulation",
            "source_ref": "",
        },
    )

    assert rejected.status_code == 400
    assert "CASE_INPUT_NOT_DEIDENTIFIED" in rejected.text
    assert identifier not in rejected.text
