"""P0-C FastAPI 契约与端到端测试。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medidiag.api.app import create_app
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


@pytest.fixture
def api_runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    app = create_app(session_factory=factory)
    with TestClient(app) as client:
        yield client, factory
    engine.dispose()


def _create_case(client: TestClient, key: str = "case-key") -> dict:
    response = client.post(
        "/api/v1/cases",
        headers={"Idempotency-Key": key},
        json={
            "question": "Deidentified simulated case for API workflow testing.",
            "input_kind": "deidentified_simulation",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_case_create_is_idempotent(api_runtime) -> None:
    client, _ = api_runtime
    first = _create_case(client)
    second = _create_case(client)
    assert first["case_id"] == second["case_id"]
    assert first["trace_id"] == second["trace_id"]


def test_case_create_requires_idempotency_key(api_runtime) -> None:
    client, _ = api_runtime
    response = client.post(
        "/api/v1/cases",
        json={
            "question": "Deidentified simulated case for API workflow testing.",
            "input_kind": "deidentified_simulation",
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "IDEMPOTENCY_KEY_MISSING"


def test_case_create_rejects_obvious_identifier(api_runtime) -> None:
    client, _ = api_runtime
    response = client.post(
        "/api/v1/cases",
        headers={"Idempotency-Key": "phi-case"},
        json={
            "question": "Contact patient@example.com about this simulated record.",
            "input_kind": "deidentified_simulation",
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "CASE_INPUT_NOT_DEIDENTIFIED"


def test_public_dataset_requires_source_ref(api_runtime) -> None:
    client, _ = api_runtime
    response = client.post(
        "/api/v1/cases",
        headers={"Idempotency-Key": "public-case"},
        json={
            "question": "Public benchmark case without a declared source reference.",
            "input_kind": "public_dataset",
        },
    )
    assert response.status_code == 400
    assert response.json()["code"] == "CASE_INVALID_INPUT"


def test_workflow_start_idempotency_and_conflict(api_runtime) -> None:
    client, _ = api_runtime
    case = _create_case(client)
    url = f"/api/v1/cases/{case['case_id']}/workflow"
    first = client.post(url, headers={"Idempotency-Key": "workflow-1"})
    second = client.post(url, headers={"Idempotency-Key": "workflow-1"})
    conflict = client.post(url, headers={"Idempotency-Key": "workflow-2"})
    assert first.status_code == second.status_code == 200
    assert first.json()["task_id"] == second.json()["task_id"]
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "WORKFLOW_ALREADY_RUNNING"


def test_report_not_ready_and_missing_case(api_runtime) -> None:
    client, _ = api_runtime
    case = _create_case(client)
    response = client.get(f"/api/v1/cases/{case['case_id']}/report")
    assert response.status_code == 409
    assert response.json()["code"] == "REPORT_NOT_READY"
    assert client.get("/api/v1/cases/missing").status_code == 404


def test_api_to_worker_to_report_end_to_end(api_runtime) -> None:
    client, factory = api_runtime
    case = _create_case(client)
    case_id = case["case_id"]
    started = client.post(
        f"/api/v1/cases/{case_id}/workflow",
        headers={"Idempotency-Key": "workflow-1"},
    )
    assert started.status_code == 200
    result = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="api-worker"
    ).run_once()
    assert result.final_state == "CLOSED_SUCCESS"

    current = client.get(f"/api/v1/cases/{case_id}")
    assert current.status_code == 200
    assert current.json()["status"] == "CLOSED_SUCCESS"
    assert current.json()["active_task"] is None

    report = client.get(f"/api/v1/cases/{case_id}/report")
    assert report.status_code == 200
    assert report.json()["structured_report"]["summary"] == (
        "确定性 fixture 未生成明确诊断，仅用于工作流测试。"
    )
    assert report.json()["risk_warnings"]

    first_page = client.get(
        f"/api/v1/cases/{case_id}/events", params={"limit": 2}
    ).json()
    assert len(first_page["items"]) == 2
    assert first_page["next_cursor"] is not None
    second_page = client.get(
        f"/api/v1/cases/{case_id}/events",
        params={"cursor": first_page["next_cursor"], "limit": 200},
    ).json()
    assert second_page["items"]
    assert second_page["items"][0]["event_id"] > first_page["next_cursor"]
    event_types = {
        item["event_type"]
        for item in first_page["items"] + second_page["items"]
    }
    assert {"workflow_started", "lease_acquired", "stage_completed"}.issubset(
        event_types
    )


def test_human_decision_closes_or_resumes_escalated_case(api_runtime) -> None:
    client, factory = api_runtime
    case = _create_case(client)
    case_id = case["case_id"]
    client.post(
        f"/api/v1/cases/{case_id}/workflow",
        headers={"Idempotency-Key": "workflow-escalate"},
    )
    SingleMachineWorker(
        factory,
        DeterministicWorkflowProvider(review_verdict="ESCALATED"),
        worker_id="review-worker",
    ).run_once()
    escalated = client.get(f"/api/v1/cases/{case_id}").json()
    assert escalated["status"] == "ESCALATED"
    assert escalated["available_human_actions"] == [
        "REVISION_REQUIRED", "APPROVED", "CLOSED_ESCALATED",
    ]

    decision = client.post(
        f"/api/v1/cases/{case_id}/human-decisions",
        json={"decision": "APPROVED", "reason": "Fixture reviewed by test human."},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "APPROVED"
    client.post(
        f"/api/v1/cases/{case_id}/workflow",
        headers={"Idempotency-Key": "workflow-resume"},
    )
    resumed = SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="resume-worker"
    ).run_once()
    assert resumed.final_state == "CLOSED_SUCCESS"


def test_human_decision_rejected_outside_escalated(api_runtime) -> None:
    client, _ = api_runtime
    case = _create_case(client)
    response = client.post(
        f"/api/v1/cases/{case['case_id']}/human-decisions",
        json={"decision": "APPROVED", "reason": "Not in escalation state."},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "ILLEGAL_STATE_TRANSITION"
