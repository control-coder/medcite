"""M14：任务事件 SSE 推送（阶段顺序、结束关闭、归属隔离、心跳与最长时长）。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from medidiag.api import app as app_module
from medidiag.api.app import create_app
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_SSE_POLL_SECONDS", 0.02)
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'sse.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    app = create_app(session_factory=factory)
    with TestClient(app) as client:
        yield app, client, factory
    engine.dispose()


def _case(client: TestClient) -> str:
    response = client.post(
        "/api/v1/cases", headers={"Idempotency-Key": "sse-case"},
        json={"question": "Deidentified simulated case for SSE testing.",
              "input_kind": "deidentified_simulation"})
    assert response.status_code == 201, response.text
    return response.json()["case_id"]


def _frames(text: str) -> list[dict[str, str]]:
    out = []
    for block in text.split("\n\n"):
        if block.strip():
            out.append(dict(line.split(": ", 1) if ": " in line else (line, "") for line in block.splitlines()))
    return out


def _run_to_close(client: TestClient, factory, case_id: str) -> None:
    client.post(f"/api/v1/cases/{case_id}/workflow", headers={"Idempotency-Key": "sse-wf"})
    result = SingleMachineWorker(factory, DeterministicWorkflowProvider(), worker_id="sse-worker").run_once()
    assert result.final_state == "CLOSED_SUCCESS"


def test_finished_case_streams_stages_in_order_then_closes(runtime) -> None:
    _, client, factory = runtime
    case_id = _case(client)
    _run_to_close(client, factory, case_id)
    response = client.get(f"/api/v1/cases/{case_id}/events/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _frames(response.text)
    assert frames[-1] == {"event": "end", "data": '{"reason": "finished"}'}
    events = [json.loads(f["data"]) for f in frames if f.get("event") == "case_event"]
    ids = [e["event_id"] for e in events]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    stages = [e["detail"]["stage"] for e in events if e["event_type"] == "stage_completed"]
    assert stages[0] == "normalize" and len(stages) >= 3
    assert events[-1]["to_status"] == "CLOSED_SUCCESS"
    # 与轮询端点的事件一致
    polled = client.get(f"/api/v1/cases/{case_id}/events?limit=200").json()["items"]
    assert ids == [e["event_id"] for e in polled]


def test_resume_from_last_event_id(runtime) -> None:
    _, client, factory = runtime
    case_id = _case(client)
    _run_to_close(client, factory, case_id)
    all_ids = [e["event_id"] for e in client.get(f"/api/v1/cases/{case_id}/events?limit=200").json()["items"]]
    response = client.get(f"/api/v1/cases/{case_id}/events/stream", headers={"Last-Event-ID": str(all_ids[1])})
    ids = [json.loads(f["data"])["event_id"] for f in _frames(response.text) if f.get("event") == "case_event"]
    assert ids == all_ids[2:]


def test_other_owner_and_missing_case_rejected(runtime) -> None:
    app, client, _ = runtime
    case_id = _case(client)
    with TestClient(app) as stranger:  # 独立 cookie，即另一个匿名 owner
        denied = stranger.get(f"/api/v1/cases/{case_id}/events/stream")
        assert denied.status_code == 404
        assert denied.json()["code"] == "CASE_NOT_FOUND"
        assert "text/event-stream" not in denied.headers["content-type"]
    assert client.get("/api/v1/cases/missing/events/stream").status_code == 404


def test_unfinished_case_sends_heartbeat_and_times_out(runtime, monkeypatch) -> None:
    _, client, _ = runtime
    case_id = _case(client)
    monkeypatch.setattr(app_module, "_SSE_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(app_module, "_SSE_MAX_SECONDS", 0.3)
    response = client.get(f"/api/v1/cases/{case_id}/events/stream")
    assert ": keep-alive" in response.text
    assert response.text.rstrip().endswith('{"reason": "timeout"}')
