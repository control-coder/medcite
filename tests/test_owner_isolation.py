"""所有外部读取/操作入口必须隔离用户，伪造作用域不能获得归属。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from medidiag.api.app import create_app
from medidiag.db.models import Case, WebSession
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


@pytest.fixture
def runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'owners.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    app = create_app(session_factory=factory)
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.get("/api/v1/session")
        bob.get("/api/v1/session")
        yield alice, bob, factory
    engine.dispose()


def create(client, key="same"):
    response = client.post("/api/v1/cases", headers={"Idempotency-Key": key, "X-User-Scope": "forged"},
                           json={"question": "公开模拟病例：仅用于验证用户隔离与任务取消。", "input_kind": "deidentified_simulation"})
    assert response.status_code == 201
    return response.json()["case_id"]


def test_every_route_filters_owner(runtime):
    alice, bob, factory = runtime
    own = create(alice)
    other = create(bob)
    assert own != other
    assert create(alice) == own
    assert [x["case_id"] for x in alice.get("/api/v1/cases").json()["items"]] == [own]
    for suffix in ("", "/events", "/report", "/analysis"):
        assert bob.get(f"/api/v1/cases/{own}{suffix}", headers={"X-User-Scope": "forged"}).status_code == 404
    for suffix, body in (("workflow", None), ("cancel", None), ("human-decisions", {"decision": "APPROVED", "reason": "模拟审核"})):
        assert bob.post(f"/api/v1/cases/{own}/{suffix}", headers={"Idempotency-Key": "attack"}, json=body).status_code == 404
    for prefix in ("assistant", "demo"):
        for suffix in ("", "/status"):
            assert bob.get(f"/{prefix}/cases/{own}{suffix}").status_code == 404
        assert own not in bob.get(f"/{prefix}").text
    assert bob.post(f"/demo/cases/{own}/workflow").status_code == 404
    assert bob.post(f"/demo/cases/{own}/human-decisions", data={"decision": "APPROVED", "reason": "模拟审核"}).status_code == 404
    with factory() as session:
        assert session.scalar(select(Case.owner_id).where(Case.case_id == own)) != "forged"
        token = alice.cookies.get("medidiag_session")
        assert session.get(WebSession, token) is None


def test_unowned_legacy_is_not_claimed(runtime):
    alice, _, factory = runtime
    with factory() as session:
        legacy = WorkflowExecutor().create_case(session, "原有无归属的公开模拟输入。", "legacy", "local-demo")
        case_id = legacy.case_id
    assert alice.get(f"/api/v1/cases/{case_id}").status_code == 404
    assert alice.get("/api/v1/cases").json()["items"] == []


def test_session_and_csrf_boundary(runtime):
    alice, bob, _ = runtime
    response = alice.post("/api/v1/cases", headers={"Origin": "https://attacker.invalid"})
    assert response.status_code == 403
    assert alice.get("/api/v1/cases", headers={"Host": "attacker.invalid"}).status_code == 400
    assert alice.get("/api/v1/session").headers["cache-control"] == "no-store"
    bob.cookies.set("medidiag_session", "invalid", domain="testserver.local", path="/")
    assert bob.get("/api/v1/cases").status_code == 401


def test_cancel_fences_running_worker(runtime):
    alice, _, factory = runtime
    case_id = create(alice)
    task = alice.post(f"/api/v1/cases/{case_id}/workflow", headers={"Idempotency-Key": "run"}).json()
    started, release = Event(), Event()
    class Slow(DeterministicWorkflowProvider):
        def retrieve(self, query):
            started.set()
            assert release.wait(10)
            return super().retrieve(query)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(SingleMachineWorker(factory, Slow()).run_task, task["task_id"])
        assert started.wait(10)
        try:
            assert alice.post(f"/api/v1/cases/{case_id}/cancel").status_code == 200
        finally:
            release.set()
        with pytest.raises(MediDiagError) as exc:
            future.result(timeout=10)
        assert exc.value.code == "TASK_LEASE_LOST"
    assert alice.get(f"/api/v1/cases/{case_id}/analysis").json()["outcome"] == "cancelled"
    assert alice.post(f"/api/v1/cases/{case_id}/cancel").status_code == 200
