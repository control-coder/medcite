"""应用契约：正常、空证据和失败路径，均不调用外部模型。"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from medidiag.api.app import create_app
from medidiag.db.models import StageArtifact
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


class EmptyProvider(DeterministicWorkflowProvider):
    def retrieve(self, normalized_query):
        return {"query": normalized_query, "chunks": []}

    def generate(self, *args):
        raise AssertionError("无证据不允许生成")


class FailedProvider(DeterministicWorkflowProvider):
    def retrieve(self, normalized_query):
        raise MediDiagError("COMPLIANCE_BLOCKED", detail="不应暴露的上游信息")


@pytest.fixture
def runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'contract.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    with TestClient(create_app(session_factory=factory)) as client:
        yield client, factory
    engine.dispose()


def submit(client, confirmed=True):
    return client.post("/api/v1/consultations", headers={"Idempotency-Key": "contract"}, json={
        "symptoms": "公开模拟输入：轻微不适，仅用于工程验证。", "duration": "模拟两天",
        "input_kind": "deidentified_simulation", "non_sensitive_confirmed": confirmed,
    })


@pytest.mark.parametrize("provider,outcome", [
    (DeterministicWorkflowProvider(), "ready"), (EmptyProvider(), "insufficient_evidence"),
    (FailedProvider(), "failed"),
])
def test_three_outcomes(runtime, provider, outcome):
    client, factory = runtime
    created = submit(client)
    assert created.status_code == 201
    case_id = created.json()["case_id"]
    prefix = f"/api/v1/cases/{case_id}"
    assert client.get(prefix + "/analysis").json()["outcome"] == "processing"
    assert client.post(prefix + "/workflow", headers={"Idempotency-Key": "run"}).status_code == 200
    SingleMachineWorker(factory, provider).run_once()
    result = client.get(prefix + "/analysis").json()
    assert result["outcome"] == outcome
    assert "不应暴露" not in str(result)
    if outcome == "ready":
        ids = {item["chunk_id"] for item in result["evidence"]}
        assert all(set(item["evidence_ids"]) <= ids for item in result["claims"])
        # 未绑定引用不能继续展示，原报告不修改。
        with factory() as session:
            artifact = session.scalar(select(StageArtifact).where(StageArtifact.stage == "retrieval"))
            artifact.payload = {"chunks": []}
            session.commit()
        assert client.get(prefix + "/analysis").json()["outcome"] == "insufficient_evidence"
    else:
        assert result["claims"] == []
        assert result["summary"] is None


def test_confirmation_required(runtime):
    client, _ = runtime
    assert submit(client, False).status_code == 422
