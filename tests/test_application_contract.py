"""应用契约：正常、空证据和失败路径，均不调用外部模型。"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from medidiag.api.app import create_app
from medidiag.db.models import CaseEventLog, CaseReport, StageArtifact
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.errors import MediDiagError
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.provider_runtime import ProviderResponse
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


class UsageProvider(DeterministicWorkflowProvider):
    def generate(self, *args):
        return ProviderResponse(payload=super().generate(*args),
                                metadata={"usage": {"input_tokens": 23, "output_tokens": 11}})


def test_projection_hides_unlinked_summary_and_reports_only_recorded_usage(runtime):
    client, factory = runtime
    case_id = submit(client).json()["case_id"]
    prefix = f"/api/v1/cases/{case_id}"
    client.post(prefix + "/workflow", headers={"Idempotency-Key": "usage"})
    SingleMachineWorker(factory, UsageProvider()).run_once()
    with factory() as session:
        report = session.scalar(select(CaseReport).where(CaseReport.case_id == case_id))
        data = dict(report.structured_report)
        data["summary"] = "此摘要没有引用，不应该暴露"
        data["claims"] = data["claims"] + [{"claim_id": "bad", "text": "无支持的结论",
                                          "citation_chunk_ids": ["missing"]}]
        report.structured_report = data
        session.add(CaseEventLog(case_id=case_id, event_type="stage_completed", trigger_subject="system",
                                detail={"task_id": "another_task", "usage": {"input_tokens": 999}}))
        session.commit()
    result = client.get(prefix + "/analysis").json()
    assert result["outcome"] == "ready"
    assert "不应该暴露" not in result["summary"] and "无支持的结论" not in result["summary"]
    assert all(claim["claim_id"] != "bad" for claim in result["claims"])
    observation = result["observation"]
    assert observation["provider_attempts"] == 7
    assert observation["recorded_stage_latency_ms"] >= 0
    assert observation["recorded_input_tokens"] == 23
    assert observation["recorded_output_tokens"] == 11
    assert observation["usage_status"] == "partial"
    assert observation["cost_usd"] is None


def test_unrecorded_usage_is_unknown_not_zero(runtime):
    client, factory = runtime
    case_id = submit(client).json()["case_id"]
    prefix = f"/api/v1/cases/{case_id}"
    client.post(prefix + "/workflow", headers={"Idempotency-Key": "unknown"})
    SingleMachineWorker(factory, DeterministicWorkflowProvider()).run_once()
    observation = client.get(prefix + "/analysis").json()["observation"]
    assert observation["recorded_input_tokens"] is None
    assert observation["usage_status"] == "not_recorded"
    assert observation["cost_usd"] is None
