"""P1-A structured trace exporter and scenario tests."""

from __future__ import annotations

import json

import httpx
from sqlalchemy import update

from medidiag.agents.llm_client import LLMClient
from medidiag.compliance.status import ComplianceStatus, is_compliance_hit
from medidiag.db.models import StageArtifact
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.observability.scenarios import generate_trace_examples
from medidiag.observability.trace_exporter import TraceExporter
from medidiag.workflow.deepseek_provider import DeepSeekWorkflowProvider
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


def _load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exporter_emits_required_fields_and_redacted_links(tmp_path) -> None:
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'trace.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session,
            "Deidentified trace exporter fixture.",
            "secret-idempotency-value",
            "trace-test",
        )
        executor.start_workflow(
            session, case.case_id, "case_workflow", "workflow-trace", "input-hash"
        )
        case_id = case.case_id
    SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="trace-worker"
    ).run_once()

    with factory() as session:
        result = TraceExporter().export_case(
            session,
            case_id,
            raw_dir=tmp_path / "raw",
            summary_dir=tmp_path / "summary",
        )
    engine.dispose()

    raw = _load_jsonl(result.raw_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    required = {
        "event_id", "trace_id", "case_id", "task_id", "agent_run_id",
        "from_state", "to_state", "trigger_subject", "worker_id", "attempt",
        "input_hash", "output_hash", "component_version", "started_at",
        "ended_at", "latency_ms", "retry_count", "error_code", "retryable",
        "action", "alert", "compliance_hit", "citation_verdicts",
        "human_decision",
    }
    assert raw
    assert all(required.issubset(item) for item in raw)
    assert all(item["trace_id"] == result.trace_id for item in raw)
    assert "secret-idempotency-value" not in result.raw_path.read_text(encoding="utf-8")
    created = next(item for item in raw if item["event_type"] == "case_created")
    assert created["detail"]["idempotency_key"] == "[REDACTED]"
    assert TraceExporter._sanitize({"deepseek_api_key": "key-value"}) == {
        "deepseek_api_key": "[REDACTED]"
    }
    assert TraceExporter._sanitize("Bearer token-value") == "Bearer [REDACTED]"
    stage_events = [item for item in raw if item["event_type"] == "stage_completed"]
    assert all(item["task_id"] for item in stage_events)
    normalize = next(item for item in stage_events if item["stage"] == "normalize")
    generation = next(item for item in stage_events if item["stage"] == "generation")
    assert normalize["agent_run_ids"] == []
    assert len(generation["agent_run_ids"]) == 2
    assert summary["raw_event_ids"] == [item["event_id"] for item in raw]
    assert summary["raw_file"] == result.raw_path.name
    assert summary["final_state"] == "CLOSED_SUCCESS"
    assert "text" not in summary["evidence"][0]
    assert "text" not in summary["claims"][0]


def test_generate_reproducible_trace_scenarios(tmp_path) -> None:
    results = generate_trace_examples(tmp_path / "traces")
    by_name = {item.scenario: item for item in results}
    assert set(by_name) == {"success", "lease_recovery", "review_escalation", "dual_specialist_negative"}
    assert all(item.final_state == "CLOSED_SUCCESS" for item in results)

    success = json.loads(
        by_name["success"].export.summary_path.read_text(encoding="utf-8")
    )
    assert success["report"] is not None
    assert success["evidence"]
    assert success["citation_verdicts"]

    recovery = json.loads(
        by_name["lease_recovery"].export.summary_path.read_text(encoding="utf-8")
    )
    assert recovery["recovery"]["lease_reclaimed_event_ids"]
    assert recovery["recovery"]["lease_lost_event_ids"]

    escalation = json.loads(
        by_name["review_escalation"].export.summary_path.read_text(encoding="utf-8")
    )
    assert escalation["human_decisions"][0]["decision"] == "APPROVED"
    assert any(
        item["to_state"] == "ESCALATED" for item in escalation["timeline"]
    )


def test_dual_specialist_trace_records_negative_case(tmp_path) -> None:
    result = next(
        item for item in generate_trace_examples(tmp_path / "traces")
        if item.scenario == "dual_specialist_negative"
    )
    summary = json.loads(result.export.summary_path.read_text(encoding="utf-8"))
    assert [item["specialty"] for item in summary["agents"]] == ["cardiology", "pulmonology"]
    assert summary["arbitration"]["verdict"] == "NO_CLEAR_SPECIALIST_GAIN"
    assert summary["arbitration"]["conflicts"]


# ===== 合规状态判定（C8）=====


def _live_provider_review_status() -> str:
    """取 live provider 实际返回的合规状态，而不是复制一个字面量。"""

    def fake_post(url: str, **kwargs) -> httpx.Response:
        content = json.dumps(
            {
                "claims": [
                    {
                        "text": "现有演示证据不足以形成确定性医学结论，需由合格临床人员复核。",
                        "citation_chunk_ids": ["live_demo_evidence"],
                        "confidence": 0.2,
                    }
                ],
                "uncertainty": "本输出受本地演示证据限制，不构成医疗建议。",
                "risk_flags": ["qualified_clinician_review_required"],
            }
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            headers={"x-request-id": "req-compliance-1"},
            request=httpx.Request("POST", url),
        )

    provider = DeepSeekWorkflowProvider(
        LLMClient(api_key="test-key", model="demo-model", post=fake_post)
    )
    retrieval = {
        "chunks": [
            {
                "chunk_id": "live_demo_evidence",
                "source": "fixture",
                "text": "Fixture evidence for a non-diagnostic engineering demonstration.",
            }
        ]
    }
    generation = provider.generate(
        "Deidentified demo input.", retrieval, provider.plan("q", retrieval)
    ).payload
    review = provider.review(
        generation, provider.arbitrate(generation, retrieval)
    )
    return review["compliance_status"]


def test_live_provider_normal_review_is_not_a_compliance_hit() -> None:
    """live provider 返回 PASS_WITH_DEMO_LIMITATION，不含 'ED' 前缀词。

    旧的 startswith("PASSED") 约定会把每一次正常的实时复核标成合规命中，
    而确定性 provider 的 PASSED_NON_DIAGNOSTIC_FIXTURE 让测试永远看不到。
    """
    status = _live_provider_review_status()
    assert status == ComplianceStatus.PASS_WITH_DEMO_LIMITATION.value
    assert status.startswith("PASSED") is False
    assert is_compliance_hit(status) is False


def test_compliance_hit_covers_every_registered_status() -> None:
    assert is_compliance_hit(ComplianceStatus.BLOCKED.value) is True
    assert is_compliance_hit(ComplianceStatus.PASSED_NON_DIAGNOSTIC_FIXTURE.value) is False
    assert is_compliance_hit(None) is False
    # 未登记的状态按命中处理：宁可进入人工视野，也不要被静默放行。
    assert is_compliance_hit("SOME_FUTURE_STATUS") is True


def test_exported_trace_does_not_flag_a_normal_live_review(tmp_path) -> None:
    """端到端：带 live provider 合规状态的 review 事件不得标记 compliance_hit。"""
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'compliance.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(
            session, "Deidentified compliance fixture.", "idem-compliance", "trace-test"
        )
        executor.start_workflow(
            session, case.case_id, "case_workflow", "workflow-compliance", "input-hash"
        )
        case_id = case.case_id
    SingleMachineWorker(
        factory, DeterministicWorkflowProvider(), worker_id="compliance-worker"
    ).run_once()

    live_status = _live_provider_review_status()
    with factory() as session:
        artifact = (
            session.query(StageArtifact)
            .filter_by(case_id=case_id, stage="review")
            .one()
        )
        payload = dict(artifact.payload)
        payload["compliance_status"] = live_status
        session.execute(
            update(StageArtifact)
            .where(StageArtifact.id == artifact.id)
            .values(payload=payload)
        )
        session.commit()
        result = TraceExporter().export_case(
            session, case_id, raw_dir=tmp_path / "raw", summary_dir=tmp_path / "summary"
        )
    engine.dispose()

    review_events = [
        item for item in _load_jsonl(result.raw_path) if item["stage"] == "review"
    ]
    assert review_events
    assert all(item["compliance_status"] == live_status for item in review_events)
    assert not any(item["compliance_hit"] for item in review_events)
