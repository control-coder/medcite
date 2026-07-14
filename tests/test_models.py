"""数据库模型测试。

覆盖:
1. 8 张表创建
2. Case CRUD + version 乐观锁机制
3. WorkflowTask lease 字段
4. CaseEventLog 事件记录（状态推进 + 事件同事务）
5. AgentRun input_hash 幂等
6. Citation verdict
7. Review round
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from medidiag.db.models import (
    ALL_TABLES,
    AgentRun,
    Base,
    Case,
    CaseEventLog,
    CaseReport,
    Citation,
    Review,
    StageArtifact,
    WorkflowTask,
)
from medidiag.db.session import create_db_engine, get_session_factory, init_db


@pytest.fixture
def engine():
    eng = create_db_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    factory = get_session_factory(engine)
    s = factory()
    yield s
    s.close()


@pytest.fixture
def case(session):
    """创建一个测试病例。"""
    c = Case(
        case_id="c1",
        status="CREATED",
        version=1,
        question="What is the diagnosis?",
        idempotency_key="ik1",
        idempotency_user_scope="user1",
    )
    session.add(c)
    session.commit()
    return c


# ===== 表创建测试 =====


class TestTableCreation:
    def test_all_six_tables_created(self, engine) -> None:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        expected = {
            "cases", "workflow_tasks", "case_event_log", "agent_runs",
            "citations", "reviews", "stage_artifacts", "case_reports",
        }
        assert expected.issubset(tables), f"missing tables: {expected - tables}"

    def test_all_tables_constant(self) -> None:
        assert ALL_TABLES == (
            "cases", "workflow_tasks", "case_event_log",
            "agent_runs", "citations", "reviews", "stage_artifacts",
            "case_reports",
        )

    def test_no_evidence_chunks_table(self, engine) -> None:
        """evidence_chunks 不建数据库表（阶段 1 已作为 JSONL）。"""
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert "evidence_chunks" not in tables

    def test_p0b_unique_constraints_exist(self, engine) -> None:
        inspector = inspect(engine)
        assert "uq_cases_scope_idempotency" in {
            item["name"] for item in inspector.get_unique_constraints("cases")
        }
        assert "uq_workflow_tasks_case_type_idempotency" in {
            item["name"]
            for item in inspector.get_unique_constraints("workflow_tasks")
        }
        assert "uq_agent_runs_execution_identity" in {
            item["name"] for item in inspector.get_unique_constraints("agent_runs")
        }


# ===== Case 模型测试 =====


class TestCase:
    def test_create_case(self, session) -> None:
        c = Case(
            case_id="c1", status="CREATED", version=1,
            question="q", idempotency_key="k1", idempotency_user_scope="u1",
        )
        session.add(c)
        session.commit()
        assert c.id is not None
        assert c.version == 1
        assert c.status == "CREATED"
        assert c.created_at is not None

    def test_case_id_unique(self, session, case) -> None:
        """case_id 唯一约束。"""
        c2 = Case(
            case_id="c1", status="CREATED", version=1,
            question="q2", idempotency_key="k2", idempotency_user_scope="u2",
        )
        session.add(c2)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    def test_scope_and_idempotency_key_unique(self, session, case) -> None:
        duplicate = Case(
            case_id="c2", status="CREATED", version=1,
            question="other", idempotency_key="ik1",
            idempotency_user_scope="user1",
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    def test_active_task_id_is_persisted(self, session, case) -> None:
        case.active_task_id = "task-1"
        session.commit()
        session.refresh(case)
        assert case.active_task_id == "task-1"

    def test_optimistic_lock_success(self, session, case) -> None:
        """乐观锁：version 匹配时更新成功。"""
        result = session.execute(
            update(Case)
            .where(Case.case_id == "c1", Case.version == 1)
            .values(status="NORMALIZED", version=2)
        )
        session.commit()
        assert result.rowcount == 1
        session.refresh(case)
        assert case.status == "NORMALIZED"
        assert case.version == 2

    def test_optimistic_lock_conflict(self, session, case) -> None:
        """乐观锁：version 不匹配时更新失败（rowcount=0）。"""
        # 先把 version 推进到 2
        session.execute(
            update(Case)
            .where(Case.case_id == "c1", Case.version == 1)
            .values(status="NORMALIZED", version=2)
        )
        session.commit()

        # 用旧 version=1 更新，应失败
        result = session.execute(
            update(Case)
            .where(Case.case_id == "c1", Case.version == 1)
            .values(status="EVIDENCE_RETRIEVED", version=3)
        )
        session.commit()
        assert result.rowcount == 0  # 没有行被更新


# ===== WorkflowTask 模型测试 =====


class TestWorkflowTask:
    def test_create_task_with_lease(self, session, case) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        t = WorkflowTask(
            task_id="t1", case_id="c1", task_type="normalize",
            status="RUNNING", lease_owner="worker-1",
            lease_until=now + timedelta(seconds=60),
            heartbeat_at=now, attempt=1,
            input_hash="h1", idempotency_key="ik1",
        )
        session.add(t)
        session.commit()
        assert t.id is not None
        assert t.lease_owner == "worker-1"
        assert t.lease_until is not None
        assert t.attempt == 1

    def test_task_status_values(self, session, case) -> None:
        """任务状态支持 PENDING/RUNNING/SUCCEEDED/FAILED/STALE。"""
        for status in ["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "STALE"]:
            t = WorkflowTask(
                task_id=f"t_{status}", case_id="c1", task_type="normalize",
                status=status, input_hash=f"h_{status}", idempotency_key=f"ik_{status}",
            )
            session.add(t)
        session.commit()
        assert session.query(WorkflowTask).count() == 5

    def test_task_result_json(self, session, case) -> None:
        """result 字段支持 JSON。"""
        t = WorkflowTask(
            task_id="t1", case_id="c1", task_type="retrieve",
            status="SUCCEEDED", input_hash="h1", idempotency_key="ik1",
            result={"chunks": ["kb_001", "kb_002"], "scores": [0.9, 0.8]},
        )
        session.add(t)
        session.commit()
        session.refresh(t)
        assert t.result["chunks"] == ["kb_001", "kb_002"]
        assert t.result["scores"] == [0.9, 0.8]

    def test_workflow_execution_identity_unique(self, session, case) -> None:
        for task_id in ("t1", "t2"):
            session.add(WorkflowTask(
                task_id=task_id, case_id="c1", task_type="normalize",
                status="PENDING", input_hash="h1", idempotency_key="ik1",
            ))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


# ===== CaseEventLog 模型测试 =====


class TestCaseEventLog:
    def test_state_transition_with_event_same_transaction(self, session, case) -> None:
        """状态推进 + 事件日志在同一事务内提交。"""
        # 状态推进
        session.execute(
            update(Case)
            .where(Case.case_id == "c1", Case.version == 1)
            .values(status="NORMALIZED", version=2)
        )
        # 事件日志
        event = CaseEventLog(
            case_id="c1", event_type="state_transition",
            from_status="CREATED", to_status="NORMALIZED",
            trigger_subject="worker", trigger_entity="worker-1",
            detail={"task_id": "t1"},
        )
        session.add(event)
        session.commit()

        assert event.id is not None
        assert event.from_status == "CREATED"
        assert event.to_status == "NORMALIZED"

    def test_event_log_query(self, session, case) -> None:
        """可以按 case_id 查询事件历史。"""
        for i in range(3):
            session.add(CaseEventLog(
                case_id="c1", event_type=f"event_{i}",
                trigger_subject="worker",
            ))
        session.commit()
        events = session.query(CaseEventLog).filter_by(case_id="c1").all()
        assert len(events) == 3


# ===== AgentRun 模型测试 =====


class TestAgentRun:
    def test_create_agent_run(self, session, case) -> None:
        ar = AgentRun(
            run_id="r1", case_id="c1", agent_name="diagnosis",
            input_hash="h1", attempt_group="g1",
            status="SUCCEEDED", latency_ms=500,
            output_payload={"diagnosis": "flu", "confidence": 0.85},
        )
        session.add(ar)
        session.commit()
        assert ar.id is not None
        assert ar.output_payload["diagnosis"] == "flu"
        assert ar.latency_ms == 500

    def test_input_hash_idempotency_check(self, session, case) -> None:
        """通过 input_hash 查询是否已有成功结果（幂等）。"""
        ar = AgentRun(
            run_id="r1", case_id="c1", agent_name="diagnosis",
            input_hash="h1", attempt_group="g1",
            status="SUCCEEDED",
        )
        session.add(ar)
        session.commit()

        # 查询是否已有同一 input_hash 的成功结果
        existing = session.query(AgentRun).filter_by(
            case_id="c1", agent_name="diagnosis",
            input_hash="h1", status="SUCCEEDED",
        ).first()
        assert existing is not None
        assert existing.run_id == "r1"

    def test_agent_execution_identity_unique(self, session, case) -> None:
        for run_id in ("r1", "r2"):
            session.add(AgentRun(
                run_id=run_id, case_id="c1", agent_name="diagnosis",
                input_hash="h1", attempt_group="g1", status="PENDING",
            ))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


# ===== Citation 模型测试 =====


class TestCitation:
    def test_create_citation(self, session, case) -> None:
        cit = Citation(
            case_id="c1", claim_text="Patient has influenza.",
            chunk_id="kb_001", verdict="SUPPORTED",
            verifier_model="microsoft/deberta-v3-base-mnli",
            verifier_score=0.92, human_reviewed=False,
        )
        session.add(cit)
        session.commit()
        assert cit.id is not None
        assert cit.verdict == "SUPPORTED"
        assert cit.verifier_score == 0.92

    def test_verdict_values(self, session, case) -> None:
        for verdict in ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]:
            session.add(Citation(
                case_id="c1", claim_text=f"claim_{verdict}",
                chunk_id=f"kb_{verdict}", verdict=verdict,
                verifier_model="test",
            ))
        session.commit()
        assert session.query(Citation).count() == 3


# ===== Review 模型测试 =====


class TestReview:
    def test_create_review(self, session, case) -> None:
        r = Review(
            case_id="c1", review_type="citation",
            reviewer="reviewer-1", result="REVISION_REQUIRED",
            round=1, detail={"unsupported_count": 3},
        )
        session.add(r)
        session.commit()
        assert r.id is not None
        assert r.round == 1
        assert r.detail["unsupported_count"] == 3

    def test_review_round_increment(self, session, case) -> None:
        """审核轮次递增。"""
        for rnd in range(1, 4):
            session.add(Review(
                case_id="c1", review_type="logic",
                reviewer=f"reviewer-{rnd}", result="REVISION_REQUIRED",
                round=rnd,
            ))
        session.commit()
        reviews = session.query(Review).filter_by(case_id="c1").order_by(Review.round).all()
        assert [r.round for r in reviews] == [1, 2, 3]


class TestWorkflowArtifacts:
    def test_stage_artifact_and_report(self, session, case) -> None:
        task = WorkflowTask(
            task_id="t-artifact", case_id="c1", task_type="case_workflow",
            status="RUNNING", input_hash="input", idempotency_key="workflow-1",
        )
        session.add(task)
        session.flush()
        session.add(StageArtifact(
            artifact_id="artifact-1", case_id="c1", task_id=task.task_id,
            stage="normalize", attempt=0, payload={"normalized_query": "q"},
            input_hash="in", output_hash="out", component_version="test-v1",
            latency_ms=1,
        ))
        session.add(CaseReport(
            report_id="report-1", case_id="c1", version=1,
            structured_report={"summary": "draft"},
            risk_warnings=["review_required"], compliance_status="PASSED",
            generation_version="test-v1",
        ))
        session.commit()
        assert session.query(StageArtifact).count() == 1
        assert session.query(CaseReport).one().structured_report["summary"] == "draft"

    def test_orphan_stage_artifact_rejected(self, session) -> None:
        session.add(StageArtifact(
            artifact_id="artifact-orphan", case_id="missing-case",
            task_id="missing-task", stage="normalize", attempt=0,
            payload={}, input_hash="in", output_hash="out",
            component_version="test-v1", latency_ms=1,
        ))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
