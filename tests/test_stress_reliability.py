"""可靠性压测脚本：核对逻辑必须能报告违规，小规模端到端运行必须通过。"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import update

from medidiag.db.models import Case, CaseReport, StageArtifact, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.executor import WorkflowExecutor
from scripts import stress_reliability as sr


def make_args(**overrides) -> argparse.Namespace:
    values = dict(
        backend="sqlite", workers=2, cases=6, kills=1, freezes=0, stage_delay_ms=150, lease_seconds=2,
        freeze_extra_s=1.0, event_interval_s=1.0, first_event_after_s=0.6, timeout_s=90, seed=1,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_collect_reports_violations(tmp_path: Path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(session, "公开模拟输入", uuid.uuid4().hex, "t")
        task = executor.start_workflow(session, case.case_id, "case_workflow", "run", "hash")
        case_id, task_id = case.case_id, task.task_id
        session.execute(update(Case).where(Case.case_id == case_id).values(status="CLOSED_SUCCESS"))
        session.execute(update(WorkflowTask).where(WorkflowTask.task_id == task_id).values(status="RUNNING"))
        for attempt in (0, 1):  # 同一阶段被提交两次 = 重复副作用
            session.add(StageArtifact(
                artifact_id=f"a{attempt}", case_id=case_id, task_id=task_id, stage="normalize", attempt=attempt,
                payload={}, input_hash="x", output_hash="y", component_version="v", latency_ms=1))
        for index in range(2):  # 两个版本的报告（同版本重复已被唯一约束挡住）
            session.add(CaseReport(
                report_id=f"r{index}", case_id=case_id, version=index + 1, structured_report={}, risk_warnings=[],
                compliance_status="x", generation_version="v"))
        session.commit()
    fleet = SimpleNamespace(workdir=tmp_path, invocation_log="", lease_log="")
    report = sr.collect(factory, [case_id], fleet, [], 1.0, 0, 0, 0, make_args(), None)
    violations = report["side_effects"]["violations"]
    assert violations["cases_without_exactly_one_report"] == 1
    assert violations["duplicate_stage_artifacts"] == 1
    assert violations["tasks_left_running_or_pending"] == 1
    assert report["passed"] is False
    engine.dispose()


def test_small_chaos_run_recovers_everything(tmp_path: Path):
    url = f"sqlite:///{(tmp_path / 'run.db').as_posix()}"
    report = sr.run_scenario(url, make_args())
    assert report["injected"]["kills"] == 1
    assert report["outcome"]["terminal_success_rate"] == 1.0
    assert report["outcome"]["lease_reclaims"] >= 1
    assert all(v == 0 for v in report["side_effects"]["violations"].values())
    assert report["side_effects"]["reexecuted_provider_calls"] >= 1
