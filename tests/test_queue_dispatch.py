"""任务标识派发、重复消费与阶段恢复的离线回归。"""

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from medidiag.db.models import CaseReport, StageArtifact, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory, init_db
from medidiag.workflow.dispatcher import scan_pending
from medidiag.workflow.executor import WorkflowExecutor
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker


@pytest.fixture
def runtime(tmp_path):
    engine = create_db_engine(f"sqlite:///{(tmp_path / 'queue.db').as_posix()}")
    init_db(engine)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    with factory() as session:
        case = executor.create_case(session, "模拟公开问题用于测试数据库队列补偿。", uuid.uuid4().hex, "test")
        task = executor.start_workflow(session, case.case_id, "case_workflow", "run", "hash")
        task_id, case_id = task.task_id, case.case_id
    yield factory, task_id, case_id
    engine.dispose()


def test_duplicate_delivery(runtime):
    factory, task_id, case_id = runtime
    worker = SingleMachineWorker(factory, DeterministicWorkflowProvider())
    assert worker.run_task(task_id).processed
    assert not worker.run_task(task_id).processed
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(CaseReport).where(CaseReport.case_id == case_id)) == 1


def test_dispatch_failure_is_recoverable(runtime):
    factory, task_id, _ = runtime
    def fail(_):
        raise ConnectionError("模拟派发失败")
    assert scan_pending(factory, fail) == {"sent": 0, "failed": 1}
    sent = []
    assert scan_pending(factory, sent.append) == {"sent": 1, "failed": 0}
    assert sent == [task_id]


def test_interrupted_worker_resumes_committed_stage(runtime):
    factory, task_id, case_id = runtime
    class Interrupted(DeterministicWorkflowProvider):
        def retrieve(self, query):
            raise KeyboardInterrupt("模拟进程中断")
    with pytest.raises(KeyboardInterrupt):
        SingleMachineWorker(factory, Interrupted(), worker_id="old").run_task(task_id)
    with factory() as session:
        task = session.scalar(select(WorkflowTask).where(WorkflowTask.task_id == task_id))
        task.lease_until = task.lease_until - timedelta(minutes=5)
        session.commit()
    result = SingleMachineWorker(factory, DeterministicWorkflowProvider(), worker_id="new").run_task(task_id)
    assert result.final_state == "CLOSED_SUCCESS"
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(StageArtifact).where(
            StageArtifact.case_id == case_id, StageArtifact.stage == "normalize")) == 1
        assert session.scalar(select(WorkflowTask.attempt).where(WorkflowTask.task_id == task_id)) == 1
