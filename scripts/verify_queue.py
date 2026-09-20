"""在专用验收数据库验证真实 Redis/Celery 与进程中断；不接触用户数据库。"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy import func, select

from medidiag.config import get_settings
from medidiag.db.models import Case, CaseReport, StageArtifact, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory
from medidiag.workflow.dispatcher import scan_pending
from medidiag.workflow.executor import WorkflowExecutor


def main() -> None:
    if not os.environ.get("DATABASE_URL", "").endswith("/medidiag_acceptance"):
        raise SystemExit("必须显式指定专用 medidiag_acceptance 数据库。")
    os.environ["MEDIDIAG_LEASE_SECONDS"] = "3"
    os.environ["MEDIDIAG_HEARTBEAT_SECONDS"] = "1"
    get_settings.cache_clear()
    from medidiag.workflow.queue import publish

    engine = create_db_engine(get_settings().database_url)
    factory = get_session_factory(engine)
    executor = WorkflowExecutor()
    log_dir = Path(".cache/implementation")
    log_dir.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    def create():
        with factory() as session:
            case = executor.create_case(session, "公开模拟输入，用于真实队列工程验收。", uuid.uuid4().hex, "acceptance")
            task = executor.start_workflow(session, case.case_id, "case_workflow", "run", "hash")
            return case.case_id, task.task_id

    def wait(case_id):
        for _ in range(150):
            with factory() as session:
                state = session.scalar(select(Case.status).where(Case.case_id == case_id))
                if state == "CLOSED_SUCCESS":
                    assert session.scalar(select(func.count()).select_from(CaseReport).where(CaseReport.case_id == case_id)) == 1
                    return
                if state == "ESCALATED":
                    raise AssertionError("验收任务发生升级")
            time.sleep(0.2)
        raise AssertionError("等待队列任务完成超时")

    # 先真实中断独立 worker，使归一化已提交、检索尚未完成。
    interrupted_case, interrupted_task = create()
    child = """
import os
from medidiag.config import get_settings
from medidiag.db.session import create_db_engine, get_session_factory
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker
class Crash(DeterministicWorkflowProvider):
    def retrieve(self, query):
        os._exit(73)
SingleMachineWorker(get_session_factory(create_db_engine(get_settings().database_url)), Crash(), worker_id='crash-test').run_task(os.environ['MEDIDIAG_CRASH_TASK'])
"""
    env = dict(os.environ, MEDIDIAG_CRASH_TASK=interrupted_task)
    crashed = subprocess.run([sys.executable, "-c", child], env=env, creationflags=flags, timeout=20, capture_output=True)
    assert crashed.returncode == 73
    time.sleep(3.5)
    with (log_dir / "celery-acceptance.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen([sys.executable, "-m", "celery", "-A", "medidiag.workflow.queue:celery_app",
                                   "worker", "--pool=solo", "--concurrency=1", "--loglevel=WARNING"],
                                  stdout=log, stderr=log, creationflags=flags)
        try:
            case_id, task_id = create()
            publish(task_id)
            publish(task_id)
            wait(case_id)
            failed_case, _ = create()
            def fail(_):
                raise ConnectionError("模拟 broker 派发窗口故障")
            assert scan_pending(factory, fail)["failed"] >= 1
            assert scan_pending(factory, publish)["sent"] >= 1
            wait(failed_case)
            wait(interrupted_case)
            with factory() as session:
                assert session.scalar(select(func.count()).select_from(StageArtifact).where(
                    StageArtifact.case_id == interrupted_case, StageArtifact.stage == "normalize")) == 1
                assert session.scalar(select(WorkflowTask.attempt).where(WorkflowTask.task_id == interrupted_task)) == 1
            print("真实 PostgreSQL/Redis/Celery：重复投递、派发失败补偿、独立进程中断恢复均通过。")
        finally:
            worker.terminate()
            worker.wait(timeout=15)
            engine.dispose()


if __name__ == "__main__":
    main()
