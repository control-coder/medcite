"""Redis/Celery 仅派发 task_id，任务状态仍以数据库为准。"""

from __future__ import annotations

import uuid

from celery import Celery

from medidiag.config import get_settings
from medidiag.db.session import create_db_engine, get_session_factory
from medidiag.workflow.provider import DeterministicWorkflowProvider
from medidiag.workflow.worker import SingleMachineWorker

celery_app = Celery("medidiag", broker=get_settings().medidiag_broker_url)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"], task_ignore_result=True,
    task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True, broker_connection_timeout=3,
    broker_transport_options={"socket_connect_timeout": 3, "socket_timeout": 3},
    task_default_queue="medidiag",
)


@celery_app.task(name="medidiag.execute", ignore_result=True)
def execute_task(task_id: str) -> None:
    # 默认明确使用离线 Provider，队列启动不会隐式调用付费模型。
    engine = create_db_engine(get_settings().database_url)
    try:
        SingleMachineWorker(
            get_session_factory(engine), DeterministicWorkflowProvider(),
            worker_id=f"celery-{uuid.uuid4().hex}",
        ).run_task(task_id)
    finally:
        engine.dispose()


def publish(task_id: str) -> None:
    celery_app.send_task("medidiag.execute", args=[task_id], retry=False)
