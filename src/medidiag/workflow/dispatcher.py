"""数据库待派发扫描与中断补偿，不维护第二套 outbox。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, sessionmaker

from medidiag.config import get_settings
from medidiag.db.models import Case, WorkflowTask
from medidiag.db.session import create_db_engine, get_session_factory
from medidiag.observability.logging import get_logger
from medidiag.workflow.state_machine import TERMINAL_STATES

_log = get_logger(__name__)


def scan_pending(factory: sessionmaker[Session], publish: Callable[[str], None]) -> dict[str, int]:
    now = datetime.now(UTC).replace(tzinfo=None)
    with factory() as session:
        ids = list(session.scalars(select(WorkflowTask.task_id).join(
            Case, Case.active_task_id == WorkflowTask.task_id,
        ).where(
            Case.status.not_in([s.value for s in TERMINAL_STATES] + ["ESCALATED"]),
            or_(WorkflowTask.status == "PENDING", and_(WorkflowTask.status == "RUNNING",
                                                       WorkflowTask.lease_until <= now)),
        ).order_by(WorkflowTask.updated_at, WorkflowTask.id).limit(100)))
    sent, failed = 0, 0
    for task_id in ids:
        try:
            publish(task_id)
            sent += 1
        except Exception:
            # 不输出可能含 broker 凭据的异常；数据库任务保持可扫描，下一轮补偿。
            failed += 1
            _log.warning("dispatch.failed", task_id=task_id)
    return {"sent": sent, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描数据库任务并向队列派发标识")
    parser.add_argument("--once", action="store_true", help="只扫描一次")
    args = parser.parse_args()
    from medidiag.workflow.queue import publish

    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    try:
        while True:
            result = scan_pending(get_session_factory(engine), publish)
            _log.info("dispatch.scanned", **result)
            if args.once:
                break
            time.sleep(max(1, settings.medidiag_dispatch_seconds))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
