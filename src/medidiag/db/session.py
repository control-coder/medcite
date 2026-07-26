"""数据库会话管理。

提供 engine 创建、表初始化和 session 工厂。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from sqlalchemy import CursorResult, Engine, create_engine, event
from sqlalchemy.engine import Result
from sqlalchemy.orm import Session, sessionmaker

from medidiag.db.models import Base

#: SQLite 写锁等待时长。`medidiag demo` 在同一进程内让 uvicorn 请求线程与
#: worker 线程并发访问同一个文件库；默认 busy_timeout 为 0，任何写冲突都会
#: 立刻抛 `database is locked`，而不是短暂等待后成功。
SQLITE_BUSY_TIMEOUT_MS = 5000


def create_db_engine(
    database_url: str = "sqlite:///./medidiag.db",
    echo: bool = False,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> Engine:
    """创建数据库 engine。

    Args:
        database_url: 数据库连接 URL，默认 SQLite。
        echo: 是否打印 SQL 日志（调试用）。
        busy_timeout_ms: SQLite 写锁等待毫秒数，仅对 SQLite 生效。
    """
    # SQLite 需要启用外键约束
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, echo=echo, connect_args=connect_args)
    if database_url.startswith("sqlite"):
        event.listen(
            engine,
            "connect",
            _sqlite_pragmas(busy_timeout_ms),
        )
    return engine


def _sqlite_pragmas(busy_timeout_ms: int) -> Callable[[Any, Any], None]:
    """返回 SQLite 连接级 PRAGMA 设置回调。

    - ``foreign_keys=ON``：SQLite 默认不强制外键。
    - ``journal_mode=WAL``：让读不阻塞写、写不阻塞读。`check_same_thread=False`
      已经允许跨线程共享连接，但没有 WAL 时 uvicorn 与 worker 线程仍会互相
      阻塞。内存库不支持 WAL，PRAGMA 会返回 ``memory`` 而不报错。
    - ``busy_timeout``：写锁被占用时等待而不是立即抛 ``database is locked``。
    """

    def _apply(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        finally:
            cursor.close()

    return _apply


def init_db(engine: Engine) -> None:
    """创建所有表（开发/测试用；生产用 Alembic 迁移）。"""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """创建 session 工厂。"""
    return sessionmaker(bind=engine, expire_on_commit=False)


def rowcount(result: Result[Any]) -> int:
    """返回一条 DML 语句影响的行数。

    ``Session.execute`` 的静态返回类型是 ``Result``，但 INSERT/UPDATE/DELETE
    实际返回带 ``rowcount`` 的 ``CursorResult``。全仓库的条件 UPDATE（租约
    fencing、乐观锁）都依赖这个值，因此把这次窄化集中在一处并加以说明，而不是
    在十几个调用点各写一次 cast 或忽略注释。
    """
    return cast("CursorResult[Any]", result).rowcount
