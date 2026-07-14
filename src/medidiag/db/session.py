"""数据库会话管理。

提供 engine 创建、表初始化和 session 工厂。
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from medidiag.db.models import Base


def create_db_engine(
    database_url: str = "sqlite:///./medidiag.db",
    echo: bool = False,
) -> Engine:
    """创建数据库 engine。

    Args:
        database_url: 数据库连接 URL，默认 SQLite。
        echo: 是否打印 SQL 日志（调试用）。
    """
    # SQLite 需要启用外键约束
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, echo=echo, connect_args=connect_args)
    if database_url.startswith("sqlite"):
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db(engine: Engine) -> None:
    """创建所有表（开发/测试用；生产用 Alembic 迁移）。"""
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """创建 session 工厂。"""
    return sessionmaker(bind=engine, expire_on_commit=False)
