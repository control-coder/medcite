"""Alembic 迁移环境配置。

- 从环境变量 DATABASE_URL 读取数据库 URL（默认 SQLite）
- import medidiag.db.models 的 Base 和所有模型（autogenerate 支持）
- render_as_batch=True 支持 SQLite 的 ALTER 操作
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from alembic import context

# 添加 src 到 sys.path，让 medidiag 包可 import
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# import Base 和所有 models（autogenerate 需要检测表结构）
from medidiag.db.models import Base  # noqa: E402
from medidiag.db.models import (  # noqa: E402, F401
    AgentRun,
    Case,
    CaseEventLog,
    Citation,
    Review,
    WorkflowTask,
)

# Alembic Config 对象
config = context.config

# 从环境变量覆盖 database_url（优先于 alembic.ini）
db_url = os.environ.get("DATABASE_URL", "sqlite:///./medidiag.db")
config.set_main_option("sqlalchemy.url", db_url)

# 配置 Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate 的目标 metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：仅生成 SQL，不连接数据库。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite ALTER 支持
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库执行迁移。"""
    url = config.get_main_option("sqlalchemy.url")
    connect_args = {}
    if url and url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite ALTER 支持
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
