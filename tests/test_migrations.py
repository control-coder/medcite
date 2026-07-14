"""Alembic migration round-trip tests on an isolated SQLite database."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def _config(project_root: Path) -> Config:
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "alembic"))
    return config


def test_p0b_migration_upgrade_downgrade_round_trip(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = Path(__file__).resolve().parent.parent
    database_path = tmp_path / "migration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    config = _config(project_root)

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    inspector = inspect(engine)
    assert "active_task_id" in {
        column["name"] for column in inspector.get_columns("cases")
    }
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
    engine.dispose()

    command.downgrade(config, "a146135f3dc0")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    inspector = inspect(engine)
    assert "active_task_id" not in {
        column["name"] for column in inspector.get_columns("cases")
    }
    engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    assert "active_task_id" in {
        column["name"] for column in inspect(engine).get_columns("cases")
    }
    engine.dispose()
