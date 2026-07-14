"""add P0-B concurrency constraints

Revision ID: b72f0f4c1a8e
Revises: a146135f3dc0
Create Date: 2026-07-14
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "b72f0f4c1a8e"
down_revision: Union[str, Sequence[str], None] = "a146135f3dc0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("cases", schema=None) as batch_op:
        batch_op.add_column(sa.Column("active_task_id", sa.String(64), nullable=True))
        batch_op.create_index("ix_cases_active_task_id", ["active_task_id"], unique=False)
        batch_op.create_unique_constraint(
            "uq_cases_scope_idempotency",
            ["idempotency_user_scope", "idempotency_key"],
        )

    with op.batch_alter_table("workflow_tasks", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_workflow_tasks_case_type_idempotency",
            ["case_id", "task_type", "idempotency_key"],
        )

    with op.batch_alter_table("agent_runs", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_agent_runs_execution_identity",
            ["case_id", "agent_name", "input_hash", "attempt_group"],
        )


def downgrade() -> None:
    with op.batch_alter_table("agent_runs", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_agent_runs_execution_identity", type_="unique"
        )

    with op.batch_alter_table("workflow_tasks", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_workflow_tasks_case_type_idempotency", type_="unique"
        )

    with op.batch_alter_table("cases", schema=None) as batch_op:
        batch_op.drop_constraint("uq_cases_scope_idempotency", type_="unique")
        batch_op.drop_index("ix_cases_active_task_id")
        batch_op.drop_column("active_task_id")
