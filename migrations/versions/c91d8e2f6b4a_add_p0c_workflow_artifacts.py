"""增加 P0-C 工作流产物和报告

迁移版本：c91d8e2f6b4a
前置版本：b72f0f4c1a8e
创建时间：2026-07-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c91d8e2f6b4a"
down_revision: str | Sequence[str] | None = "b72f0f4c1a8e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("cases", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "input_kind",
                sa.String(32),
                nullable=False,
                server_default="deidentified_simulation",
            )
        )
        batch_op.add_column(sa.Column("source_ref", sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("trace_id", sa.String(64), nullable=True))
        batch_op.add_column(
            sa.Column("review_round", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.create_index("ix_cases_trace_id", ["trace_id"], unique=True)

    # 保留历史 revision；按方言生成不含病例内容的唯一 trace 标识。
    expression = (
        "md5(random()::text || clock_timestamp()::text)"
        if op.get_context().dialect.name == "postgresql"
        else "lower(hex(randomblob(16)))"
    )
    op.execute(f"UPDATE cases SET trace_id = 'trace_' || {expression} WHERE trace_id IS NULL")
    with op.batch_alter_table("cases", schema=None) as batch_op:
        batch_op.alter_column("trace_id", existing_type=sa.String(64), nullable=False)

    op.create_table(
        "stage_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("artifact_id", sa.String(64), nullable=False),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("output_hash", sa.String(64), nullable=False),
        sa.Column("component_version", sa.String(128), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.case_id"]),
        sa.ForeignKeyConstraint(["task_id"], ["workflow_tasks.task_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "case_id", "task_id", "stage", "attempt",
            name="uq_stage_artifacts_execution_stage",
        ),
    )
    op.create_index("ix_stage_artifacts_artifact_id", "stage_artifacts", ["artifact_id"], unique=True)
    op.create_index("ix_stage_artifacts_case_id", "stage_artifacts", ["case_id"])
    op.create_index("ix_stage_artifacts_task_id", "stage_artifacts", ["task_id"])
    op.create_index("ix_stage_artifacts_stage", "stage_artifacts", ["stage"])

    op.create_table(
        "case_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("report_id", sa.String(64), nullable=False),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("structured_report", sa.JSON(), nullable=False),
        sa.Column("risk_warnings", sa.JSON(), nullable=False),
        sa.Column("compliance_status", sa.String(32), nullable=False),
        sa.Column("generation_version", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.case_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("case_id", "version", name="uq_case_reports_version"),
    )
    op.create_index("ix_case_reports_report_id", "case_reports", ["report_id"], unique=True)
    op.create_index("ix_case_reports_case_id", "case_reports", ["case_id"])


def downgrade() -> None:
    op.drop_index("ix_case_reports_case_id", table_name="case_reports")
    op.drop_index("ix_case_reports_report_id", table_name="case_reports")
    op.drop_table("case_reports")

    op.drop_index("ix_stage_artifacts_stage", table_name="stage_artifacts")
    op.drop_index("ix_stage_artifacts_task_id", table_name="stage_artifacts")
    op.drop_index("ix_stage_artifacts_case_id", table_name="stage_artifacts")
    op.drop_index("ix_stage_artifacts_artifact_id", table_name="stage_artifacts")
    op.drop_table("stage_artifacts")

    with op.batch_alter_table("cases", schema=None) as batch_op:
        batch_op.drop_index("ix_cases_trace_id")
        batch_op.drop_column("review_round")
        batch_op.drop_column("trace_id")
        batch_op.drop_column("source_ref")
        batch_op.drop_column("input_kind")
