"""创建 6 张数据表

迁移版本：a146135f3dc0
前置版本：
创建时间：2026-07-06 16:02:31.306131
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# Alembic 使用的迁移版本标识。
revision: str = 'a146135f3dc0'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """升级数据库结构。"""
    # ### 以下命令由 Alembic 自动生成，请按需调整。###
    op.create_table('cases',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('gold_answer', sa.String(length=64), nullable=True),
    sa.Column('normalized_query', sa.Text(), nullable=True),
    sa.Column('idempotency_key', sa.String(length=128), nullable=False),
    sa.Column('idempotency_user_scope', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('cases', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_cases_case_id'), ['case_id'], unique=True)
        batch_op.create_index(batch_op.f('ix_cases_idempotency_key'), ['idempotency_key'], unique=False)
        batch_op.create_index(batch_op.f('ix_cases_status'), ['status'], unique=False)

    op.create_table('agent_runs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('run_id', sa.String(length=64), nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('agent_name', sa.String(length=64), nullable=False),
    sa.Column('input_hash', sa.String(length=64), nullable=False),
    sa.Column('attempt_group', sa.String(length=64), nullable=False),
    sa.Column('input_payload', sa.JSON(), nullable=True),
    sa.Column('output_payload', sa.JSON(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['case_id'], ['cases.case_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_agent_runs_agent_name'), ['agent_name'], unique=False)
        batch_op.create_index(batch_op.f('ix_agent_runs_case_id'), ['case_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_agent_runs_run_id'), ['run_id'], unique=True)

    op.create_table('case_event_log',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('event_type', sa.String(length=32), nullable=False),
    sa.Column('from_status', sa.String(length=32), nullable=True),
    sa.Column('to_status', sa.String(length=32), nullable=True),
    sa.Column('trigger_subject', sa.String(length=32), nullable=False),
    sa.Column('trigger_entity', sa.String(length=128), nullable=True),
    sa.Column('detail', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['cases.case_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('case_event_log', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_case_event_log_case_id'), ['case_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_case_event_log_created_at'), ['created_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_case_event_log_event_type'), ['event_type'], unique=False)

    op.create_table('citations',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('claim_text', sa.Text(), nullable=False),
    sa.Column('chunk_id', sa.String(length=64), nullable=False),
    sa.Column('verdict', sa.String(length=16), nullable=False),
    sa.Column('verifier_model', sa.String(length=128), nullable=False),
    sa.Column('verifier_score', sa.Float(), nullable=True),
    sa.Column('human_reviewed', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['cases.case_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('citations', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_citations_case_id'), ['case_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_citations_chunk_id'), ['chunk_id'], unique=False)

    op.create_table('reviews',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('review_type', sa.String(length=32), nullable=False),
    sa.Column('reviewer', sa.String(length=128), nullable=False),
    sa.Column('result', sa.String(length=32), nullable=False),
    sa.Column('round', sa.Integer(), nullable=False),
    sa.Column('detail', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['cases.case_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_reviews_case_id'), ['case_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_reviews_review_type'), ['review_type'], unique=False)

    op.create_table('workflow_tasks',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('task_id', sa.String(length=64), nullable=False),
    sa.Column('case_id', sa.String(length=64), nullable=False),
    sa.Column('task_type', sa.String(length=32), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('lease_owner', sa.String(length=128), nullable=True),
    sa.Column('lease_until', sa.DateTime(), nullable=True),
    sa.Column('heartbeat_at', sa.DateTime(), nullable=True),
    sa.Column('attempt', sa.Integer(), nullable=False),
    sa.Column('input_hash', sa.String(length=64), nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=False),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.Column('error_code', sa.String(length=64), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['case_id'], ['cases.case_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('workflow_tasks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_workflow_tasks_case_id'), ['case_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_workflow_tasks_status'), ['status'], unique=False)
        batch_op.create_index(batch_op.f('ix_workflow_tasks_task_id'), ['task_id'], unique=True)
        batch_op.create_index(batch_op.f('ix_workflow_tasks_task_type'), ['task_type'], unique=False)

    # ### Alembic 自动生成命令结束。###


def downgrade() -> None:
    """降级数据库结构。"""
    # ### 以下命令由 Alembic 自动生成，请按需调整。###
    with op.batch_alter_table('workflow_tasks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_workflow_tasks_task_type'))
        batch_op.drop_index(batch_op.f('ix_workflow_tasks_task_id'))
        batch_op.drop_index(batch_op.f('ix_workflow_tasks_status'))
        batch_op.drop_index(batch_op.f('ix_workflow_tasks_case_id'))

    op.drop_table('workflow_tasks')
    with op.batch_alter_table('reviews', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_reviews_review_type'))
        batch_op.drop_index(batch_op.f('ix_reviews_case_id'))

    op.drop_table('reviews')
    with op.batch_alter_table('citations', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_citations_chunk_id'))
        batch_op.drop_index(batch_op.f('ix_citations_case_id'))

    op.drop_table('citations')
    with op.batch_alter_table('case_event_log', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_case_event_log_event_type'))
        batch_op.drop_index(batch_op.f('ix_case_event_log_created_at'))
        batch_op.drop_index(batch_op.f('ix_case_event_log_case_id'))

    op.drop_table('case_event_log')
    with op.batch_alter_table('agent_runs', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_agent_runs_run_id'))
        batch_op.drop_index(batch_op.f('ix_agent_runs_case_id'))
        batch_op.drop_index(batch_op.f('ix_agent_runs_agent_name'))

    op.drop_table('agent_runs')
    with op.batch_alter_table('cases', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_cases_status'))
        batch_op.drop_index(batch_op.f('ix_cases_idempotency_key'))
        batch_op.drop_index(batch_op.f('ix_cases_case_id'))

    op.drop_table('cases')
    # ### Alembic 自动生成命令结束。###
