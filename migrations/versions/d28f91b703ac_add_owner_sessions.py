"""增加服务端会话与病例归属；历史病例不自动认领。"""

import sqlalchemy as sa
from alembic import op

revision = "d28f91b703ac"
down_revision = "c91d8e2f6b4a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("cases") as batch:
        batch.add_column(sa.Column("owner_id", sa.String(64), nullable=True))
        batch.create_index("ix_cases_owner_id", ["owner_id"])
    op.create_table("web_sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_web_sessions_owner_id", "web_sessions", ["owner_id"])


def downgrade() -> None:
    op.drop_index("ix_web_sessions_owner_id", table_name="web_sessions")
    op.drop_table("web_sessions")
    with op.batch_alter_table("cases") as batch:
        batch.drop_index("ix_cases_owner_id")
        batch.drop_column("owner_id")
