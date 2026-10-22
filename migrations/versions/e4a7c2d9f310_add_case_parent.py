"""病例增加 parent_case_id，支持同一归属下的多轮追问；历史病例为空，行为不变。"""

import sqlalchemy as sa
from alembic import op

revision = "e4a7c2d9f310"
down_revision = "d28f91b703ac"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("cases") as batch:
        batch.add_column(sa.Column("parent_case_id", sa.String(64), nullable=True))
        batch.create_index("ix_cases_parent_case_id", ["parent_case_id"])


def downgrade() -> None:
    with op.batch_alter_table("cases") as batch:
        batch.drop_index("ix_cases_parent_case_id")
        batch.drop_column("parent_case_id")
