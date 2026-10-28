"""新增知识库向量表 kb_documents、kb_chunks（M9）。仅 PostgreSQL 执行：启用 pgvector 扩展并建 HNSW 索引；SQLite 跳过。"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f5b8d3a1c6e2"
down_revision = "e4a7c2d9f310"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    from pgvector.sqlalchemy import Vector

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "kb_documents",
        sa.Column("doc_id", sa.Text, primary_key=True),
        sa.Column("content_hash", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "kb_chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.Text, sa.ForeignKey("kb_documents.doc_id"), nullable=False),
        sa.Column("chunk_id", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("chunk_hash", sa.Text, nullable=False),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False),
    )
    op.create_index("ix_kb_chunks_chunk_hash", "kb_chunks", ["chunk_hash"])
    op.create_index("ix_kb_chunks_doc_status", "kb_chunks", ["doc_id", "status"])
    op.create_index("ix_kb_chunks_chunk_id", "kb_chunks", ["chunk_id"])
    op.execute(
        "CREATE INDEX ix_kb_chunks_embedding_hnsw ON kb_chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    # 扩展可能被其他对象使用，降级只删表，不删扩展
    op.drop_table("kb_chunks")
    op.drop_table("kb_documents")
