"""PostgreSQL + pgvector 向量存储：表由 Alembic 迁移创建，本类只读写，不建表。

查询全部使用参数化 SQL；元数据过滤形如 ``metadata ->> %s = ANY(%s)``，键和值都作为参数传入。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from medidiag.rag.store.base import ACTIVE, RETIRED, StoredChunk, StoreHit, filter_values


def _to_psycopg_dsn(dsn: str) -> str:
    """SQLAlchemy 风格的 ``postgresql+psycopg://`` 转成 psycopg 可直接使用的形式。"""
    return dsn.replace("postgresql+psycopg://", "postgresql://", 1)


class PgVectorStore:
    def __init__(self, dsn: str, *, ef_search: int = 100) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(_to_psycopg_dsn(dsn), autocommit=True)
        register_vector(self._conn)
        self._ef_search = int(ef_search)

    def close(self) -> None:
        self._conn.close()

    def documents(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT doc_id, content_hash FROM kb_documents WHERE status = %s", (ACTIVE,)).fetchall()
        return {doc_id: content_hash for doc_id, content_hash in rows}

    def corpus_version(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(version), 0) FROM kb_documents").fetchone()
        return int(row[0]) if row else 0

    def upsert_document(self, doc_id: str, content_hash: str, chunks: Sequence[StoredChunk],
                        version: int) -> None:
        with self._conn.transaction():
            self._conn.execute(
                "INSERT INTO kb_documents (doc_id, content_hash, status, version) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
                "status = EXCLUDED.status, version = EXCLUDED.version, updated_at = now()",
                (doc_id, content_hash, ACTIVE, version))
            self._conn.execute(
                "UPDATE kb_chunks SET status = %s WHERE doc_id = %s AND status = %s", (RETIRED, doc_id, ACTIVE))
            with self._conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO kb_chunks (doc_id, chunk_id, text, chunk_hash, strategy, metadata, "
                    "embedding, status, version) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)",
                    [(doc_id, c.chunk_id, c.text, c.chunk_hash, c.strategy,
                      json.dumps(c.metadata, ensure_ascii=False), np.asarray(c.vector, dtype=np.float32),
                      ACTIVE, version) for c in chunks])

    def retire_document(self, doc_id: str, version: int) -> None:
        with self._conn.transaction():
            self._conn.execute(
                "UPDATE kb_documents SET status = %s, version = %s, updated_at = now() WHERE doc_id = %s",
                (RETIRED, version, doc_id))
            self._conn.execute(
                "UPDATE kb_chunks SET status = %s WHERE doc_id = %s AND status = %s", (RETIRED, doc_id, ACTIVE))

    def count(self, status: str = ACTIVE) -> int:
        row = self._conn.execute("SELECT count(*) FROM kb_chunks WHERE status = %s", (status,)).fetchone()
        return int(row[0]) if row else 0

    def chunk_ids(self) -> set[str]:
        rows = self._conn.execute("SELECT chunk_id FROM kb_chunks WHERE status = %s", (ACTIVE,)).fetchall()
        return {r[0] for r in rows}

    def vectors(self, chunk_ids: Sequence[str]) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT chunk_id, embedding FROM kb_chunks WHERE status = %s AND chunk_id = ANY(%s)",
            (ACTIVE, list(chunk_ids))).fetchall()
        return {cid: np.asarray(vec, dtype=np.float32) for cid, vec in rows}

    def find_by_hash(self, chunk_hash: str) -> list[StoredChunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, text, chunk_hash, strategy, metadata, status, version "
            "FROM kb_chunks WHERE chunk_hash = %s ORDER BY id", (chunk_hash,)).fetchall()
        return [StoredChunk(chunk_id=r[0], doc_id=r[1], text=r[2], chunk_hash=r[3], strategy=r[4],
                            metadata=r[5], status=r[6], version=r[7]) for r in rows]

    def search(self, query_vec: Any, k: int, filters: Mapping[str, Any] | None = None) -> list[StoreHit]:
        sql = "SELECT chunk_id, 1 - (embedding <=> %s) AS score FROM kb_chunks WHERE status = %s"
        params: list[Any] = [np.asarray(query_vec, dtype=np.float32), ACTIVE]
        for key, wanted in (filters or {}).items():
            sql += " AND metadata ->> %s = ANY(%s)"
            params += [key, filter_values(wanted)]
        sql += " ORDER BY embedding <=> %s, chunk_id LIMIT %s"
        params += [np.asarray(query_vec, dtype=np.float32), int(k)]
        with self._conn.transaction():
            # 过滤条件与 retired 行会让 HNSW 近似扫描少返回结果；放宽扫描范围并开启迭代扫描
            self._conn.execute(f"SET LOCAL hnsw.ef_search = {self._ef_search}")
            self._conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = self._conn.execute(sql, params).fetchall()
        return [StoreHit(cid, float(score)) for cid, score in rows]
