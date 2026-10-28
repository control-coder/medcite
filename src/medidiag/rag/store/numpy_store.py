"""NumPy 内存向量存储：供 SQLite 演示、测试和启动缓存使用，可持久化到 ``.cache/index/``。

持久化为两个文件：``{base}.npy``（全部向量，含 retired）与 ``{base}.json``（块元数据、文档状态、版本）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from medidiag.rag.store.base import (
    ACTIVE,
    RETIRED,
    StoredChunk,
    StoreHit,
    filter_value,
    filter_values,
)


def corpus_hash(chunks: Sequence[Any]) -> str:
    """语料指纹：按 chunk_id 与文本计算，用于 ``{corpus_hash}_{model_revision}`` 缓存文件名。"""
    payload = json.dumps([[c.chunk_id, c.text] for c in chunks], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_base(index_dir: str | Path, chunks: Sequence[Any], model_revision: str) -> Path:
    return Path(index_dir) / f"{corpus_hash(chunks)}_{model_revision[:12]}"


class NumpyStore:
    def __init__(self) -> None:
        self._rows: list[StoredChunk] = []
        self._docs: dict[str, dict[str, Any]] = {}
        self._version = 0
        self._matrix: Any = None  # 行与 _rows 对齐的缓存矩阵，变更后失效

    # ---- 写入 ----
    def upsert_document(self, doc_id: str, content_hash: str, chunks: Sequence[StoredChunk],
                        version: int) -> None:
        self._retire_rows(doc_id)
        for chunk in chunks:
            self._rows.append(StoredChunk(
                chunk_id=chunk.chunk_id, doc_id=doc_id, text=chunk.text, chunk_hash=chunk.chunk_hash,
                strategy=chunk.strategy, metadata=dict(chunk.metadata),
                vector=np.asarray(chunk.vector, dtype=np.float32), status=ACTIVE, version=version))
        self._docs[doc_id] = {"content_hash": content_hash, "status": ACTIVE, "version": version}
        self._version = max(self._version, version)
        self._matrix = None

    def retire_document(self, doc_id: str, version: int) -> None:
        self._retire_rows(doc_id)
        if doc_id in self._docs:
            self._docs[doc_id].update(status=RETIRED, version=version)
        self._version = max(self._version, version)
        self._matrix = None

    def _retire_rows(self, doc_id: str) -> None:
        for row in self._rows:
            if row.doc_id == doc_id and row.status == ACTIVE:
                row.status = RETIRED

    # ---- 读取 ----
    def documents(self) -> dict[str, str]:
        return {d: v["content_hash"] for d, v in self._docs.items() if v["status"] == ACTIVE}

    def corpus_version(self) -> int:
        return self._version

    def count(self, status: str = ACTIVE) -> int:
        return sum(1 for row in self._rows if row.status == status)

    def chunk_ids(self) -> set[str]:
        return {row.chunk_id for row in self._rows if row.status == ACTIVE}

    def _active_index(self) -> dict[str, int]:
        return {row.chunk_id: i for i, row in enumerate(self._rows) if row.status == ACTIVE}

    def _stacked(self) -> Any:
        if self._matrix is None:
            dim = len(self._rows[0].vector) if self._rows else 0
            self._matrix = (np.stack([row.vector for row in self._rows]) if self._rows
                            else np.zeros((0, dim), dtype=np.float32))
        return self._matrix

    def vectors(self, chunk_ids: Sequence[str]) -> dict[str, Any]:
        index = self._active_index()
        matrix = self._stacked()
        return {cid: matrix[index[cid]] for cid in chunk_ids if cid in index}

    def find_by_hash(self, chunk_hash: str) -> list[StoredChunk]:
        return [StoredChunk(**{**asdict(row), "vector": None}) for row in self._rows
                if row.chunk_hash == chunk_hash]

    def search(self, query_vec: Any, k: int, filters: Mapping[str, Any] | None = None) -> list[StoreHit]:
        if not self._rows:
            return []
        query = np.asarray(query_vec, dtype=np.float64)
        scores = self._stacked().astype(np.float64) @ query
        candidates = [
            i for i, row in enumerate(self._rows)
            if row.status == ACTIVE and _match(row.metadata, filters)
        ]
        candidates.sort(key=lambda i: (-float(scores[i]), self._rows[i].chunk_id))
        return [StoreHit(self._rows[i].chunk_id, float(scores[i])) for i in candidates[:k]]

    # ---- 持久化 ----
    @staticmethod
    def exists(base: str | Path) -> bool:
        base = Path(base)
        return base.with_suffix(".npy").exists() and base.with_suffix(".json").exists()

    def save(self, base: str | Path) -> None:
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self._version,
            "docs": self._docs,
            "rows": [{**asdict(row), "vector": None} for row in self._rows],
        }
        np.save(base.with_suffix(".npy"), self._stacked())
        base.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, base: str | Path) -> NumpyStore:
        base = Path(base)
        meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        matrix = np.load(base.with_suffix(".npy"))
        store = cls()
        store._version = int(meta["version"])
        store._docs = meta["docs"]
        store._rows = [StoredChunk(**{**row, "vector": matrix[i]}) for i, row in enumerate(meta["rows"])]
        return store


def _match(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    for key, wanted in (filters or {}).items():
        if filter_value(metadata.get(key)) not in filter_values(wanted):
            return False
    return True
