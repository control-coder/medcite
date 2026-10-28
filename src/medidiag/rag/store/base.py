"""向量存储的最小协议与共用数据结构。

一个文档（doc_id）对应一组块；文档内容哈希（content_hash）变化时整篇换新：旧块标 ``retired``
但不删除，历史证据包仍可按 chunk_hash 回查。``corpus_version`` 为整数，每次有变更的同步加 1。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

ACTIVE = "active"
RETIRED = "retired"
EMBEDDING_DIM = 512


@dataclass
class StoredChunk:
    """存储中的一个检索块；vector 为已归一化的向量，历史回查时可为 None。"""

    chunk_id: str
    doc_id: str
    text: str
    chunk_hash: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Any = None
    status: str = ACTIVE
    version: int = 0


@dataclass(frozen=True)
class StoreHit:
    chunk_id: str
    score: float


class VectorStore(Protocol):
    """向量存储接口；search 只返回 active 块，同分按 chunk_id 稳定排序。"""

    def documents(self) -> dict[str, str]:
        """有效文档的 doc_id → content_hash。"""

    def corpus_version(self) -> int: ...

    def upsert_document(self, doc_id: str, content_hash: str, chunks: Sequence[StoredChunk],
                        version: int) -> None:
        """新增或替换一篇文档：该文档原有 active 块标 retired，再写入新块。"""

    def retire_document(self, doc_id: str, version: int) -> None: ...

    def search(self, query_vec: Any, k: int,
               filters: Mapping[str, Any] | None = None) -> list[StoreHit]: ...

    def count(self, status: str = ACTIVE) -> int: ...

    def chunk_ids(self) -> set[str]:
        """active 块的 chunk_id 集合。"""

    def vectors(self, chunk_ids: Sequence[str]) -> dict[str, Any]:
        """按 chunk_id 取 active 块的向量；缺失的不返回。"""

    def find_by_hash(self, chunk_hash: str) -> list[StoredChunk]:
        """按 chunk_hash 回查（含 retired），不带向量。"""


def filter_value(value: Any) -> str | None:
    """元数据值转成可比较的文本；与 PostgreSQL 的 ``metadata ->> key`` 口径一致。"""
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def filter_values(wanted: Any) -> list[str]:
    items = wanted if isinstance(wanted, (list, tuple, set, frozenset)) else [wanted]
    return [text for text in (filter_value(item) for item in items) if text is not None]
