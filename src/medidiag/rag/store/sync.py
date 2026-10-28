"""增量同步：按文档 content_hash 判断新增、变更、删除，只为新增和变更的文档重新编码。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from medidiag.errors import MediDiagError
from medidiag.rag.store.base import StoredChunk, VectorStore
from medidiag.schemas import KnowledgeChunk


@dataclass
class SyncReport:
    added: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: int
    encoded_chunks: int
    corpus_version: int


def _doc_id(chunk: KnowledgeChunk) -> str:
    return str(chunk.metadata.get("doc_id") or chunk.source_id)


def _chunk_hash(chunk: KnowledgeChunk) -> str:
    return str(chunk.metadata.get("chunk_hash") or hashlib.sha256(chunk.text.encode("utf-8")).hexdigest())


def group_by_document(chunks: Sequence[KnowledgeChunk]) -> dict[str, tuple[str, list[KnowledgeChunk]]]:
    """按文档分组，返回 doc_id → (content_hash, 块列表)；缺 content_hash 时用各块哈希拼接代替。"""
    groups: dict[str, list[KnowledgeChunk]] = defaultdict(list)
    for chunk in chunks:
        groups[_doc_id(chunk)].append(chunk)
    result = {}
    for doc_id, items in groups.items():
        content_hash = str(items[0].metadata.get("content_hash") or "") or hashlib.sha256(
            "".join(_chunk_hash(c) for c in items).encode("utf-8")).hexdigest()
        result[doc_id] = (content_hash, items)
    return result


def sync_corpus(store: VectorStore, chunks: Sequence[KnowledgeChunk], encoder: Callable[[list[str]], Any],
                *, retire_missing: bool = True) -> SyncReport:
    """把目标语料同步进存储；没有任何变化时不编码、不升 corpus_version。"""
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise MediDiagError("RAG_CORPUS_INVALID", detail="同步语料中 chunk_id 重复")
    target = group_by_document(chunks)
    current = store.documents()
    added = sorted(d for d in target if d not in current)
    changed = sorted(d for d in target if d in current and current[d] != target[d][0])
    removed = sorted(d for d in current if d not in target) if retire_missing else []
    todo = added + changed
    version = store.corpus_version()
    if todo or removed:
        version += 1
    texts = [c.text for d in todo for c in target[d][1]]
    vectors = np.asarray(encoder(texts), dtype=np.float64) if texts else np.zeros((0, 0))
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12) if texts else vectors
    offset = 0
    for doc_id in todo:
        content_hash, items = target[doc_id]
        stored = []
        for chunk in items:
            stored.append(StoredChunk(
                chunk_id=chunk.chunk_id, doc_id=doc_id, text=chunk.text, chunk_hash=_chunk_hash(chunk),
                strategy=str(chunk.metadata.get("strategy", "")), metadata=dict(chunk.metadata),
                vector=vectors[offset].astype(np.float32)))
            offset += 1
        store.upsert_document(doc_id, content_hash, stored, version)
    for doc_id in removed:
        store.retire_document(doc_id, version)
    return SyncReport(added=added, changed=changed, removed=removed,
                      unchanged=len(target) - len(todo), encoded_chunks=len(texts), corpus_version=version)
