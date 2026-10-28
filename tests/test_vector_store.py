"""M9 向量存储与增量同步：NumpyStore 始终运行，PgVectorStore 仅在提供 MEDIDIAG_TEST_PG_DSN 时运行。"""

from __future__ import annotations

import copy
import hashlib
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml

from medidiag.rag.dense import DenseRetriever
from medidiag.rag.store import NumpyStore, cache_base, sync_corpus
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.application import load_application_config

ROOT = Path(__file__).resolve().parents[1]
PG_DSN = os.environ.get("MEDIDIAG_TEST_PG_DSN")
WORDS = ["发热", "咳嗽", "疫苗", "糖尿病", "血压", "饮食", "运动", "吸烟", "传播", "预防", "症状", "治疗",
         "儿童", "老年", "筛查", "感染", "营养", "睡眠", "肥胖", "癌症"]
DIM = 512
pg_only = pytest.mark.skipif(not PG_DSN, reason="未设置 MEDIDIAG_TEST_PG_DSN")


def encode(texts: list[str]) -> np.ndarray:
    """确定性的字符二元组哈希向量，只用于测试。"""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for i in range(len(text) - 1):
            out[row, int(hashlib.md5(text[i:i + 2].encode()).hexdigest(), 16) % DIM] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def make_doc(doc_id: str, seed: int, version: str = "v1", topic: str = "a") -> list[KnowledgeChunk]:
    rng = np.random.default_rng(seed)
    chunks = []
    for n in range(3):
        text = "，".join(str(w) for w in rng.choice(WORDS, 5, replace=False)) + f"。{doc_id}第{n}段{version}"
        chunks.append(KnowledgeChunk(
            chunk_id=f"{doc_id}:0:p:{n}", source=doc_id, source_id=doc_id, text=text,
            metadata={"doc_id": doc_id, "content_hash": f"{doc_id}-{version}", "strategy": "parent_child",
                      "topic": topic, "chunk_hash": hashlib.sha256(text.encode()).hexdigest()}))
    return chunks


def corpus(overrides: dict[str, tuple[int, str]] | None = None, skip: set[str] = frozenset(),  # type: ignore[assignment]
           extra: tuple[str, ...] = ()) -> list[KnowledgeChunk]:
    overrides = overrides or {}
    chunks: list[KnowledgeChunk] = []
    for i in range(8):
        doc_id = f"doc{i}"
        if doc_id in skip:
            continue
        seed, version = overrides.get(doc_id, (i, "v1"))
        chunks += make_doc(doc_id, seed, version, topic="a" if i % 2 == 0 else "b")
    for doc_id in extra:
        chunks += make_doc(doc_id, 99)
    return chunks


QUERIES = [" ".join(WORDS[i:i + 3]) for i in range(0, 15, 3)]


def _top(store, query: str, k: int = 5, filters=None) -> list[str]:
    return [h.chunk_id for h in store.search(encode([query])[0], k, filters)]


def _pg_clean():
    from medidiag.rag.store.pg_store import PgVectorStore

    store = PgVectorStore(PG_DSN)  # type: ignore[arg-type]
    store._conn.execute("TRUNCATE kb_chunks, kb_documents")
    return store


@pytest.fixture(scope="module")
def pg_ready():
    """对测试库执行 Alembic 升级到最新版本。"""
    from alembic import command
    from alembic.config import Config

    os.environ["DATABASE_URL"] = PG_DSN or ""
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")


def _factories() -> list:
    return [
        pytest.param(lambda: NumpyStore(), id="numpy"),
        pytest.param(_pg_clean, id="pgvector", marks=[pytest.mark.pgvector, pg_only]),
    ]


@pytest.mark.parametrize("make_store", _factories())
def test_incremental_sync_matches_full_rebuild(make_store: Callable, request) -> None:
    if PG_DSN:
        request.getfixturevalue("pg_ready")
    store = make_store()
    first = sync_corpus(store, corpus(), encode)
    assert (len(first.added), first.corpus_version, first.encoded_chunks) == (8, 1, 24)
    old_hash = corpus()[3 * 2].metadata["chunk_hash"]  # doc2 的第一块

    # 改 doc2、删 doc5、加 doc8
    updated = corpus({"doc2": (50, "v2")}, skip={"doc5"}, extra=("doc8",))
    report = sync_corpus(store, updated, encode)
    assert (report.changed, report.removed, report.added) == (["doc2"], ["doc5"], ["doc8"])
    assert report.encoded_chunks == 6 and report.unchanged == 6 and report.corpus_version == 2
    assert sync_corpus(store, updated, encode).encoded_chunks == 0  # 无变化不编码、不升版本
    assert store.corpus_version() == 2
    assert store.count() == 3 * 8 and store.count("retired") == 6
    # 旧块仍可按 chunk_hash 回查
    history = store.find_by_hash(old_hash)
    assert history and history[0].status == "retired" and history[0].doc_id == "doc2"

    incremental = {q: _top(store, q) for q in QUERIES}
    incremental_filtered = _top(store, QUERIES[0], filters={"topic": "a"})
    assert all(cid.startswith(("doc0", "doc2", "doc4", "doc6", "doc8")) for cid in incremental_filtered)

    rebuilt = make_store()
    sync_corpus(rebuilt, updated, encode)
    assert {q: _top(rebuilt, q) for q in QUERIES} == incremental
    assert _top(rebuilt, QUERIES[0], filters={"topic": "a"}) == incremental_filtered


def test_numpy_store_persistence_round_trip(tmp_path: Path) -> None:
    store = NumpyStore()
    sync_corpus(store, corpus(), encode)
    sync_corpus(store, corpus({"doc1": (60, "v2")}), encode)
    store.save(tmp_path / "idx")
    loaded = NumpyStore.load(tmp_path / "idx")
    assert loaded.corpus_version() == 2 and loaded.count() == 24 and loaded.count("retired") == 3
    assert _top(loaded, QUERIES[1]) == _top(store, QUERIES[1])


def test_dense_retriever_numpy_cache_skips_encoding(tmp_path: Path) -> None:
    chunks = corpus()
    calls: list[int] = []

    def counting(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return encode(texts)

    base = cache_base(tmp_path, chunks, "a" * 40)
    cold = DenseRetriever(chunks, encoder=counting, model_name="m", model_revision="a" * 40, store_base=base)
    cold.build_index()
    assert not cold.cache_hit and calls == [24]
    warm = DenseRetriever(chunks, encoder=counting, model_name="m", model_revision="a" * 40, store_base=base)
    warm.build_index()
    assert warm.cache_hit and calls == [24]  # 第二次启动没有再编码语料
    legacy = DenseRetriever(chunks, encoder=encode, model_name="m", model_revision="a" * 40)
    legacy.build_index()
    assert [r.chunk_id for r in warm.search(QUERIES[0], 5)] == [r.chunk_id for r in legacy.search(QUERIES[0], 5)]
    assert warm.vector_for(chunks[0].chunk_id) is not None


def test_vector_store_config_whitelist(tmp_path: Path) -> None:
    base = yaml.safe_load((ROOT / "configs" / "application_v4_hybrid.yaml").read_text(encoding="utf-8"))

    def load(mutate: Callable[[dict], None]) -> dict:
        config = copy.deepcopy(base)
        mutate(config)
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        return load_application_config(path)

    assert load(lambda c: c["retrieval"].update(vector_store="numpy"))["retrieval"]["vector_store"] == "numpy"
    assert load(lambda c: c["retrieval"].update(vector_store="pgvector"))
    with pytest.raises(ValueError):
        load(lambda c: c["retrieval"].update(vector_store="faiss"))
    with pytest.raises(ValueError):  # BM25 单路没有向量，不接受该键
        def bm25(c: dict) -> None:
            c.pop("embedding")
            c["retrieval"] = {"method": "bm25", "top_k": 3, "vector_store": "numpy"}
        load(bm25)


def test_kb_migration_skips_sqlite(tmp_path: Path, monkeypatch) -> None:
    from alembic import command
    from alembic.config import Config

    db = tmp_path / "m.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "cases" in names and "kb_chunks" not in names


@pg_only
@pytest.mark.pgvector
def test_kb_migration_round_trip_on_postgres() -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    os.environ["DATABASE_URL"] = PG_DSN or ""
    cfg = Config(str(ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")
    engine = create_engine(PG_DSN or "")
    assert {"kb_documents", "kb_chunks"} <= set(inspect(engine).get_table_names())
    command.downgrade(cfg, "-1")
    assert "kb_chunks" not in inspect(engine).get_table_names()
    command.upgrade(cfg, "head")
    engine.dispose()
