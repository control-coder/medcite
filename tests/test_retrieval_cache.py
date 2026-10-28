"""M14：检索结果缓存（命中/未命中、TTL、LRU、键区分、结果一致）。"""

from __future__ import annotations

import pytest

from medidiag.rag.cache import RetrievalCache, cache_key
from medidiag.rag.retrieval import Bm25Retriever
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.application import load_application_config


def _chunk(i: int, text: str, topic: str = "a") -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=f"c{i}", source="s", source_id=f"s{i}", text=text,
                          evidence_level="L1", metadata={"topic": topic})


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _rag(cache: dict | None = None, version: str = "v1", texts: list[str] | None = None) -> RuntimeMedicalRAG:
    texts = texts or ["高血压 患者 饮食 限盐", "糖尿病 血糖 监测", "高血压 药物 治疗 随访"]
    chunks = [_chunk(i, t) for i, t in enumerate(texts)]
    retrieval: dict = {"method": "bm25", "top_k": 2}
    if cache is not None:
        retrieval["cache"] = cache
    retriever = Bm25Retriever(chunks, bm25_tokenizer="cjk_bigram")
    retriever.build_index()
    return RuntimeMedicalRAG(chunks=chunks, corpus_version=version, retrieval_config=retrieval,
                             retriever=retriever, corpus_path="x", leakage_gate={"status": "passed"})


def test_hit_equals_miss_and_is_recorded() -> None:
    rag = _rag({"enabled": True})
    first = rag.retrieve("高血压 饮食")
    assert first["retrieval_cache"] == {"hit": False}
    second = rag.retrieve("高血压 饮食")
    base = dict(top_k=None, filters=None, corpus_hash="h", corpus_version="v", config_hash="c")
    assert cache_key("高血压  饮食 ", **base) == cache_key("高血压 饮食", **base)  # 空白规范化后同键
    assert second["retrieval_cache"] == {"hit": True}
    assert {k: v for k, v in first.items() if k != "retrieval_cache"} == \
        {k: v for k, v in second.items() if k != "retrieval_cache"}
    assert rag.cache is not None and rag.cache.stats()["hits"] == 1


def test_disabled_by_default_keeps_result_shape() -> None:
    rag = _rag()
    assert rag.cache is None
    assert "retrieval_cache" not in rag.retrieve("高血压 饮食")
    assert rag.last_cache_hit is None
    # 开关不改变配置哈希，证据包标识一致
    assert _rag({"enabled": True}).retrieval_config_hash == rag.retrieval_config_hash


def test_top_k_and_filters_distinguish_keys() -> None:
    rag = _rag({"enabled": True})
    rag.search_evidence("高血压", top_k=1)
    assert rag.last_cache_hit is False
    rag.search_evidence("高血压", top_k=3)
    assert rag.last_cache_hit is False
    rag.search_evidence("高血压", top_k=1)
    assert rag.last_cache_hit is True
    base = dict(top_k=None, corpus_hash="h", corpus_version="v", config_hash="c")
    assert cache_key("q", filters={"topic": "x"}, **base) != cache_key("q", filters=None, **base)
    assert cache_key("q", filters={"topic": "x"}, **base) != cache_key("q", filters={"topic": "y"}, **base)


def test_corpus_version_and_content_distinguish_keys() -> None:
    base = dict(top_k=None, filters=None, config_hash="c")
    assert cache_key("q", corpus_hash="h", corpus_version="v1", **base) != \
        cache_key("q", corpus_hash="h", corpus_version="v2", **base)
    assert cache_key("q", corpus_hash="h1", corpus_version="v", **base) != \
        cache_key("q", corpus_hash="h2", corpus_version="v", **base)
    assert _rag(version="v1").corpus_hash != _rag(texts=["别的 语料"]).corpus_hash


def test_ttl_expiry_and_lru_eviction() -> None:
    clock = _Clock()
    cache = RetrievalCache(max_entries=2, ttl_s=10, clock=clock)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == (True, 1)  # a 变为最近使用
    cache.put("c", 3)  # 淘汰最久未用的 b
    assert cache.get("b") == (False, None)
    assert cache.get("a")[0] and cache.get("c")[0]
    clock.now = 11
    assert cache.get("a") == (False, None)  # 过期并删除
    assert len(cache) == 1


def test_cached_value_is_isolated_from_caller_mutation() -> None:
    rag = _rag({"enabled": True})
    first = rag.search_evidence("高血压")
    first[0].metadata["topic"] = "被改动"
    assert rag.search_evidence("高血压")[0].metadata["topic"] == "a"


def test_application_config_whitelist(tmp_path) -> None:
    text = open("configs/application.yaml", encoding="utf-8").read()

    def load(extra: str):
        path = tmp_path / "app.yaml"
        path.write_text(text.replace("retrieval:\n", "retrieval:\n" + extra, 1), encoding="utf-8")
        return load_application_config(path)

    assert load("  cache: {enabled: true, max_entries: 64, ttl_s: 600}\n")["retrieval"]["cache"]["enabled"] is True
    for bad in ("  cache: {enabled: true, evil: 1}\n", "  cache: {enabled: yes_no}\n",
                "  cache: {max_entries: 0}\n", "  cache: {ttl_s: -1}\n"):
        with pytest.raises(ValueError):
            load(bad)
