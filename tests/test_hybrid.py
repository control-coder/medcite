"""M8 混合检索与 jieba 分词的离线测试：不联网、不下载模型、不读 .cache 语料。"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from medidiag.rag.dense import DenseRetriever
from medidiag.rag.hybrid import HybridRetriever, mmr_select
from medidiag.rag.lexicon import build_lexicon, extract_terms
from medidiag.rag.retrieval import Bm25Retriever, tokenize_text
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.application import load_application_config

ROOT = Path(__file__).resolve().parents[1]


def _chunk(chunk_id: str, text: str, **metadata: object) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=chunk_id, source="测试来源", source_id=chunk_id, text=text,
                          metadata=dict(metadata))


def _hybrid_chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return _chunk(chunk_id, text, doc_id=chunk_id, title=chunk_id)


# ---- jieba 分词与自定义词典 ----

def test_jieba_tokenize_is_deterministic_and_filters_blanks() -> None:
    text = "青少年精神卫生与心理健康"
    first = tokenize_text(text, "jieba")
    second = tokenize_text(text, "jieba")
    assert first == second  # 独立实例 + 关闭 HMM：同一输入切分结果一致
    assert all(token.strip() for token in first)
    assert all(token == token.lower() for token in first)


def test_existing_tokenizers_unchanged() -> None:
    assert tokenize_text("Hello World  fever", "whitespace") == ["hello", "world", "fever"]
    assert tokenize_text("发热", "cjk_bigram") == ["发热"]
    assert tokenize_text("发热咳嗽", "cjk_bigram") == ["发热", "热咳", "咳嗽"]


def test_jieba_custom_lexicon_keeps_terms_whole() -> None:
    chunks = [_chunk("c1", "正文", title="人工流产", section_path=["人工流产", "概述"])]
    lexicon = build_lexicon(chunks)
    assert "人工流产" in lexicon
    retriever = Bm25Retriever(chunks, bm25_tokenizer="jieba", lexicon=lexicon)
    assert "人工流产" in retriever.tokenize("什么是人工流产")
    default = Bm25Retriever(chunks, bm25_tokenizer="jieba")
    # 没有自定义词典时不保证整词；不验证具体切法，只验证两者实例互不影响
    assert default.tokenize("什么是人工流产")


def test_lexicon_rules_are_deterministic() -> None:
    assert extract_terms("流行性感冒（流感）") == ["流行性感冒"]
    assert extract_terms("高血压、糖尿病和心血管疾病") == ["高血压", "糖尿病", "心血管疾病"]
    assert extract_terms("a") == []  # 长度不足
    assert extract_terms("12345") == []  # 纯数字
    assert extract_terms("covid") == []  # 纯英文
    chunks = [_chunk("c1", "正文", title="流感", section_path=["流感", "重要事实"])]
    assert build_lexicon(chunks) == build_lexicon(chunks)  # 幂等


# ---- RRF 融合 ----

def _fixed_encoder(vectors: dict[str, list[float]]):
    def encode(texts: list[str]) -> np.ndarray:
        return np.array([vectors[text] for text in texts])
    return encode


def _hybrid() -> HybridRetriever:
    chunks = [
        _hybrid_chunk("c1", "甲"),
        _hybrid_chunk("c2", "乙"),
        _hybrid_chunk("c3", "丙"),
    ]
    bm25 = Bm25Retriever(chunks, bm25_tokenizer="whitespace")
    vectors = {"甲": [1.0, 0.0], "乙": [0.9, 0.1], "丙": [0.0, 1.0],
               "为这个句子生成表示以用于检索相关文章：甲": [1.0, 0.0]}
    dense = DenseRetriever(chunks, encoder=_fixed_encoder(vectors), model_name="m", model_revision="r")
    return HybridRetriever(chunks, bm25=bm25, dense=dense, candidates=3, rrf_k=60)


def test_rrf_fuses_both_lists_and_fills_scores() -> None:
    retriever = _hybrid()
    retriever.build_index()
    results = retriever.search("甲", top_k=3)
    by_id = {r.chunk_id: r for r in results}
    # c1 两路都命中（BM25 第 1，向量第 1），RRF 分数为两路之和
    assert by_id["c1"].score_sources == ("bm25", "dense")
    assert by_id["c1"].final_score == pytest.approx(1 / 61 + 1 / 61)
    assert by_id["c1"].bm25_score == pytest.approx(1.0)
    assert by_id["c1"].embedding_score == pytest.approx(1.0)
    # c2、c3 只被向量召回：BM25 分数为 0 且 score_sources 标出只有 dense
    assert by_id["c2"].score_sources == ("dense",)
    assert by_id["c2"].bm25_score == 0.0
    assert by_id["c2"].final_score == pytest.approx(1 / 62)
    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]


# ---- MMR 去冗余 ----

class _Item:
    def __init__(self, chunk_id: str, final_score: float) -> None:
        self.chunk_id = chunk_id
        self.final_score = final_score


def test_mmr_suppresses_near_duplicates() -> None:
    vectors = {"a": np.array([1.0, 0.0]), "b": np.array([0.99, 0.01]), "c": np.array([0.0, 1.0])}
    items = [_Item("a", 1.0), _Item("b", 0.95), _Item("c", 0.9)]
    selected = mmr_select(items, vector_for=vectors.get, top_k=2, mmr_lambda=0.7)
    # b 与 a 近重复被压，第二个入选的是主题不同的 c
    assert [item.chunk_id for item in selected] == ["a", "c"]


def test_mmr_tolerates_missing_vectors() -> None:
    items = [_Item("a", 1.0), _Item("b", 0.9)]
    selected = mmr_select(items, vector_for=lambda cid: None, top_k=2, mmr_lambda=0.7)
    assert [item.chunk_id for item in selected] == ["a", "b"]


def test_runtime_applies_mmr_on_rrf_results() -> None:
    chunks = [_hybrid_chunk("c1", "甲"), _hybrid_chunk("c2", "甲乙"), _hybrid_chunk("c3", "丙")]
    bm25 = Bm25Retriever(chunks, bm25_tokenizer="whitespace")
    vectors = {"甲": [1.0, 0.0], "甲乙": [0.99, 0.01], "丙": [0.0, 1.0],
               "为这个句子生成表示以用于检索相关文章：甲": [1.0, 0.0]}
    dense = DenseRetriever(chunks, encoder=_fixed_encoder(vectors), model_name="m", model_revision="r")
    retriever = HybridRetriever(chunks, bm25=bm25, dense=dense, candidates=3, rrf_k=60)
    backend = RuntimeMedicalRAG(
        chunks=chunks, corpus_version="v-test",
        retrieval_config={"method": "hybrid", "top_k": 1, "hybrid": {"mmr_lambda": 0.7}},
        retriever=retriever, corpus_path="corpus.jsonl",
        leakage_gate={"status": "passed", "rule_version": "test"},
    )
    retriever.build_index()
    # 无重排时 MMR 直接作用于 RRF 结果；top_k=1 只保留第一名，近重复的 c2 不进结果
    evidence = backend.search_evidence("甲")
    assert [item.chunk_id for item in evidence] == ["c1"]


def test_mmr_prefers_diverse_candidate_over_redundant_one() -> None:
    chunks = [_hybrid_chunk("c1", "甲"), _hybrid_chunk("c2", "甲乙"), _hybrid_chunk("c3", "丙")]
    bm25 = Bm25Retriever(chunks, bm25_tokenizer="whitespace")
    vectors = {"甲": [1.0, 0.0], "甲乙": [0.99, 0.01], "丙": [0.0, 1.0],
               "为这个句子生成表示以用于检索相关文章：甲": [1.0, 0.0]}
    dense = DenseRetriever(chunks, encoder=_fixed_encoder(vectors), model_name="m", model_revision="r")
    retriever = HybridRetriever(chunks, bm25=bm25, dense=dense, candidates=3, rrf_k=60)
    retriever.build_index()
    results = retriever.search("甲", top_k=3)
    assert [r.chunk_id for r in results] == ["c1", "c2", "c3"]  # RRF 名次
    picked = mmr_select(results, vector_for=retriever.vector_for, top_k=2, mmr_lambda=0.7)
    # c2 与 c1 近重复被压后，第二个入选的是主题不同的 c3
    assert [r.chunk_id for r in picked] == ["c1", "c3"]


# ---- 配置白名单 ----

def _hybrid_config() -> dict:
    return yaml.safe_load((ROOT / "configs/application_v4_hybrid.yaml").read_text(encoding="utf-8"))


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_existing_configs_still_accepted() -> None:
    for name in ("application.yaml", "application_agent.yaml", "application_rewrite.yaml",
                 "application_dense.yaml", "application_dense_agent.yaml",
                 "application_dense_agent_rerank.yaml", "application_dense_agent_rerank_multiturn.yaml"):
        config = load_application_config(ROOT / "configs" / name)
        assert config["retrieval"]["method"] in {"bm25", "dense"}


def test_hybrid_config_accepted_with_provenance_fields() -> None:
    config = load_application_config(ROOT / "configs/application_v4_hybrid.yaml")
    retrieval = config["retrieval"]
    assert retrieval["method"] == "hybrid"
    assert retrieval["hybrid"] == {"candidates": 20, "rrf_k": 60, "mmr_lambda": 0.7}


@pytest.mark.parametrize("change", [
    "missing_embedding", "floating_revision", "bad_hybrid_key", "hybrid_key_on_bm25",
    "hybrid_with_rewrite", "bad_tokenizer", "bad_mmr_lambda", "bad_rrf_k",
])
def test_unsafe_hybrid_config_rejected(tmp_path: Path, change: str) -> None:
    config = copy.deepcopy(_hybrid_config())
    if change == "missing_embedding":
        del config["embedding"]
    elif change == "floating_revision":
        config["embedding"]["revision"] = "main"
    elif change == "bad_hybrid_key":
        config["retrieval"]["hybrid"]["weights"] = [0.5, 0.5]
    elif change == "hybrid_key_on_bm25":
        config["retrieval"]["method"] = "bm25"
        del config["embedding"]
    elif change == "hybrid_with_rewrite":
        config["generation"]["rewrite"] = True
    elif change == "bad_tokenizer":
        config["retrieval"]["bm25_tokenizer"] = "whitespace"
    elif change == "bad_mmr_lambda":
        config["retrieval"]["hybrid"]["mmr_lambda"] = 1.5
    else:
        config["retrieval"]["hybrid"]["rrf_k"] = 0
    with pytest.raises(ValueError):
        load_application_config(_write(tmp_path, config))
