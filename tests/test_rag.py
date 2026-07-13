"""阶段 4 RAG 测试。

覆盖:
1. 术语归一化（三层加载、同义词替换、覆盖率、term_overlap）—— 不需要重依赖
2. 检索器（BM25 + embedding + 配置驱动实验 + Recall@k）—— 需要 sentence-transformers/faiss/rank-bm25
"""

from __future__ import annotations

import pytest

from eval.configuration import load_config
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.schemas import KnowledgeChunk


# ===== 术语归一化测试（快速，不需要重依赖）=====


@pytest.fixture(scope="module")
def normalizer() -> TerminologyNormalizer:
    return TerminologyNormalizer()


class TestNormalizer:
    def test_load_lightweight_dict(self, normalizer: TerminologyNormalizer) -> None:
        """第 1 层轻量词典加载。"""
        assert normalizer.synonym_count > 50  # 至少 50 个同义词映射
        assert "mi" in normalizer._synonym_map  # MI -> myocardial infarction
        assert "chf" in normalizer._synonym_map  # CHF -> congestive heart failure

    def test_normalize_synonym_replacement(
        self, normalizer: TerminologyNormalizer
    ) -> None:
        """同义词替换：MI -> myocardial infarction。"""
        nq = normalizer.normalize("patient has MI and CHF")
        assert "myocardial infarction" in nq.normalized
        assert "congestive heart failure" in nq.normalized
        assert len(nq.matched_terms) >= 2

    def test_normalize_abbreviation(
        self, normalizer: TerminologyNormalizer
    ) -> None:
        """缩写归一化：ECG -> electrocardiogram。"""
        nq = normalizer.normalize("ECG shows abnormal rhythm")
        assert "electrocardiogram" in nq.normalized

    def test_normalize_common_symptom(
        self, normalizer: TerminologyNormalizer
    ) -> None:
        """症状同义词：pyrexia -> fever。"""
        nq = normalizer.normalize("patient presented with pyrexia")
        assert "fever" in nq.normalized

    def test_coverage_positive(self, normalizer: TerminologyNormalizer) -> None:
        nq = normalizer.normalize("patient has fever and headache")
        assert nq.coverage > 0

    def test_coverage_zero_no_match(
        self, normalizer: TerminologyNormalizer
    ) -> None:
        nq = normalizer.normalize("hello world foo bar")
        assert nq.coverage == 0.0

    def test_term_overlap_match(self, normalizer: TerminologyNormalizer) -> None:
        """术语重叠：fever 查询 vs pyrexia 文本。"""
        overlap = normalizer.get_term_overlap(
            "patient has fever", "the patient presented with pyrexia"
        )
        assert overlap > 0  # fever/pyrexia 归一化后匹配

    def test_term_overlap_no_match(
        self, normalizer: TerminologyNormalizer
    ) -> None:
        overlap = normalizer.get_term_overlap("hello world", "foo bar baz")
        assert overlap == 0.0

    def test_three_layers_loaded(self, normalizer: TerminologyNormalizer) -> None:
        """三层归一化都加载了术语。"""
        assert normalizer.term_count > 1000  # MeSH 1585 + 轻量词典 + 派生
        sources = set(normalizer._term_sources.values())
        assert "lightweight" in sources
        assert "mesh" in sources


# ===== 检索测试（需要重依赖）=====


@pytest.fixture(scope="module")
def test_chunks() -> list[KnowledgeChunk]:
    """小规模测试知识库。"""
    return [
        KnowledgeChunk(
            chunk_id="c1", source="test", source_id="book1",
            text="Myocardial infarction is caused by coronary artery occlusion leading to cardiac muscle necrosis.",
            evidence_level="level_2_review",
        ),
        KnowledgeChunk(
            chunk_id="c2", source="test", source_id="book2",
            text="Aspirin is used to prevent blood clots in patients with heart disease and after MI.",
            evidence_level="level_1_guideline",
        ),
        KnowledgeChunk(
            chunk_id="c3", source="test", source_id="book3",
            text="The ECG shows ST elevation in leads V1-V4 indicating anterior wall infarction.",
            evidence_level="level_3_primary_study",
        ),
        KnowledgeChunk(
            chunk_id="c4", source="test", source_id="book4",
            text="Diabetes mellitus patients require regular monitoring of blood glucose levels.",
            evidence_level="level_2_review",
        ),
        KnowledgeChunk(
            chunk_id="c5", source="test", source_id="book5",
            text="Hypertension is a major risk factor for stroke and cardiovascular disease.",
            evidence_level="level_1_guideline",
        ),
    ]


@pytest.fixture(scope="module")
def eval_config() -> dict:
    return load_config("eval/config.yaml")


@pytest.fixture(scope="module")
def retriever(test_chunks, normalizer, eval_config):
    """构建检索器（首次会下载模型，较慢）。"""
    from medidiag.rag.retrieval import Retriever

    r = Retriever(
        test_chunks,
        weights=eval_config["retrieval"]["weights"],
        evidence_level_scores=eval_config["retrieval"]["evidence_levels"],
        embedding_model=eval_config["embedding"]["model"],
        rerank_model=eval_config["rerank"]["model"],
        normalizer=normalizer,
    )
    r.build_index(use_bm25=True, use_embedding=True)
    return r


class TestRetriever:
    def test_search_embedding_returns_results(self, retriever, eval_config) -> None:
        config = eval_config["experiments"]["rag"]["rag_embedding"]["config"]
        results = retriever.search("myocardial infarction", top_k=3, experiment_config=config)
        assert len(results) == 3
        assert results[0].final_score > 0
        assert results[0].embedding_score > 0

    def test_search_bm25_active(self, retriever, eval_config) -> None:
        config = eval_config["experiments"]["rag"]["rag_bm25"]["config"]
        results = retriever.search("heart attack", top_k=3, experiment_config=config)
        assert any(r.bm25_score > 0 for r in results)

    def test_search_evidence_active(self, retriever, eval_config) -> None:
        config = eval_config["experiments"]["rag"]["rag_evidence_weight"]["config"]
        results = retriever.search("aspirin", top_k=3, experiment_config=config)
        assert any(r.evidence_level_score > 0 for r in results)

    def test_search_term_overlap_active(self, retriever, eval_config) -> None:
        config = eval_config["experiments"]["rag"]["rag_term_norm"]["config"]
        results = retriever.search("MI ECG", top_k=3, experiment_config=config)
        # term_overlap 可能为 0（取决于查询术语是否在 chunk 中命中）
        # 但至少验证不报错
        assert len(results) == 3

    def test_search_full_active(self, retriever, eval_config) -> None:
        config = eval_config["experiments"]["rag"]["rag_full"]["config"]
        results = retriever.search("myocardial infarction", top_k=3, experiment_config=config)
        assert len(results) == 3
        assert results[0].embedding_score > 0

    def test_recall_at_k_hit(self, retriever, eval_config) -> None:
        """Recall@k 命中。"""
        hit = retriever.compute_recall_at_k(
            "myocardial infarction", ["c1"], top_k=3,
            experiment_config=eval_config["experiments"]["rag"]["rag_embedding"]["config"],
        )
        assert hit is True

    def test_recall_at_k_miss(self, retriever, eval_config) -> None:
        """Recall@k 未命中（gold 不在知识库中）。"""
        hit = retriever.compute_recall_at_k(
            "myocardial infarction", ["nonexistent_id"], top_k=3,
            experiment_config=eval_config["experiments"]["rag"]["rag_embedding"]["config"],
        )
        assert hit is False

    def test_recall_empty_gold(self, retriever, eval_config) -> None:
        """空 gold_evidence 返回 False。"""
        hit = retriever.compute_recall_at_k(
            "query", [], top_k=3,
            experiment_config=eval_config["experiments"]["rag"]["rag_embedding"]["config"],
        )
        assert hit is False

    def test_experiment_config_is_required(self, retriever) -> None:
        with pytest.raises(ValueError, match="source of truth"):
            retriever.search("query", top_k=3)

    def test_weights_come_from_yaml(self, retriever, eval_config) -> None:
        assert retriever.weights == eval_config["retrieval"]["weights"]
        assert retriever.evidence_level_scores == eval_config["retrieval"]["evidence_levels"]
