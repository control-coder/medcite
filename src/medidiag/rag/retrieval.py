"""检索模块：BM25 + embedding + rerank + 证据等级加权 + 消融配置。

检索排序公式（PLAN.md）：
    final_score = w1*bm25 + w2*embedding + w3*evidence_level + w4*term_overlap

权重从 eval/config.yaml 读取，参与消融实验。

消融组别：
    A: 纯 embedding（基线）
    B: A + BM25
    C: A + evidence level weighting
    D: A + terminology normalization
    E: A + citation verifier（阶段 5 实现，阶段 4 降级为 A）
    F: A + B + C + D 全量组合

注意: sentence-transformers / faiss / rank-bm25 在方法内部延迟导入，
模块本身可被 import（调用方法时才需要依赖）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.schemas import KnowledgeChunk


# 证据等级 -> 分数映射（PLAN.md evidence_levels）
EVIDENCE_LEVEL_SCORES: dict[str, float] = {
    "level_1_guideline": 1.0,
    "level_2_review": 0.8,
    "level_3_primary_study": 0.9,
    "level_4_case_report": 0.5,
    "level_5_other": 0.3,
}

# 消融组别配置（对应 eval/config.yaml 的 ablation.groups）
ABLATION_CONFIGS: dict[str, dict[str, bool]] = {
    "A": {
        "use_bm25": False,
        "use_evidence_weighting": False,
        "use_term_normalization": False,
        "use_citation_verifier": False,
    },
    "B": {
        "use_bm25": True,
        "use_evidence_weighting": False,
        "use_term_normalization": False,
        "use_citation_verifier": False,
    },
    "C": {
        "use_bm25": False,
        "use_evidence_weighting": True,
        "use_term_normalization": False,
        "use_citation_verifier": False,
    },
    "D": {
        "use_bm25": False,
        "use_evidence_weighting": False,
        "use_term_normalization": True,
        "use_citation_verifier": False,
    },
    "E": {
        # citation verifier 在阶段 5 实现，阶段 4 降级为 A
        "use_bm25": False,
        "use_evidence_weighting": False,
        "use_term_normalization": False,
        "use_citation_verifier": True,
    },
    "F": {
        "use_bm25": True,
        "use_evidence_weighting": True,
        "use_term_normalization": True,
        "use_citation_verifier": True,
    },
}

# 默认权重（对应 eval/config.yaml retrieval.weights）
DEFAULT_WEIGHTS: dict[str, float] = {
    "w1_bm25": 0.25,
    "w2_embedding": 0.45,
    "w3_evidence_level": 0.20,
    "w4_term_overlap": 0.10,
}


@dataclass
class SearchResult:
    """单个检索结果。"""

    chunk_id: str
    final_score: float
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    evidence_level_score: float = 0.0
    term_overlap: float = 0.0
    chunk: KnowledgeChunk | None = None


class Retriever:
    """检索器。

    支持 BM25 + embedding + rerank + 证据等级加权 + 术语归一化。
    消融配置 A-F 可切换。

    用法:
        retriever = Retriever(chunks, weights=...)
        retriever.build_index()
        results = retriever.search(query, top_k=5, ablation_group="F")
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        weights: dict[str, float] | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        normalizer: TerminologyNormalizer | None = None,
    ) -> None:
        # 优先使用本地模型路径（避免 HuggingFace 缓存问题）
        local_model = os.path.join(os.getcwd(), ".cache", "model_all_MiniLM")
        if embedding_model == "sentence-transformers/all-MiniLM-L6-v2" and os.path.isdir(local_model):
            embedding_model = local_model
        self.chunks = chunks
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.embedding_model_name = embedding_model
        self.rerank_model_name = rerank_model
        self.normalizer = normalizer

        self._texts = [c.text for c in chunks]
        self._bm25 = None
        self._embedder = None
        self._reranker = None
        self._chunk_embeddings = None
        self._faiss_index = None

    def build_index(
        self, use_bm25: bool = True, use_embedding: bool = True
    ) -> None:
        """构建检索索引。"""
        if use_bm25:
            self._build_bm25()
        if use_embedding:
            self._build_embedding_index()

    def _build_bm25(self) -> None:
        """构建 BM25 索引。"""
        from rank_bm25 import BM25Okapi

        tokenized = [text.lower().split() for text in self._texts]
        self._bm25 = BM25Okapi(tokenized)

    def _build_embedding_index(self) -> None:
        """构建 embedding + FAISS 索引。"""
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)
        embeddings = self._embedder.encode(
            self._texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._chunk_embeddings = np.array(embeddings, dtype=np.float32)

        dim = self._chunk_embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)  # 内积 = cosine（已归一化）
        self._faiss_index.add(self._chunk_embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
        ablation_group: str = "F",
    ) -> list[SearchResult]:
        """检索 top-k 相关 chunks。

        根据消融组别使用不同的检索策略和排序公式。

        Args:
            query: 查询文本。
            top_k: 返回结果数。
            ablation_group: 消融组别 A-F。

        Returns:
            SearchResult 列表（按 final_score 降序）。
        """
        import numpy as np

        config = ABLATION_CONFIGS.get(ablation_group, ABLATION_CONFIGS["F"])

        # 术语归一化（D/F 组）
        normalized_query = query
        if config["use_term_normalization"] and self.normalizer:
            nq = self.normalizer.normalize(query)
            normalized_query = nq.normalized

        # 计算各维度分数
        bm25_scores = None
        if config["use_bm25"]:
            bm25_scores = self._get_bm25_scores(normalized_query)

        embedding_scores = self._get_embedding_scores(normalized_query)

        evidence_scores = None
        if config["use_evidence_weighting"]:
            evidence_scores = self._get_evidence_scores()

        term_overlaps = None
        if config["use_term_normalization"] and self.normalizer:
            # 优化: query 只归一化一次，不为每个 chunk 重复归一化
            if nq is None:
                nq = self.normalizer.normalize(query)
            unique_preferred = list(
                {m.preferred.lower() for m in nq.matched_terms}
            )
            if unique_preferred:
                # 构建 preferred -> variants 映射（一次）
                pref_variants: dict[str, list[str]] = {}
                for syn, pref in self.normalizer._synonym_map.items():
                    pl = pref.lower()
                    if pl not in pref_variants:
                        pref_variants[pl] = [pl]
                    pref_variants[pl].append(syn)

                term_overlaps = np.zeros(len(self._texts), dtype=np.float32)
                for i, text in enumerate(self._texts):
                    text_lower = text.lower()
                    hits = 0
                    for term in unique_preferred:
                        variants = pref_variants.get(term, [term])
                        for v in variants:
                            if re.search(
                                r"\b" + re.escape(v) + r"\b", text_lower
                            ):
                                hits += 1
                                break
                    term_overlaps[i] = hits / len(unique_preferred)
            else:
                term_overlaps = np.zeros(
                    len(self._texts), dtype=np.float32
                )

        # 检索排序公式: final_score = w1*bm25 + w2*embedding + w3*evidence + w4*term_overlap
        w = self.weights
        scores = w["w2_embedding"] * embedding_scores  # A 组基线

        if bm25_scores is not None:
            scores = scores + w["w1_bm25"] * bm25_scores
        if evidence_scores is not None:
            scores = scores + w["w3_evidence_level"] * evidence_scores
        if term_overlaps is not None:
            scores = scores + w["w4_term_overlap"] * term_overlaps

        # 排序取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    chunk_id=self.chunks[idx].chunk_id,
                    final_score=float(scores[idx]),
                    bm25_score=float(bm25_scores[idx]) if bm25_scores is not None else 0.0,
                    embedding_score=float(embedding_scores[idx]),
                    evidence_level_score=float(evidence_scores[idx])
                    if evidence_scores is not None
                    else 0.0,
                    term_overlap=float(term_overlaps[idx])
                    if term_overlaps is not None
                    else 0.0,
                    chunk=self.chunks[idx],
                )
            )

        return results

    def _get_bm25_scores(self, query: str):
        """获取 BM25 分数（归一化到 [0, 1]）。"""
        import numpy as np

        if self._bm25 is None:
            self._build_bm25()
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        max_score = max(scores.max(), 1e-8)
        return np.array(scores, dtype=np.float32) / max_score

    def _get_embedding_scores(self, query: str):
        """获取 embedding cosine 相似度分数。"""
        import numpy as np

        if self._embedder is None or self._faiss_index is None:
            self._build_embedding_index()
        query_vec = self._embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.array(query_vec, dtype=np.float32)
        scores, _ = self._faiss_index.search(query_vec, len(self._texts))
        return scores[0]

    def _get_evidence_scores(self):
        """获取证据等级分数。"""
        import numpy as np

        return np.array(
            [
                EVIDENCE_LEVEL_SCORES.get(c.evidence_level, 0.3)
                for c in self.chunks
            ],
            dtype=np.float32,
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """用 cross-encoder rerank 候选结果。

        Args:
            query: 查询文本。
            candidates: 候选结果列表。
            top_k: 返回结果数。

        Returns:
            rerank 后的 SearchResult 列表。
        """
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(self.rerank_model_name)

        pairs = [(query, c.chunk.text) for c in candidates if c.chunk]
        scores = self._reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i].final_score = float(score)

        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]

    def compute_recall_at_k(
        self,
        query: str,
        gold_evidence_ids: list[str],
        top_k: int = 5,
        ablation_group: str = "F",
    ) -> bool:
        """计算单个查询的 Recall@k 是否命中。

        Recall@k = top_k 结果中至少命中 1 条 gold_evidence。

        Args:
            query: 查询文本。
            gold_evidence_ids: 标准证据 chunk_id 列表。
            top_k: top-k。
            ablation_group: 消融组别。

        Returns:
            True 如果至少命中 1 条 gold_evidence。
        """
        if not gold_evidence_ids:
            return False
        results = self.search(query, top_k=top_k, ablation_group=ablation_group)
        result_ids = {r.chunk_id for r in results}
        gold_set = set(gold_evidence_ids)
        return len(result_ids & gold_set) > 0
