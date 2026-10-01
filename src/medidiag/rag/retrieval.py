"""检索模块：BM25 + embedding + rerank + 证据等级加权。

检索排序公式（PLAN.md）：
    final_score = w1*bm25 + w2*embedding + w3*evidence_level + w4*term_overlap

权重、模型和实验开关必须由调用方从 eval/config.yaml 显式传入。

注意: sentence-transformers / faiss / rank-bm25 在方法内部延迟导入，
模块本身可被 import（调用方法时才需要依赖）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from medidiag.acceleration import resolve_torch_device
from medidiag.rag.normalizer import TerminologyNormalizer
from medidiag.schemas import KnowledgeChunk


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
    用法:
        retriever = Retriever(chunks, weights=..., evidence_level_scores=...,
                              embedding_model=..., rerank_model=...,
                              embedding_revision=..., rerank_revision=...)
        retriever.build_index()
        results = retriever.search(query, top_k=5, experiment_config=...)
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        weights: dict[str, float],
        evidence_level_scores: dict[str, float],
        embedding_model: str,
        rerank_model: str,
        normalizer: TerminologyNormalizer | None = None,
        embedding_revision: str | None = None,
        rerank_revision: str | None = None,
        embedding_encoder: Any | None = None,
        device: str = "auto",
        embedding_batch_size: int = 32,
        rerank_batch_size: int = 32,
        bm25_tokenizer: str = "whitespace",
    ) -> None:
        if bm25_tokenizer not in {"whitespace", "cjk_bigram"}:
            raise ValueError("不支持的 BM25 切分方式")
        self.bm25_tokenizer = bm25_tokenizer
        self.chunks = chunks
        self.weights = dict(weights)
        self.evidence_level_scores = dict(evidence_level_scores)
        self.embedding_model_name = embedding_model
        self.rerank_model_name = rerank_model
        self.embedding_model_revision = embedding_revision
        self.rerank_model_revision = rerank_revision
        self.normalizer = normalizer
        self.requested_device = device
        self.actual_device = resolve_torch_device(device)
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.rerank_batch_size = max(1, int(rerank_batch_size))

        self._texts = [c.text for c in chunks]
        # 这四个是惰性构建的第三方对象（BM25Okapi / SentenceTransformer /
        # CrossEncoder / faiss.IndexFlatIP），都没有可用的类型存根，因此标注为
        # `Any | None`：`None` 表示尚未构建，构建入口统一返回已构建对象，
        # 使用点不再面对 Optional。
        self._bm25: Any | None = None
        # 测试可注入 encoder 以避免网络访问；正式 runner 不传该参数。
        self._embedder: Any | None = embedding_encoder
        self._reranker: Any | None = None
        self._chunk_embeddings: Any | None = None
        self._faiss_index: Any | None = None
        self._embedding_score_cache: dict[str, Any] = {}
        self._bm25_score_cache: dict[str, Any] = {}
        self._cache_hits = {"embedding": 0, "bm25": 0}
        self._cache_misses = {"embedding": 0, "bm25": 0}

    def build_index(
        self, use_bm25: bool = True, use_embedding: bool = True
    ) -> None:
        """构建检索索引。"""
        if use_bm25:
            self._build_bm25()
        if use_embedding:
            self._build_embedding_index()

    def tokenize(self, text: str) -> list[str]:
        """中英文采用同一查询/正文规则；默认保留历史空格切分。"""
        if self.bm25_tokenizer == "whitespace":
            return text.lower().split()
        tokens: list[str] = []
        for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.lower()):
            if "\u3400" <= part[0] <= "\u9fff" and len(part) > 1:
                tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            else:
                tokens.append(part)
        return tokens

    def _build_bm25(self) -> Any:
        """构建 BM25 索引并返回它。

        返回而不是只写 `self._bm25`，使调用点可以直接拿到非 None 的对象。
        """
        from rank_bm25 import BM25Okapi

        tokenized = [self.tokenize(text) for text in self._texts]
        self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def _build_embedding_index(self) -> Any:
        """构建 embedding + FAISS 索引，返回 FAISS 索引。"""
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if self._embedder is None:
            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                revision=self.embedding_model_revision,
                device=self.actual_device,
            )
        embeddings = self._encode_texts(self._texts)
        chunk_embeddings = np.array(embeddings, dtype=np.float32)
        self._chunk_embeddings = chunk_embeddings

        index = faiss.IndexFlatIP(chunk_embeddings.shape[1])  # 内积 = cosine（已归一化）
        index.add(chunk_embeddings)
        self._faiss_index = index
        return index

    def search(
        self,
        query: str,
        top_k: int = 5,
        experiment_config: dict[str, bool] | None = None,
    ) -> list[SearchResult]:
        """检索 top-k 相关 chunks。

        根据消融组别使用不同的检索策略和排序公式。

        Args:
            query: 查询文本。
            top_k: 返回结果数。
            experiment_config: 从 eval/config.yaml 读取的 RAG 开关。

        Returns:
            SearchResult 列表（按 final_score 降序）。
        """
        import numpy as np

        if experiment_config is None:
            raise ValueError("experiment_config is required; eval/config.yaml is the source of truth")
        config = experiment_config

        # 术语归一化（D/F 组）
        normalized_query = query
        if config["use_term_normalization"] and self.normalizer:
            nq = self.normalizer.normalize(query)
            normalized_query = nq.normalized

        # 计算各维度分数
        bm25_scores = None
        if config["use_bm25"]:
            bm25_scores = self._get_bm25_scores(normalized_query)

        # 产品小回归可显式关闭向量检索，默认保留历史研究组行为。
        use_embedding = config.get("use_embedding", True)
        embedding_scores = (self._get_embedding_scores(normalized_query) if use_embedding
                            else np.zeros(len(self.chunks), dtype=np.float32))

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
        if use_embedding:
            top_indices: list[int] = np.argsort(scores)[::-1][:top_k].tolist()
        else:
            # 词法路径无命中时不能靠全零分数硬凑证据，同分按 ID 稳定排序。
            top_indices = sorted(
                (i for i in range(len(scores)) if scores[i] > 0),
                key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
            )[:top_k]

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

    def _encode_texts(self, texts: list[str]) -> Any:
        embedder = self._embedder
        if embedder is None:
            # 只有 `_build_embedding_index` 与 `_get_embedding_scores` 会到这里，
            # 二者都先保证 embedder 存在；显式报错好过 None 的 AttributeError。
            raise RuntimeError(
                "embedding model is not built; call build_index() first"
            )
        kwargs = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
            "batch_size": self.embedding_batch_size,
        }
        try:
            return embedder.encode(texts, **kwargs)
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            kwargs.pop("batch_size")
            return embedder.encode(texts, **kwargs)

    def _get_bm25_scores(self, query: str) -> Any:
        """获取与 chunk 原始顺序对齐的 BM25 分数（归一化到 [0, 1]）。"""
        import numpy as np

        cached = self._bm25_score_cache.get(query)
        if cached is not None:
            self._cache_hits["bm25"] += 1
            return cached
        self._cache_misses["bm25"] += 1
        bm25 = self._bm25 if self._bm25 is not None else self._build_bm25()
        tokenized_query = self.tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        max_score = max(scores.max(), 1e-8)
        normalized = np.array(scores, dtype=np.float32) / max_score
        self._bm25_score_cache[query] = normalized
        return normalized

    def _get_embedding_scores(self, query: str) -> Any:
        """获取与 ``self.chunks`` 原始顺序严格对齐的 cosine 分数。

        FAISS 返回按相似度排序后的 ``scores`` 和对应 ``indices``。必须按
        indices 回填；否则排序后分数会被错误绑定到原始 chunk 下标。
        """
        import numpy as np

        cached = self._embedding_score_cache.get(query)
        if cached is not None:
            self._cache_hits["embedding"] += 1
            return cached
        self._cache_misses["embedding"] += 1
        if self._embedder is None or self._faiss_index is None:
            index = self._build_embedding_index()
        else:
            index = self._faiss_index
        query_vec = self._encode_texts([query])
        query_vec = np.array(query_vec, dtype=np.float32)
        ranked_scores, ranked_indices = index.search(query_vec, len(self._texts))
        aligned_scores = np.zeros(len(self._texts), dtype=np.float32)
        aligned_scores[ranked_indices[0]] = ranked_scores[0]
        self._embedding_score_cache[query] = aligned_scores
        return aligned_scores

    def cache_stats(self) -> dict[str, dict[str, int]]:
        """返回检索分数组件缓存计数，避免与 provider KV cache 混淆。"""
        return {
            name: {"hits": self._cache_hits[name], "misses": self._cache_misses[name]}
            for name in ("embedding", "bm25")
        }

    def _get_evidence_scores(self) -> Any:
        """获取证据等级分数。"""
        import numpy as np

        return np.array(
            [
                self.evidence_level_scores.get(c.evidence_level, 0.0)
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
        reranker = self._reranker
        if reranker is None:
            from sentence_transformers import CrossEncoder

            reranker = CrossEncoder(
                self.rerank_model_name,
                revision=self.rerank_model_revision,
                device=self.actual_device,
            )
            self._reranker = reranker

        # 只有带 chunk 的候选可以打分。必须先固定这个子集，再把分数写回同一个
        # 子集：早期版本按未过滤的 candidates 下标回写，任何 chunk=None 的候选
        # 都会让其后每个候选拿到别人的分数（静默错排，不报错）。
        # 同时保留 chunk 引用，避免在下面重复解包 Optional。
        scorable_pairs = [(c, c.chunk) for c in candidates if c.chunk is not None]
        if not scorable_pairs:
            return []
        scorable = [candidate for candidate, _ in scorable_pairs]

        pairs = [(query, chunk.text) for _, chunk in scorable_pairs]
        scores = reranker.predict(pairs, batch_size=self.rerank_batch_size)

        for candidate, score in zip(scorable, scores, strict=True):
            candidate.final_score = float(score)

        # 无 chunk 的候选被丢弃而不是保留过期的 final_score，且返回新列表，
        # 不就地重排调用方传入的 candidates。
        ranked = sorted(scorable, key=lambda x: x.final_score, reverse=True)
        return ranked[:top_k]

    def compute_recall_at_k(
        self,
        query: str,
        gold_evidence_ids: list[str],
        top_k: int = 5,
        experiment_config: dict[str, bool] | None = None,
    ) -> bool:
        """计算单个查询的 Recall@k 是否命中。

        Recall@k = top_k 结果中至少命中 1 条 gold_evidence。

        Args:
            query: 查询文本。
            gold_evidence_ids: 标准证据 chunk_id 列表。
            top_k: top-k。
            experiment_config: 从 eval/config.yaml 读取的 RAG 开关。

        Returns:
            True 如果至少命中 1 条 gold_evidence。
        """
        if not gold_evidence_ids:
            return False
        results = self.search(
            query, top_k=top_k, experiment_config=experiment_config
        )
        result_ids = {r.chunk_id for r in results}
        gold_set = set(gold_evidence_ids)
        return len(result_ids & gold_set) > 0
