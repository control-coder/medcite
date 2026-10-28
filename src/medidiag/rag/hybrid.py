"""混合检索（M8）：BM25 与向量检索双路召回，RRF 融合，重排后可用 MMR 去冗余。

- 双路各取 ``candidates`` 个候选（默认 20），倒数排名融合（RRF，k=60）；
  融合结果的 ``final_score`` 为 RRF 分数，``bm25_score`` 与 ``embedding_score``
  同时填写。某一路没有召回该块时对应分数为 0，``score_sources`` 标出实际命中了
  哪几路（``("bm25",)``、``("dense",)`` 或 ``("bm25", "dense")``），避免把 0
  误读为该路的真实分数。
- 最大边际相关（MMR，λ=0.7）用于重排之后去冗余：子块向量复用
  ``DenseRetriever.vector_for`` 的已编码向量，不重复编码；没有向量可用的候选
  相似度按 0 处理，即不参与压制、仍可按相关性入选。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from medidiag.rag.dense import DenseRetriever
from medidiag.rag.retrieval import Bm25Retriever, SearchResult
from medidiag.schemas import KnowledgeChunk

DEFAULT_CANDIDATES = 20
DEFAULT_RRF_K = 60
DEFAULT_MMR_LAMBDA = 0.7


class HybridRetriever:
    """BM25 与向量双路召回后按 RRF 融合；同分按 chunk_id 稳定排序。

    ``chunks`` 仅供参数校验与后续扩展；索引与召回委托给两路检索器，要求
    两路建立在同一份子块列表上（由 ``runtime._hybrid_factory`` 保证）。
    """

    def __init__(self, chunks: Sequence[KnowledgeChunk], *, bm25: Bm25Retriever, dense: DenseRetriever,
                 candidates: int = DEFAULT_CANDIDATES, rrf_k: int = DEFAULT_RRF_K) -> None:
        self.chunks = list(chunks)
        self.bm25 = bm25
        self.dense = dense
        self.candidates = candidates
        self.rrf_k = rrf_k
        # 供 runtime 的 retrieval_provenance 记录向量模型版本（与单路 dense 一致）
        self.embedding_model_name = dense.embedding_model_name
        self.embedding_model_revision = dense.embedding_model_revision

    def build_index(self) -> None:
        self.bm25.build_index()
        self.dense.build_index()

    def search(self, query: str, top_k: int = 5, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        """``allowed_ids`` 给出时两路都只在这些块里召回，再融合；None 时行为不变。"""
        extra = {} if allowed_ids is None else {"allowed_ids": allowed_ids}
        bm25_hits = self.bm25.search(query, top_k=self.candidates, **extra)
        dense_hits = self.dense.search(query, top_k=self.candidates, **extra)
        fused: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(bm25_hits):
            entry = fused.setdefault(hit.chunk_id, _empty_entry(hit))
            entry["rrf"] += 1.0 / (self.rrf_k + rank + 1)
            entry["bm25_score"] = hit.bm25_score
            entry["bm25_hit"] = True
        for rank, hit in enumerate(dense_hits):
            entry = fused.setdefault(hit.chunk_id, _empty_entry(hit))
            entry["rrf"] += 1.0 / (self.rrf_k + rank + 1)
            entry["embedding_score"] = hit.embedding_score
            entry["dense_hit"] = True
        order = sorted(fused, key=lambda cid: (-fused[cid]["rrf"], cid))[:top_k]
        return [
            SearchResult(
                chunk_id=cid,
                final_score=fused[cid]["rrf"],
                bm25_score=fused[cid]["bm25_score"],
                embedding_score=fused[cid]["embedding_score"],
                chunk=fused[cid]["chunk"],
                score_sources=tuple(
                    name
                    for name, hit in (
                        ("bm25", fused[cid]["bm25_hit"]), ("dense", fused[cid]["dense_hit"])
                    )
                    if hit
                ),
            )
            for cid in order
        ]

    def vector_for(self, chunk_id: str) -> Any | None:
        """按 chunk_id 取子块向量（委托给向量检索器，不重复编码）。"""
        return self.dense.vector_for(chunk_id)


def _empty_entry(hit: SearchResult) -> dict[str, Any]:
    return {"rrf": 0.0, "bm25_score": 0.0, "embedding_score": 0.0,
            "bm25_hit": False, "dense_hit": False, "chunk": hit.chunk}


def mmr_select(items: Sequence[Any], *, vector_for: Any, top_k: int,
               mmr_lambda: float = DEFAULT_MMR_LAMBDA) -> list[Any]:
    """最大边际相关去冗余：在保持相关性的同时压制与已选项近重复的候选。

    ``items`` 按相关性（final_score）降序排列；相关性先归一到本批次的最高分，
    每次选出 ``λ·归一化相关性 − (1−λ)·最大相似度`` 最高的一项（两个量都在 [0, 1]，
    λ 才是两者的相对权重）。向量由 ``vector_for(chunk_id)`` 提供（应是检索阶段
    已编码的归一化向量，不重复编码）；取不到向量的候选相似度按 0 处理，即不参与压制。
    负相似度（向量方向相反）会抬高多样性得分，这与 MMR 公式的定义一致。
    """
    import numpy as np

    best_score = max((float(item.final_score) for item in items), default=0.0)
    scale = max(best_score, 1e-12)
    remaining = list(items)
    selected: list[Any] = []
    selected_vectors: list[Any] = []
    while remaining and len(selected) < top_k:
        best_index = 0
        best_value = -float("inf")
        for index, item in enumerate(remaining):
            vector = vector_for(item.chunk_id)
            similarity = 0.0
            if vector is not None and selected_vectors:
                similarity = max(float(np.dot(vector, seen)) for seen in selected_vectors)
            value = mmr_lambda * float(item.final_score) / scale - (1.0 - mmr_lambda) * similarity
            if value > best_value:
                best_value = value
                best_index = index
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        chosen_vector = vector_for(chosen.chunk_id)
        if chosen_vector is not None:
            selected_vectors.append(chosen_vector)
    return selected
