"""应用用的向量检索：只做余弦相似度排序，模型固定版本且只读本地缓存，不联网下载。

与评测（``eval/retrieval_benchmark.py`` 的 ``DenseScorer``）使用同一套做法：
查询加 BGE 的检索前缀、向量归一化后用内积排序、同分按 chunk_id 稳定排序。
对外满足 ``RuntimeMedicalRAG`` 要求的检索器接口，因此能替换 BM25 检索器而不改其他阶段。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from medidiag.errors import MediDiagError
from medidiag.rag.retrieval import SearchResult
from medidiag.schemas import KnowledgeChunk

BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
Encoder = Callable[[list[str]], Any]


def load_local_encoder(model: str, revision: str, *, device: str = "cpu", batch_size: int = 8) -> Encoder:
    """加载固定版本的句向量模型；本地没有缓存时直接报错，不尝试下载。"""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    try:
        encoder = SentenceTransformer(model, revision=revision, device=device, local_files_only=True)
    except Exception as exc:
        raise MediDiagError(
            "RAG_INDEX_BUILD_FAILED",
            detail=f"向量模型 {model}@{revision[:8]} 不在本地缓存中；应用不会自动下载，请先手动准备模型",
        ) from exc
    return lambda texts: np.asarray(
        encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size))


class DenseRetriever:
    """按余弦相似度返回前 k 个片段；没有正分门槛，模型是否作答由后面的生成阶段判断。"""

    def __init__(self, chunks: Sequence[KnowledgeChunk], *, encoder: Encoder, model_name: str,
                 model_revision: str, query_prefix: str = BGE_ZH_QUERY_PREFIX) -> None:
        self.chunks = list(chunks)
        self.embedding_model_name = model_name
        self.embedding_model_revision = model_revision
        self.rerank_model_name = None
        self.rerank_model_revision = None
        self._encoder = encoder
        self._query_prefix = query_prefix
        self._matrix: Any = None

    def build_index(self, use_bm25: bool = True, use_embedding: bool = True) -> None:
        import numpy as np

        del use_bm25, use_embedding  # 本检索器只有向量一路
        vectors = np.asarray(self._encoder([chunk.text for chunk in self.chunks]), dtype=np.float64)
        self._matrix = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def search(self, query: str, top_k: int = 5,
               experiment_config: dict[str, bool] | None = None) -> list[SearchResult]:
        import numpy as np

        del experiment_config
        if self._matrix is None:
            raise MediDiagError("RAG_INDEX_BUILD_FAILED", detail="向量索引尚未构建")
        vector = np.asarray(self._encoder([self._query_prefix + query]), dtype=np.float64)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self._matrix @ vector
        order = sorted(range(len(self.chunks)), key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        return [SearchResult(chunk_id=self.chunks[i].chunk_id, final_score=float(scores[i]),
                             embedding_score=float(scores[i]), chunk=self.chunks[i]) for i in order[:top_k]]

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        del query
        return candidates[:top_k]
