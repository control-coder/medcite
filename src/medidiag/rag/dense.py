"""应用用的向量检索：只做余弦相似度排序，模型固定版本且只读本地缓存，不联网下载。

与评测（``eval/retrieval_benchmark.py`` 的 ``DenseScorer``）使用同一套做法：
查询加 BGE 的检索前缀、向量归一化后用内积排序、同分按 chunk_id 稳定排序。
对外满足 ``RuntimeMedicalRAG`` 要求的检索器接口（``build_index``、``search``），因此能替换 BM25 检索器而不改其他阶段。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from medidiag.errors import MediDiagError
from medidiag.rag.retrieval import SearchResult
from medidiag.schemas import KnowledgeChunk

INDEX_DIR = ".cache/index"
PGVECTOR_DSN_ENV = "MEDIDIAG_PGVECTOR_DSN"
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
                 model_revision: str, query_prefix: str = BGE_ZH_QUERY_PREFIX,
                 store: Any = None, store_base: Path | None = None) -> None:
        """``store_base`` 给出时使用 NumpyStore 并缓存到该路径（同语料同模型版本不重新编码）；
        ``store`` 给出时使用外部向量存储（pgvector），语料须已用 ``medidiag corpus sync`` 写入。"""
        self.chunks = list(chunks)
        self._store = store
        self._store_base = store_base
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.cache_hit = False
        self.embedding_model_name = model_name
        self.embedding_model_revision = model_revision
        self._encoder = encoder
        self._query_prefix = query_prefix
        self._matrix: Any = None

    def build_index(self) -> None:
        import numpy as np

        if self._store_base is not None:
            self._build_numpy_cache()
            return
        if self._store is not None:
            if not set(self._by_id) <= self._store.chunk_ids():
                raise MediDiagError("RAG_INDEX_BUILD_FAILED",
                                    detail="向量存储缺少当前语料的块，请先运行 medidiag corpus sync")
            return
        vectors = np.asarray(self._encoder([chunk.text for chunk in self.chunks]), dtype=np.float64)
        self._matrix = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def _build_numpy_cache(self) -> None:
        from medidiag.rag.store import NumpyStore, sync_corpus

        base = self._store_base
        assert base is not None
        self.cache_hit = NumpyStore.exists(base)
        store = NumpyStore.load(base) if self.cache_hit else NumpyStore()
        report = sync_corpus(store, self.chunks, self._encoder)
        if report.encoded_chunks or report.removed or not self.cache_hit:
            store.save(base)
        self._store = store

    def search(self, query: str, top_k: int = 5, allowed_ids: set[str] | None = None) -> list[SearchResult]:
        """``allowed_ids`` 给出时只返回这些块。内存矩阵与 NumpyStore 在取 top_k 前屏蔽；
        外部向量存储（pgvector）只能后过滤：先取全部候选再筛，大语料下代价高，且受存储端近似检索的候选数限制，
        可能不足 top_k，由上层的放宽重查兜底。"""
        import numpy as np

        if self._matrix is None and self._store is None:
            raise MediDiagError("RAG_INDEX_BUILD_FAILED", detail="向量索引尚未构建")
        vector = np.asarray(self._encoder([self._query_prefix + query]), dtype=np.float64)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if self._store is not None:
            fetch = top_k * 2 if allowed_ids is None else len(self._by_id)
            hits = [h for h in self._store.search(vector, fetch)
                    if h.chunk_id in self._by_id and (allowed_ids is None or h.chunk_id in allowed_ids)][:top_k]
            return [SearchResult(chunk_id=h.chunk_id, final_score=h.score, embedding_score=h.score,
                                 chunk=self._by_id[h.chunk_id], score_sources=("dense",)) for h in hits]
        scores = self._matrix @ vector
        candidates = (range(len(self.chunks)) if allowed_ids is None
                      else [i for i, c in enumerate(self.chunks) if c.chunk_id in allowed_ids])
        order = sorted(candidates, key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id))
        return [SearchResult(chunk_id=self.chunks[i].chunk_id, final_score=float(scores[i]),
                             embedding_score=float(scores[i]), chunk=self.chunks[i],
                             score_sources=("dense",)) for i in order[:top_k]]

    def vector_for(self, chunk_id: str) -> Any | None:
        """按 chunk_id 取已建索引的归一化向量（不重复编码）；未建索引或未收录返回 None。

        供混合检索后的 MMR 去冗余直接复用子块向量，见 ``hybrid.py``。
        """
        if self._store is not None:
            found = self._store.vectors([chunk_id]).get(chunk_id)
            return None if found is None else found.astype("float64")
        if self._matrix is None:
            return None
        for index, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                return self._matrix[index]
        return None


def store_options(chunks: Sequence[KnowledgeChunk], config: dict[str, Any]) -> dict[str, Any]:
    """按 ``retrieval.vector_store`` 生成 DenseRetriever 的存储参数；未配置时为空，保持原有每次重新编码。"""
    kind = config["retrieval"].get("vector_store")
    if kind is None:
        return {}
    if kind == "numpy":
        from medidiag.rag.store import cache_base

        return {"store_base": cache_base(INDEX_DIR, chunks, str(config["embedding"]["revision"]))}
    dsn = os.environ.get(PGVECTOR_DSN_ENV)
    if not dsn:
        raise MediDiagError("RAG_INDEX_BUILD_FAILED", detail=f"vector_store=pgvector 需要环境变量 {PGVECTOR_DSN_ENV}")
    from medidiag.rag.store.pg_store import PgVectorStore

    return {"store": PgVectorStore(dsn)}
