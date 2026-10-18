"""交叉编码器重排：把候选片段按与问题的相关度重新排序，模型固定版本且只读本地缓存，不联网下载。

向量检索把问题和片段分别编码再比较两个向量；交叉编码器把问题和片段拼在一起打分，能看到两边词语的对应，
排序更准，但每个候选都要单独算一次，所以只给少量候选重新排序：固定检索的前 N 段，或检索 Agent 多次检索的并集。
模型与候选数只在开发集上选定，依据见 docs/evaluation.md。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from medidiag.errors import MediDiagError

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# 每对（问题，片段）一个分数，分数越高越相关；可注入，测试不需要真实模型
PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


class Candidate(Protocol):
    @property
    def chunk_id(self) -> str: ...

    @property
    def text(self) -> str: ...


C = TypeVar("C", bound=Candidate)


def load_local_reranker(model: str, revision: str, *, device: str = "cpu", batch_size: int = 16,
                        max_length: int = 512) -> PairScorer:
    """加载固定版本的交叉编码器；本地没有缓存时直接报错，不尝试下载。"""
    from sentence_transformers import CrossEncoder

    try:
        encoder = CrossEncoder(model, revision=revision, device=device, local_files_only=True, max_length=max_length)
    except Exception as exc:
        raise MediDiagError(
            "RAG_INDEX_BUILD_FAILED",
            detail=f"重排模型 {model}@{revision[:8]} 不在本地缓存中；应用不会自动下载，请先手动准备模型",
        ) from exc
    return lambda pairs: [float(score) for score in encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)]


class Reranker:
    """对候选去重后按分数降序取前 n 段；同分按 chunk_id 稳定排序。"""

    def __init__(self, scorer: PairScorer, *, model_name: str, model_revision: str, candidates: int) -> None:
        self._scorer = scorer
        self.model_name = model_name
        self.model_revision = model_revision
        # 交给重排的候选数：固定检索取前 candidates 段；检索 Agent 每次检索各取前 candidates 段再合并
        self.candidates = candidates

    def rerank(self, question: str, items: Sequence[C], top_n: int) -> list[tuple[C, float]]:
        first: dict[str, C] = {}
        for item in items:  # 去重，保留先出现的那一个
            first.setdefault(item.chunk_id, item)
        unique = list(first.values())
        if not unique:
            return []
        scores = list(self._scorer([(question, item.text) for item in unique]))
        if len(scores) != len(unique):
            raise MediDiagError("RAG_INDEX_BUILD_FAILED", detail="重排分数数量与候选数量不一致")
        ranked = sorted(zip(unique, scores, strict=True), key=lambda pair: (-pair[1], pair[0].chunk_id))
        return ranked[:top_n]
