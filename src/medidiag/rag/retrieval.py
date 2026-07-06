"""检索模块占位。

阶段 4 实现：
- BM25 检索（rank-bm25）
- embedding 检索（FAISS）
- rerank（cross-encoder/ms-marco-MiniLM-L-6-v2）
- 证据等级加权
- 检索排序公式：
    final_score = w1*bm25 + w2*embedding + w3*evidence_level + w4*term_overlap
  权重写入 eval/config.yaml，参与消融实验

消融组别：
    A: 纯 embedding
    B: A + BM25
    C: A + evidence level weighting
    D: A + terminology normalization
    E: A + citation verifier
    F: A + B + C + D + E 全量组合
"""

from __future__ import annotations


class Retriever:
    """检索器骨架。

    TODO[阶段4]: 实现 BM25 + embedding + rerank + 证据等级加权。
    """

    def __init__(self) -> None:
        raise NotImplementedError("Retriever 将在阶段 4 实现")
