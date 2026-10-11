"""BM25 检索与中文切分。

应用只用两路检索：本模块的 BM25，或 ``dense.py`` 的向量检索。两者都返回
``SearchResult``，由 ``runtime.py`` 封装成可追溯的证据包。

注意: rank-bm25 在建索引时才导入，模块本身可被 import。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from medidiag.schemas import KnowledgeChunk


def tokenize_text(text: str, mode: str = "whitespace") -> list[str]:
    """BM25 切分：whitespace 按空格切；cjk_bigram 对连续汉字取相邻双字。"""
    if mode == "whitespace":
        return text.lower().split()
    if mode != "cjk_bigram":
        raise ValueError("不支持的 BM25 切分方式")
    tokens: list[str] = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", text.lower()):
        if "\u3400" <= part[0] <= "\u9fff" and len(part) > 1:
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


@dataclass
class SearchResult:
    """单个检索结果。final_score 就是该检索方式自己的分数。"""

    chunk_id: str
    final_score: float
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    chunk: KnowledgeChunk | None = None


class Bm25Retriever:
    """单路 BM25 检索。

    分数除以本次查询的最高分，落在 [0, 1]；没有任何词命中时返回空列表，
    不靠全零分数硬凑证据；同分按 chunk_id 稳定排序。
    """

    def __init__(self, chunks: list[KnowledgeChunk], bm25_tokenizer: str = "cjk_bigram") -> None:
        if bm25_tokenizer not in {"whitespace", "cjk_bigram"}:
            raise ValueError("不支持的 BM25 切分方式")
        self.bm25_tokenizer = bm25_tokenizer
        self.chunks = chunks
        self._bm25: Any | None = None  # rank_bm25.BM25Okapi，没有类型存根

    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text, self.bm25_tokenizer)

    def build_index(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([self.tokenize(chunk.text) for chunk in self.chunks])

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import numpy as np

        if self._bm25 is None:
            self.build_index()
        bm25 = self._bm25
        assert bm25 is not None
        raw = np.asarray(bm25.get_scores(self.tokenize(query)), dtype=np.float32)
        scores = raw / max(float(raw.max()), 1e-8)
        order = sorted(
            (i for i in range(len(scores)) if scores[i] > 0),
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )[:top_k]
        return [
            SearchResult(
                chunk_id=self.chunks[i].chunk_id,
                final_score=float(scores[i]),
                bm25_score=float(scores[i]),
                chunk=self.chunks[i],
            )
            for i in order
        ]
