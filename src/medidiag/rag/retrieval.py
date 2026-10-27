"""BM25 检索与中文切分。

应用有三路检索：本模块的 BM25、``dense.py`` 的向量检索，以及 ``hybrid.py``
的双路融合。两路单检索都返回 ``SearchResult``，由 ``runtime.py`` 封装成可追溯的证据包。

注意: rank-bm25 与 jieba 在建索引时才导入，模块本身可被 import。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from medidiag.schemas import KnowledgeChunk

# jieba 词典缓存目录：独立 Tokenizer 初始化时写缓存，不污染系统临时目录
_JIEBA_CACHE_DIR = ".cache/jieba"

# 汉字区段（CJK 统一表意文字扩展 A 到基本区），与 lexicon.py 的判定一致
_CJK_PART_RE = re.compile(r"[㐀-鿿]+|[a-z0-9]+")


def tokenize_text(text: str, mode: str = "whitespace", tokenizer: Any | None = None) -> list[str]:
    """BM25 切分：whitespace 按空格切；cjk_bigram 对连续汉字取相邻双字；jieba 用独立实例。

    jieba 模式转小写、去掉纯空白 token；确定性由独立的 ``jieba.Tokenizer``
    实例保证（关闭 HMM，不走全局默认实例），见 ``_new_jieba_tokenizer``。
    """
    if mode == "whitespace":
        return text.lower().split()
    if mode == "cjk_bigram":
        tokens: list[str] = []
        for part in _CJK_PART_RE.findall(text.lower()):
            if "㐀" <= part[0] <= "鿿" and len(part) > 1:
                tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
            else:
                tokens.append(part)
        return tokens
    if mode != "jieba":
        raise ValueError("不支持的 BM25 切分方式")
    if tokenizer is None:
        tokenizer = _default_jieba_tokenizer()
    return [word.lower() for word in tokenizer.cut(text) if word.strip()]


_jieba_tokenizer: Any | None = None


def _default_jieba_tokenizer() -> Any:
    """无自定义词典的进程级 jieba 分词器，只创建一次。"""
    global _jieba_tokenizer
    if _jieba_tokenizer is None:
        _jieba_tokenizer = _new_jieba_tokenizer()
    return _jieba_tokenizer


def _new_jieba_tokenizer(custom_words: list[str] | None = None) -> Any:
    """创建独立的 ``jieba.Tokenizer`` 实例，缓存写到 .cache/，可注入自定义词典。

    不用全局默认实例：自定义词典只影响本实例；关闭 HMM（新词发现）使切分
    完全由固定词典决定，同一输入始终得到同样的 token 序列。
    """
    import jieba

    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = _JIEBA_CACHE_DIR
    tokenizer.HMM = False
    if custom_words:
        for word in custom_words:
            tokenizer.add_word(word)
    return tokenizer


@dataclass
class SearchResult:
    """单个检索结果。final_score 就是该检索方式自己的分数。

    混合检索时 final_score 为 RRF 分数；某一路没有召回该块时对应分数为 0，
    由 ``score_sources`` 标出实际命中了哪几路（单路检索恒为自身一路，融合见 ``hybrid.py``），
    避免把 0 误读为该路的真实分数。
    """

    chunk_id: str
    final_score: float
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    chunk: KnowledgeChunk | None = None
    score_sources: tuple[str, ...] = ()


class Bm25Retriever:
    """单路 BM25 检索。

    分数除以本次查询的最高分，落在 [0, 1]；没有任何词命中时返回空列表，
    不靠全零分数硬凑证据；同分按 chunk_id 稳定排序。
    ``bm25_tokenizer="jieba"`` 时可注入自定义词典（见 ``lexicon.build_lexicon``）。
    """

    def __init__(self, chunks: list[KnowledgeChunk], bm25_tokenizer: str = "cjk_bigram",
                 lexicon: list[str] | None = None) -> None:
        if bm25_tokenizer not in {"whitespace", "cjk_bigram", "jieba"}:
            raise ValueError("不支持的 BM25 切分方式")
        self.bm25_tokenizer = bm25_tokenizer
        self.chunks = chunks
        self._bm25: Any | None = None  # rank_bm25.BM25Okapi，没有类型存根
        # jieba 模式使用独立实例；自定义词典只注入本实例，不污染全局默认词典
        self._jieba_tokenizer: Any | None = None
        if bm25_tokenizer == "jieba":
            self._jieba_tokenizer = _new_jieba_tokenizer(lexicon)

    def tokenize(self, text: str) -> list[str]:
        return tokenize_text(text, self.bm25_tokenizer, tokenizer=self._jieba_tokenizer)

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
                score_sources=("bm25",),
            )
            for i in order
        ]
