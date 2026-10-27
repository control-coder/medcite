"""BM25 jieba 分词的自定义词典：由语料标题与章节路径按规则生成，不手工维护词表。

jieba 的默认词典对中文医学主题词（如「人工流产」「青少年精神卫生」）切分不稳，
会把完整主题词拆碎，影响 BM25 命中。这里从子块 metadata 的 ``title`` 与
``section_path`` 按确定性规则抽取候选词，生成自定义词典：

- 去掉中英文括号及括号内内容，按「、」「和」「与」切分；
- 只保留长度 2–12、且至少含一个汉字的词（过滤纯数字与纯英文）；
- 输出排序去重，同一语料生成的词条集合与顺序完全一致（幂等）。

生成是纯函数：同样的语料输入得到同样的词典，可复现、可测试。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from medidiag.schemas import KnowledgeChunk

# 括号及其内容（中英文括号），标题里的补充说明不参与主题词
_PAREN_RE = re.compile(r"[（(][^）)]*[）)]")
# 并列切分符：顿号、「和」、「与」
_SPLIT_RE = re.compile(r"[、，,]|和|与")
# 至少一个汉字才算主题词
_CJK_RE = re.compile(r"[㐀-鿿]")

MIN_WORD_CHARS = 2
MAX_WORD_CHARS = 12


def extract_terms(text: str) -> list[str]:
    """从单个标题或章节名按规则抽取候选词；确定性、保序。"""
    cleaned = _PAREN_RE.sub("", text)
    terms: list[str] = []
    for part in _SPLIT_RE.split(cleaned):
        word = part.strip().strip("：:。；; ")
        if MIN_WORD_CHARS <= len(word) <= MAX_WORD_CHARS and _CJK_RE.search(word):
            terms.append(word)
    return terms


def build_lexicon(chunks: Iterable[KnowledgeChunk]) -> list[str]:
    """从子块 metadata 的 title 与 section_path 生成自定义词典，排序去重。"""
    terms: set[str] = set()
    for chunk in chunks:
        metadata = chunk.metadata
        title = metadata.get("title")
        if isinstance(title, str):
            terms.update(extract_terms(title))
        section_path = metadata.get("section_path")
        if isinstance(section_path, list):
            for section in section_path:
                if isinstance(section, str):
                    terms.update(extract_terms(section))
    return sorted(terms)
