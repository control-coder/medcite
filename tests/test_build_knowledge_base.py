"""知识库构建脚本的切分单元测试。

只测 `scripts/build_knowledge_base.py` 的纯函数，不读取真实教材、不写知识库。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "_build_knowledge_base", _ROOT / "scripts" / "build_knowledge_base.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MAX_CHUNK_CHARS = _MODULE.MAX_CHUNK_CHARS
split_long_block = _MODULE.split_long_block
split_textbook_paragraphs = _MODULE.split_textbook_paragraphs


class TestSplitTextbookParagraphs:
    """空行切分的既有行为必须保持不变。"""

    def test_normal_paragraphs_unchanged(self) -> None:
        first = "A" * 200
        second = "B" * 300
        content = f"{first}\n\n{second}"
        assert split_textbook_paragraphs(content) == [first, second]

    def test_min_chars_filter_drops_short_blocks(self) -> None:
        content = "short\n\n" + "C" * 60
        assert split_textbook_paragraphs(content) == ["C" * 60]

    def test_min_chars_filter_applies_to_split_pieces(self) -> None:
        # 上限 60，最后一句只有 10 字符，切出后应被 min_chars 过滤掉。
        block = ("word " * 30).strip() + ". Tiny tail."
        pieces = split_textbook_paragraphs(block, min_chars=20, max_chars=60)
        assert pieces
        assert all(len(piece) >= 20 for piece in pieces)
        assert "Tiny tail." not in pieces

    def test_single_megablock_is_capped(self) -> None:
        # 无空行的巨块：复现 Surgery_Schwartz 的形状（250 换行、无空行边界）。
        sentence = "The patient presented with an unremarkable finding. "
        block = sentence * 2000
        pieces = split_textbook_paragraphs(block)
        assert len(pieces) > 1
        assert max(len(piece) for piece in pieces) <= MAX_CHUNK_CHARS

    def test_whitespace_only_block_is_dropped(self) -> None:
        assert split_textbook_paragraphs("   \n\n\t\n\n") == []


class TestSplitLongBlock:
    """超长块切分：优先句边界，残片硬切，始终不越界。"""

    def test_prefers_sentence_boundaries(self) -> None:
        first = "A" * 90 + "."
        second = "B" * 90 + "."
        pieces = split_long_block(f"{first} {second}", 100)
        assert pieces == [first, second]

    def test_greedily_packs_sentences_under_cap(self) -> None:
        sentences = " ".join(f"{'x' * 30}." for _ in range(6))
        pieces = split_long_block(sentences, 100)
        assert all(len(piece) <= 100 for piece in pieces)
        # 每片应装进多于一句，否则不是贪心合并。
        assert max(piece.count(".") for piece in pieces) > 1

    def test_oversized_sentence_is_hard_split(self) -> None:
        # 单句自身越界，句边界无法帮忙，只能定长硬切。
        block = "Z" * 250
        pieces = split_long_block(block, 100)
        assert [len(piece) for piece in pieces] == [100, 100, 50]
        assert "".join(pieces) == block

    def test_cjk_sentence_boundary(self) -> None:
        block = "。 ".join("患者主诉不适" * 6 for _ in range(4))
        pieces = split_long_block(block, 60)
        assert all(len(piece) <= 60 for piece in pieces)

    def test_never_exceeds_cap_on_mixed_input(self) -> None:
        block = " ".join(
            ["short." , "Y" * 500, "another sentence here.", "W" * 40 + "."]
        )
        pieces = split_long_block(block, 120)
        assert pieces
        assert all(len(piece) <= 120 for piece in pieces)
