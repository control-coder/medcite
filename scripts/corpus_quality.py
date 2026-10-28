"""v4 语料数据质量检查工具（M4–M6 数据质量修复轮）。

对分块产物（``chunks.jsonl``）统计已知 5 类问题与规模指标，输出 JSON 摘要，
用于修复前后对比。不联网、不依赖 GPU。

统计口径（与修复轮一致）：

1. ``title_equals_doc_id``：``metadata.title == metadata.doc_id``（清单缺 title 的块）；
2. ``accessed_at_null``：``metadata.accessed_at`` 为空；
3. ``page_header_noise``：正文含页首杂质（Skip to main content / 页面日期行 / 「实况报道」标签）；
   ``cr_in_text``：文本含 ``\\r``；
4. ``ascii_dominant``：去空白后 ASCII 占比 ≥80% 且块长 ≥20 字（多为参考文献/外链/页码）；
5. ``short_chunks``：块长 <15 字。

另报告子块/父块数、平均字数（按 ``--parents`` 可选纳入父块统计）与按来源的分布。

用法::

    python scripts/corpus_quality.py \
        --chunks .cache/corpus/chunks/parent_child/chunks.jsonl \
        --parents .cache/corpus/chunks/parent_child/parents.jsonl \
        --out artifacts/reports/quality/before-parent-child-20261028.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medidiag.schemas import read_jsonl  # noqa: E402

# 页首杂质：WHO 页面控件/日期行/栏目标签（与 clean.py、chunking.py 的规则一致）
_PAGE_HEADER_RES = (
    re.compile(r"Skip to main content"),
    re.compile(r"^\d{4}年\d{1,2}月\d{1,2}日$"),
    re.compile(r"^实况报道$"),
)
_ASCII_MIN_LEN = 20
_ASCII_MIN_RATIO = 0.8
_SHORT_MAX_CHARS = 15


def _ascii_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if ch.isascii()) / len(chars)


def _is_page_header_noise(text: str) -> bool:
    stripped = text.strip()
    return any(p.search(stripped) for p in _PAGE_HEADER_RES)


def analyze(chunks_path: Path, parents_path: Path | None) -> dict:
    chunks = read_jsonl(chunks_path)
    parents = read_jsonl(parents_path) if parents_path and parents_path.exists() else []

    by_source: Counter[str] = Counter()
    title_eq_doc = by_source.copy()
    accessed_null = by_source.copy()
    noise = by_source.copy()
    cr = by_source.copy()
    ascii_dom = by_source.copy()
    short = by_source.copy()

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        stype = str(meta.get("source_type", ""))
        by_source[stype] += 1
        text = str(chunk.get("text", ""))
        if meta.get("title") == meta.get("doc_id"):
            title_eq_doc[stype] += 1
        if not meta.get("accessed_at"):
            accessed_null[stype] += 1
        if _is_page_header_noise(text):
            noise[stype] += 1
        if "\r" in text:
            cr[stype] += 1
        if len(text) >= _ASCII_MIN_LEN and _ascii_ratio(text) >= _ASCII_MIN_RATIO:
            ascii_dom[stype] += 1
        if len(text) < _SHORT_MAX_CHARS:
            short[stype] += 1

    child_lens = [len(str(c.get("text", ""))) for c in chunks]
    parent_lens = [len(str(p.get("text", ""))) for p in parents]
    return {
        "chunks_file": str(chunks_path),
        "child_chunks": len(chunks),
        "parent_chunks": len(parents),
        "child_chars_mean": round(statistics.mean(child_lens), 1) if child_lens else 0.0,
        "parent_chars_mean": round(statistics.mean(parent_lens), 1) if parent_lens else 0.0,
        "problems": {
            "title_equals_doc_id": {"total": sum(title_eq_doc.values()),
                                    "by_source": dict(title_eq_doc)},
            "accessed_at_null": {"total": sum(accessed_null.values()),
                                 "by_source": dict(accessed_null)},
            "page_header_noise": {"total": sum(noise.values()), "by_source": dict(noise)},
            "cr_in_text": {"total": sum(cr.values()), "by_source": dict(cr)},
            "ascii_dominant": {"total": sum(ascii_dom.values()),
                               "by_source": dict(ascii_dom)},
            "short_chunks": {"total": sum(short.values()), "by_source": dict(short)},
        },
        "chunks_by_source": dict(by_source),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, required=True, help="检索块 chunks.jsonl")
    parser.add_argument("--parents", type=Path, default=None, help="父块 parents.jsonl（可选）")
    parser.add_argument("--out", type=Path, default=None, help="JSON 摘要输出路径（缺省打印到标准输出）")
    args = parser.parse_args()

    result = analyze(args.chunks, args.parents)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print("质量摘要：", args.out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
