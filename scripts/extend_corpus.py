"""在已有语料上加入更多 WHO 中文页面的短引，作为干扰项，用来测检索随语料变大的衰减。

输入是已抓取到仓库外缓存目录的页面网页（``--html-dir``，每页一个 ``<slug>.html``），
基础语料是 ``examples/public_health_v2``。新增的短引都是页面正文里连续 1 到 2 句的逐字原文，
选取规则固定、不看评测问题，所以同样的输入总是得到同样的输出。

查询、金标准和基础语料的每一行都原样保留；只在后面追加新的短引和来源。
用 ``--exclude-terms`` 挡掉可能让“语料没有涉及该疾病”类问题变得可回答的页面和短引，
挡完之后仍需要人工检查新增短引是否在无意中回答了某条查询（见 docs/evaluation.md）。

用法：
    python scripts/extend_corpus.py --html-dir .cache/who_raw --pages-out .cache/who_pages \\
        --base examples/public_health_v2 --out examples/public_health_v3 --max-pages 80
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_corpus import html_to_text, squash  # noqa: E402

SOURCE_BASE = "https://www.who.int/zh/news-room/fact-sheets/detail/"
ACCESSED_AT = "2026-10-10"
USAGE = "仅保留必要正文短引和出处用于非商业工程演示；未取得全文再分发许可，不包含图片、附件，不授予下游全文使用权。"
SCOPE = "历史公开科普片段，不保证覆盖现行指南，不用于个体诊疗。"
VERIFICATION = "2026-10-10 实际 HTTP 200，解码正文逐字包含此短摘录；未将全文入库。"
EXCERPTS_PER_PAGE = 4
MIN_CHARS, MAX_CHARS = 30, 110
# 带这些内容的句子要么依赖上下文、要么是参考文献或链接，不适合单独作为短引
_REJECT = re.compile(r"\(\d|\d\)|（\d|http|参见|见表|见图|见上文|下列|以下|如下|本实况报道|[:：]")
_DATE = re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$")
_STOP_HEADINGS = ("世卫组织的应对", "参考文献", "世卫组织对")
_SENTENCE = re.compile(r"[^。]+。")
_DEPENDENT_START = ("此外", "然而", "但是", "因此", "同时", "这", "它", "其", "他们", "该", "上述", "另外", "不过", "也就是说")
# 页面顶部的图片说明和署名所在的“小节”，其下的段落不是正文
_CAPTION_SECTIONS = ("Skip to main content", "世卫组织 /", "世卫组织/")


@dataclass
class Page:
    slug: str
    title: str
    published_at: str
    paragraphs: list[tuple[str, str]]  # (所属小节, 段落正文)


def _is_heading(line: str) -> bool:
    """短行且不像列表项的是小节标题；以“和”“及”“；”结尾的多半是被拆开的列表项。"""
    if not 2 <= len(line) <= 20 or re.search(r"[。；，：、]", line):
        return False
    return not line.endswith(("和", "及", "或", "的", "以及"))


def parse_page(slug: str, text: str) -> Page | None:
    """页面纯文本 → 标题、日期、各小节的段落；找不到标题或日期的页面返回 None。"""
    lines = text.splitlines()
    if not lines:
        return None
    title = lines[0].strip()
    published = ""
    for line in lines[:12]:
        match = _DATE.match(line.strip())
        if match:
            published = f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
            break
    if not title or not published:
        return None
    paragraphs: list[tuple[str, str]] = []
    section = ""
    for line in lines[1:]:
        line = line.strip()
        if any(line.startswith(stop) for stop in _STOP_HEADINGS):
            break
        if _is_heading(line):
            section = line
        elif len(line) >= MIN_CHARS and line.endswith("。") and not section.startswith(_CAPTION_SECTIONS):
            paragraphs.append((section, line))
    return Page(slug, title, published, paragraphs)


def window_candidates(paragraph: str, blocked: tuple[str, ...]) -> list[str]:
    """段落里连续 1 到 2 句、长度合适且不依赖上下文的窗口，按出现顺序。"""
    sentences = _SENTENCE.findall(paragraph)
    windows: list[str] = []
    for i, first in enumerate(sentences):
        for count in (1, 2):
            window = "".join(sentences[i:i + count])
            if len(sentences[i:i + count]) < count or not MIN_CHARS <= len(window) <= MAX_CHARS:
                continue
            if _REJECT.search(window) or first.startswith(_DEPENDENT_START):
                continue
            if any(term in window for term in blocked):
                continue
            windows.append(window)
    return windows


def pick_excerpts(page: Page, blocked: tuple[str, ...]) -> list[tuple[str, str]]:
    """在有可用短引的段落中等间隔取 EXCERPTS_PER_PAGE 个，每段只取第一个窗口。"""
    usable = []
    for section, paragraph in page.paragraphs:
        windows = window_candidates(paragraph, blocked)
        if windows:
            usable.append((section, windows[0]))
    if len(usable) <= EXCERPTS_PER_PAGE:
        return usable
    step = len(usable) / EXCERPTS_PER_PAGE
    return [usable[int(i * step)] for i in range(EXCERPTS_PER_PAGE)]


def slug_id(slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")[:40]


def chunk_record(page: Page, index: int, section: str, text: str) -> dict[str, Any]:
    sid = slug_id(page.slug)
    return {
        "chunk_id": f"kb_{sid}_{index:02d}", "source": f"世界卫生组织｜{page.title}", "source_id": sid,
        "text": text, "evidence_level": "level_5_other",
        "metadata": {"title": page.title, "source_url": SOURCE_BASE + page.slug, "published_at": page.published_at,
                     "accessed_at": ACCESSED_AT, "language": "zh-CN", "section": section,
                     "content_kind": "verbatim_short_excerpt", "usage": USAGE, "scope": SCOPE,
                     "verification": VERIFICATION}}


def build(html_dir: Path, pages_out: Path, base: Path, out: Path, max_pages: int, blocked: tuple[str, ...]) -> dict[str, Any]:
    base_chunk_lines = [line for line in (base / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    base_sources = json.loads((base / "sources.json").read_text(encoding="utf-8"))
    base_urls = {source["source_url"] for source in base_sources["sources"]}
    pages_out.mkdir(parents=True, exist_ok=True)
    picked: list[tuple[Page, list[tuple[str, str]]]] = []
    skipped: dict[str, str] = {}
    # 页面按 slug 的哈希排序再截取，不看内容，结果可复现
    for html_path in sorted(html_dir.glob("*.html"), key=lambda p: hashlib.sha256(p.stem.encode()).hexdigest()):
        slug = html_path.stem
        if SOURCE_BASE + slug in base_urls:
            continue
        text = html_to_text(html_path.read_text(encoding="utf-8", errors="replace"))
        page = parse_page(slug, text)
        if page is None:
            skipped[slug] = "缺少标题或日期"
            continue
        if any(term in page.title for term in blocked):
            skipped[slug] = "标题含屏蔽词"
            continue
        excerpts = pick_excerpts(page, blocked)
        if len(excerpts) < 3:
            skipped[slug] = "可用短引不足 3 条"
            continue
        (pages_out / f"{slug_id(slug)}.txt").write_text(text, encoding="utf-8")
        picked.append((page, excerpts))
        if len(picked) == max_pages:
            break
    out.mkdir(parents=True, exist_ok=True)
    new_chunks = [chunk_record(page, i, section, text)
                  for page, excerpts in picked for i, (section, text) in enumerate(excerpts, 1)]
    (out / "chunks.jsonl").write_text(
        "\n".join(base_chunk_lines + [json.dumps(c, ensure_ascii=False) for c in new_chunks]) + "\n", encoding="utf-8")
    shutil.copyfile(base / "queries.jsonl", out / "queries.jsonl")
    new_sources = [{
        "document_id": slug_id(page.slug), "chunk_ids": [f"kb_{slug_id(page.slug)}_{i:02d}" for i in range(1, len(excerpts) + 1)],
        "sections": [section for section, _ in excerpts], "publisher": "世界卫生组织", "title": page.title,
        "source_url": SOURCE_BASE + page.slug, "published_at": page.published_at, "accessed_at": ACCESSED_AT,
        "language": "zh-CN", "content_kind": "verbatim_short_excerpt", "usage": USAGE, "scope": SCOPE,
        "verification": VERIFICATION} for page, excerpts in picked]
    version = {"version": "public-health-excerpts-v3", "sources": base_sources["sources"] + new_sources}
    (out / "sources.json").write_text(json.dumps(version, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"base_chunks": len(base_chunk_lines), "new_pages": len(picked), "new_chunks": len(new_chunks),
            "skipped": skipped, "squashed_check": all(
                squash(c["text"]) in squash((pages_out / f"{c['source_id']}.txt").read_text(encoding="utf-8"))
                for c in new_chunks)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--html-dir", type=Path, required=True)
    parser.add_argument("--pages-out", type=Path, required=True, help="写入页面纯文本，供 public_corpus.py verify 使用（放仓库外）")
    parser.add_argument("--base", type=Path, default=Path("examples/public_health_v2"))
    parser.add_argument("--out", type=Path, default=Path("examples/public_health_v3"))
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--exclude-terms", nargs="*", default=[], help="页面标题或短引含这些词时丢弃")
    args = parser.parse_args()
    summary = build(args.html_dir, args.pages_out, args.base, args.out, args.max_pages, tuple(args.exclude_terms))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["squashed_check"] else 1


if __name__ == "__main__":
    sys.exit(main())
