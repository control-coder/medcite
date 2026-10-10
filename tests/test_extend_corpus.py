"""语料扩充脚本的测试：页面解析、短引选取规则、v2 前缀原样保留；不联网。"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import extend_corpus as ec

BODY_1 = "洗手可以减少病菌传播，也能降低腹泻的发生概率。保持厨房清洁很重要。"
BODY_2 = "充足的睡眠有助于身体恢复，成年人每天需要七到九个小时的睡眠时间。"
BODY_3 = "饮水应当来自经过处理的安全水源，避免直接饮用未经处理的河水。"
BODY_4 = "适量运动可以降低多种慢性病的风险，建议每周累计至少一百五十分钟。"
PAGE_TEXT = "\n".join([
    "示例健康问题", "Skip to main content", "示例健康问题", "2025年12月8日", "阅读时间", "概述",
    BODY_1, "预防", BODY_2, "生活方式", BODY_3, "运动", BODY_4,
    "参考文献", "(1) 某某。参考文献不应进入正文。",
])


def test_parse_page_reads_title_date_and_sections() -> None:
    page = ec.parse_page("example", PAGE_TEXT)
    assert page is not None
    assert page.title == "示例健康问题"
    assert page.published_at == "2025-12-08"
    assert [section for section, _ in page.paragraphs] == ["概述", "预防", "生活方式", "运动"]
    assert all("参考文献不应进入正文" not in paragraph for _, paragraph in page.paragraphs)


def test_parse_page_without_date_is_skipped() -> None:
    assert ec.parse_page("example", "标题\n没有日期的正文。") is None


def test_caption_section_paragraphs_are_not_body() -> None:
    text = "\n".join(["标题", "2025年1月2日", "世卫组织 / 某某", "这是一段图片说明文字，长度足够但不是正文内容的句子。"])
    page = ec.parse_page("example", text)
    assert page is not None and page.paragraphs == []


def test_window_candidates_reject_dependent_and_blocked_sentences() -> None:
    paragraph = "此外，这类措施同样适用于家庭环境中的日常防护工作。饭前便后经常用肥皂和清水洗手，是预防多种传染病最简单有效的办法之一。详见 http://example.org 的说明内容。"
    windows = ec.window_candidates(paragraph, blocked=())
    assert "饭前便后经常用肥皂和清水洗手，是预防多种传染病最简单有效的办法之一。" in windows
    assert not any("此外" in w or "http" in w for w in windows)
    assert ec.window_candidates(paragraph, blocked=("洗手",)) == []


def test_pick_excerpts_is_deterministic_and_capped() -> None:
    page = ec.parse_page("example", PAGE_TEXT)
    assert page is not None
    first = ec.pick_excerpts(page, ())
    assert first == ec.pick_excerpts(page, ())
    assert len(first) <= ec.EXCERPTS_PER_PAGE
    assert all(ec.MIN_CHARS <= len(text) <= ec.MAX_CHARS for _, text in first)


def test_slug_id_is_ascii_identifier() -> None:
    assert ec.slug_id("Human-Papilloma Virus_(HPV)") == "human_papilloma_virus_hpv"
    assert len(ec.slug_id("a" * 80)) == 40


def _write_base(base: Path) -> None:
    base.mkdir()
    chunk = {"chunk_id": "kb_old_01", "source_id": "old", "text": "旧语料中的一条摘录内容。"}
    (base / "chunks.jsonl").write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
    (base / "queries.jsonl").write_text('{"sample_id": "q1"}\n', encoding="utf-8")
    sources = {"version": "v2", "sources": [{"source_url": ec.SOURCE_BASE + "old-page"}]}
    (base / "sources.json").write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")


def _html(text: str) -> str:
    return "<html><body>" + "".join(f"<p>{line}</p>" for line in text.splitlines()) + "</body></html>"


def test_build_keeps_base_prefix_and_new_chunks_are_verbatim(tmp_path: Path) -> None:
    base, html_dir = tmp_path / "base", tmp_path / "html"
    _write_base(base)
    html_dir.mkdir()
    (html_dir / "old-page.html").write_text(_html(PAGE_TEXT), encoding="utf-8")  # v2 已有的页面不重复收
    (html_dir / "new-page.html").write_text(_html(PAGE_TEXT), encoding="utf-8")
    (html_dir / "blocked-page.html").write_text(_html(PAGE_TEXT.replace("示例健康问题", "痛风须知")), encoding="utf-8")

    summary = ec.build(html_dir, tmp_path / "pages", base, tmp_path / "out", max_pages=10, blocked=("痛风",))

    assert summary["squashed_check"] is True
    assert summary["new_pages"] == 1
    assert summary["skipped"] == {"blocked-page": "标题含屏蔽词"}
    old_lines = (base / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    new_lines = (tmp_path / "out" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert new_lines[: len(old_lines)] == old_lines
    added = [json.loads(line) for line in new_lines[len(old_lines):]]
    assert len(added) == summary["new_chunks"] >= 3
    assert {c["source_id"] for c in added} == {"new_page"}
    assert (tmp_path / "out" / "queries.jsonl").read_bytes() == (base / "queries.jsonl").read_bytes()
    sources = json.loads((tmp_path / "out" / "sources.json").read_text(encoding="utf-8"))
    assert sources["version"] == "public-health-excerpts-v3"
    assert len(sources["sources"]) == 2


def test_shipped_v3_keeps_v2_prefix() -> None:
    root = Path(__file__).resolve().parents[1] / "examples"
    v2 = (root / "public_health_v2/chunks.jsonl").read_text(encoding="utf-8").splitlines()
    v3 = (root / "public_health_v3/chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert v3[: len(v2)] == v2
    assert (root / "public_health_v2/queries.jsonl").read_bytes() == (root / "public_health_v3/queries.jsonl").read_bytes()
    ids = [json.loads(line)["chunk_id"] for line in v3]
    assert len(ids) == len(set(ids))
