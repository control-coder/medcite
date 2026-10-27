"""核对 v3 的 498 条逐字摘录在新解析的 WHO 正文中能否找到。

用 v3 sources.json 的 source_url 对应到 .cache/who_raw/ 的页面快照，
经 rag/ingest 的 HTML 解析器解析后，忽略空白逐字核对每条摘录。
未找到的条目按来源分组列出，并区分两类原因：
- 快照缺失：who_raw 没有该页面（v2 的 51 篇不在缓存中）；
- 未匹配：快照存在但摘录不在解析后的正文中（解析问题或页面已更新）。

这个脚本不修改任何文件，第二阶段会在 M4 新快照上再跑一次。

用法：
    python scripts/verify_v3_excerpts.py \
        --v3 examples/public_health_v3 --who-raw .cache/who_raw
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medidiag.rag.ingest.html import parse_html  # noqa: E402

_WHO_BASE = "https://www.who.int/zh/news-room/fact-sheets/detail/"

# WHO 页面 slug 随时间更新：v3 清单里的旧 slug → v4 清单里的新 slug。
# 只登记实际遇到的、且已在新页面核对过摘录仍逐字存在的映射。
_SLUG_ALIASES = {
    "ebola-virus-disease": "ebola-disease",
}


def squash(text: str) -> str:
    """去掉全部空白，用于逐字比较。"""
    return re.sub(r"\s+", "", text)


def slug_candidates(slug: str) -> list[str]:
    """WHO 快照文件名对 URL slug 的转义形式：括号转成 -_..._ 。"""
    return [
        slug,
        re.sub(r"\(([^)]*)\)", r"-_\1_", slug),
        slug.replace("(", "-_").replace(")", "_"),
    ]


def find_snapshot(slug: str, who_raw: Path) -> Path | None:
    for candidate in slug_candidates(slug):
        path = who_raw / f"{candidate}.html"
        if path.exists():
            return path
    return None


def normalize_slug(url: str) -> str:
    """把 URL 末段规范化为可比较形式：小写、括号转义统一、去连字符差异。"""
    slug = url.rstrip("/").rsplit("/", 1)[-1].lower()
    # 括号两种写法统一：-(x) 与 -_x_ 都去掉括号字符后比较
    slug = re.sub(r"[-_()]+", "-", slug).strip("-")
    return slug


def load_v4_who_map(v4_sources: Path) -> dict[str, str]:
    """v4 清单中 WHO 事实清单的 source_url → doc_id。"""
    data = json.loads(v4_sources.read_text(encoding="utf-8"))
    return {d["url"]: d["doc_id"] for d in data["documents"]
            if d["source_type"] == "who_factsheet"}


def build_parsed_texts(v3_dir: Path, who_raw: Path,
                       v4_sources: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """解析 v3 全部来源页面，返回 {document_id: 解析后正文(去空白)} 与缺失清单。

    优先从 v4 清单映射到新快照（who_raw 指向 .cache/corpus/raw/who_factsheet）；
    映射不到时回退到旧转义文件名（兼容 .cache/who_raw 布局）。
    """
    sources = json.loads((v3_dir / "sources.json").read_text(encoding="utf-8"))
    url_map = load_v4_who_map(v4_sources) if v4_sources else {}
    # 规范化 slug → (url, doc_id)，处理括号转义差异
    norm_map = {normalize_slug(u): (u, d) for u, d in url_map.items()}

    texts: dict[str, str] = {}
    missing: list[str] = []
    for source in sources["sources"]:
        doc_id = source["document_id"]
        url = source["source_url"]
        if not url.startswith(_WHO_BASE):
            missing.append(doc_id)
            continue
        path: Path | None = None
        # slug 更新：v3 旧 slug 有别名时换成新 URL 再匹配
        old_slug = url[len(_WHO_BASE):]
        aliased = _SLUG_ALIASES.get(old_slug)
        effective_url = (_WHO_BASE + aliased) if aliased else url
        # 1) v4 清单精确匹配（含别名后的 URL）
        new_id = (url_map.get(effective_url)
                  or (norm_map.get(normalize_slug(effective_url)) or (None, None))[1])
        if new_id:
            candidate = who_raw / f"{new_id}.html"
            if candidate.exists():
                path = candidate
        # 2) 旧转义文件名回退
        if path is None:
            path = find_snapshot(old_slug, who_raw)
        if path is None:
            missing.append(doc_id)
            continue
        html = path.read_text(encoding="utf-8", errors="replace")
        document = parse_html(html, doc_id, source_type="who_factsheet")
        texts[doc_id] = squash(document.full_text())
    return texts, missing


def verify(v3_dir: Path, who_raw: Path, v4_sources: Path | None = None) -> dict[str, object]:
    """核对全部摘录，返回统计与未找到条目明细。"""
    texts, missing = build_parsed_texts(v3_dir, who_raw, v4_sources)
    unmatched: list[dict[str, str]] = []
    found = 0
    total = 0
    for line in (v3_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        total += 1
        text = texts.get(chunk["source_id"])
        if text is None:
            continue  # 快照缺失，单独报告
        if squash(chunk["text"]) in text:
            found += 1
        else:
            unmatched.append({
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "text": chunk["text"][:60],
            })
    # 按来源分组统计未匹配
    by_source: dict[str, int] = defaultdict(int)
    for item in unmatched:
        by_source[item["source_id"]] += 1
    return {
        "total_chunks": total,
        "snapshots_missing_sources": sorted(missing),
        "snapshots_missing_chunks": sum(
            1 for line in (v3_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["source_id"] in set(missing)
        ),
        "found": found,
        "unmatched": unmatched,
        "unmatched_by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=Path("examples/public_health_v3"))
    parser.add_argument("--who-raw", type=Path, default=Path(".cache/who_raw"))
    parser.add_argument("--v4-sources", type=Path, default=None,
                        help="可选：v4 sources.json，按 URL 映射到新快照 doc_id")
    parser.add_argument("--out", type=Path, default=None, help="可选：把结果写成 JSON")
    args = parser.parse_args()
    result = verify(args.v3, args.who_raw, args.v4_sources)
    summary = {
        "total": result["total_chunks"],
        "found": result["found"],
        "missing_snapshot_chunks": result["snapshots_missing_chunks"],
        "unmatched": len(result["unmatched"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.out}")
    return 0 if not result["unmatched"] else 1


if __name__ == "__main__":
    sys.exit(main())
