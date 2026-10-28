"""把解析阶段回填的 v4 元数据写回 sources.json，使提交的清单完整（只补空字段）。

回填字段：
- ``title``：WHO 事实清单取页面 h1/标题（解析产物 metadata.title）；
- ``published_at``：WHO 事实清单取页首日期行；疫情月报取页面「时间：<em>」发布日期；
  维基百科快照无修订时间戳，留空（数据卡说明，按 revid 定位版本）；
- ``accessed_at``：取 fetch_report.json 的抓取时间（ISO 时间戳）。

规则：只补空字段（None/空串），不覆盖清单中已有的显式值；除这三个字段外
不改任何其他内容；写回后清单仍是确定性文件，与解析产物一致。

用法::

    python scripts/backfill_v4_metadata.py \
        --sources examples/public_health_v4/sources.json \
        --parsed .cache/corpus/parsed/documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 疫情月报页面发布日期：「时间：<em>2024-11-20</em>」（标题在正文容器外）
_EPIDEMIC_DATE_RE = re.compile(r"时间：\s*<em>\s*(\d{4})-(\d{2})-(\d{2})\s*</em>")


def _load_parsed_metadata(parsed_path: Path) -> dict[str, dict[str, str]]:
    """读解析产物，取每篇文档回填后的 title / published_at。"""
    out: dict[str, dict[str, str]] = {}
    with parsed_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = json.loads(line)
            meta = doc.get("metadata", {})
            out[str(doc["doc_id"])] = {
                "title": str(meta.get("title") or ""),
                "published_at": str(meta.get("published_at") or ""),
            }
    return out


def _load_accessed_at(sources_path: Path) -> dict[str, str]:
    fetch_path = sources_path.parent / "fetch_report.json"
    if not fetch_path.exists():
        return {}
    payload = json.loads(fetch_path.read_text(encoding="utf-8"))
    return {str(r["doc_id"]): str(r["accessed_at"])
            for r in payload.get("records", []) if r.get("accessed_at")}


def _epidemic_published_at(raw_dir: Path, doc_id: str) -> str:
    """疫情月报发布日期：页面「时间：<em>YYYY-MM-DD</em>」，无则空串。"""
    raw_path = raw_dir / "epidemic_report" / f"{doc_id}.html"
    if not raw_path.exists():
        return ""
    html = raw_path.read_text(encoding="utf-8", errors="replace")
    match = _EPIDEMIC_DATE_RE.search(html)
    return match.group(0) and f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path,
                        default=Path("examples/public_health_v4/sources.json"))
    parser.add_argument("--parsed", type=Path,
                        default=Path(".cache/corpus/parsed/documents.jsonl"))
    parser.add_argument("--raw", type=Path, default=Path(".cache/corpus/raw"))
    parser.add_argument("--dry-run", action="store_true", help="只打印改动，不写回")
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text(encoding="utf-8-sig"))
    parsed_meta = _load_parsed_metadata(args.parsed)
    accessed_map = _load_accessed_at(args.sources)

    filled = {"title": 0, "published_at": 0, "accessed_at": 0}
    for entry in sources.get("documents", []):
        doc_id = str(entry["doc_id"])
        meta = parsed_meta.get(doc_id, {})
        if not entry.get("title") and meta.get("title"):
            entry["title"] = meta["title"]
            filled["title"] += 1
        if not entry.get("published_at"):
            published = meta.get("published_at")
            if not published and entry.get("source_type") == "epidemic_report":
                published = _epidemic_published_at(args.raw, doc_id)
            if published:
                entry["published_at"] = published
                filled["published_at"] += 1
        if not entry.get("accessed_at") and accessed_map.get(doc_id):
            entry["accessed_at"] = accessed_map[doc_id]
            filled["accessed_at"] += 1

    print(f"回填：title {filled['title']}、published_at {filled['published_at']}、"
          f"accessed_at {filled['accessed_at']}（共 {len(sources.get('documents', []))} 篇）")
    if args.dry_run:
        print("（dry-run，未写回）")
        return 0
    args.sources.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("已写回：", args.sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
