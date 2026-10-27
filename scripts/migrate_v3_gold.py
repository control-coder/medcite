"""把 v3 的 498 条逐字摘录迁移到 v4 新分块（M6），输出 gold 映射。

映射规则（忽略空白的包含关系）：

- ``exact``：摘录被单个子块逐字包含，映射到该子块；
- ``span``：摘录跨越相邻的多个子块（按文档内出现顺序连续），映射到覆盖它的
  最短的连续子块区间，区间里全部子块计为 gold；
- ``failed``：映射失败，记录原因。已知原因：M5 核对出的 9 条缺失/未匹配
  （快照缺失 8 条、图说定位差异 1 条，见 examples/public_health_v4/README.md）。

两种策略（recursive / parent_child）各输出一份映射。引用校验以子块为准，
因此 parent_child 只映射到子块（chunks.jsonl），父块不参与。

输出（examples/public_health_v4/v3_gold_mapping.json）只含 ID、映射类型与
失败原因，不含摘录全文。该文件同时记录每条 v3 查询的可迁移 gold 子块，
供 M6 离线对比与后续评测使用。

用法::

    python scripts/migrate_v3_gold.py \
        --v3 examples/public_health_v3 \
        --chunks .cache/corpus/chunks \
        --out examples/public_health_v4/v3_gold_mapping.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medidiag.rag.ingest.clean import normalize_text  # noqa: E402
from medidiag.schemas import read_jsonl  # noqa: E402
from scripts.verify_v3_excerpts import (  # noqa: E402  # noqa: E402
    _SLUG_ALIASES,
    _WHO_BASE,
    load_v4_who_map,
    normalize_slug,
)


def squash(text: str) -> str:
    """逐字比较用：先做与 M5 清洗一致的全半角/空白归一，再去掉全部空白。

    M5 的 ``clean_document`` 把全角标点统一为半角，而 v3 摘录保留了原始全角
    标点；若不做同样归一，含全角标点的摘录会全部失配（并非内容差异）。
    """
    return re.sub(r"\s+", "", normalize_text(text))


def build_v3_to_v4_doc(v3_dir: Path, v4_sources: Path) -> dict[str, str]:
    """v3 chunk 的 source_id → v4 doc_id。

    v3 的 source_id 是自建短名（如 flu_hygiene），到 v4 doc_id 的对应只能
    经 source_url：v3 sources.json 记录每个 source_id 的 URL，v4 清单记录
    URL → doc_id；slug 更新走 verify_v3_excerpts 登记的别名。
    """
    sources = json.loads((v3_dir / "sources.json").read_text(encoding="utf-8"))
    url_map = load_v4_who_map(v4_sources)
    norm_map = {normalize_slug(u): d for u, d in url_map.items()}
    out: dict[str, str] = {}
    for source in sources["sources"]:
        url = source["source_url"]
        effective = url
        if url.startswith(_WHO_BASE):
            alias = _SLUG_ALIASES.get(url[len(_WHO_BASE):])
            if alias:
                effective = _WHO_BASE + alias
        doc_id = url_map.get(effective) or norm_map.get(normalize_slug(effective))
        if doc_id:
            out[source["document_id"]] = doc_id
    return out


def map_excerpt(excerpt: str, child_ids: list[str], child_texts_squashed: list[str]) -> tuple[str, list[str], str | None]:
    """把一条摘录映射到子块。child_ids/texts 按文档内出现顺序排列。

    返回 (映射类型, gold 子块 ID 列表, 失败原因)。
    """
    target = squash(excerpt)
    if not target:
        return "failed", [], "摘录为空"
    # 单子块包含
    hits = [i for i, text in enumerate(child_texts_squashed) if target in text]
    if len(hits) == 1:
        return "exact", [child_ids[hits[0]]], None
    if len(hits) > 1:
        # 多块同时包含（重叠造成）：取块长最短者，映射仍记 exact
        best = min(hits, key=lambda i: (len(child_texts_squashed[i]), i))
        return "exact", [child_ids[best]], None
    # 跨相邻子块：找最短连续区间，使拼接文本（去空白）包含摘录
    joined_starts: list[int] = []  # 每个子块在整体拼接中的起始下标
    total = ""
    for text in child_texts_squashed:
        joined_starts.append(len(total))
        total += text
    if target not in total:
        return "failed", [], "摘录不在该文档任何子块或相邻拼接中"
    start = total.find(target)
    end = start + len(target)
    covered = [i for i, s in enumerate(joined_starts)
               if s < end and s + len(child_texts_squashed[i]) > start]
    # 连续性校验：covered 应是连续区间
    if covered and covered == list(range(covered[0], covered[-1] + 1)):
        return "span", [child_ids[i] for i in covered], None
    return "failed", [], "覆盖子块不连续"


def migrate_strategy(v3_chunks: list[dict], doc_map: dict[str, str],
                     chunks_path: Path) -> dict:
    """一种策略的迁移结果。"""
    v4_chunks = read_jsonl(chunks_path)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in v4_chunks:
        by_doc[chunk["source_id"]].append(chunk)
    # 不做排序：JSONL 的写入顺序即 chunk_corpus 的文档/章节/序号顺序，
    # 跨章节 chunk_id 字典序与出现顺序不一致，按列表顺序拼接才是原文顺序
    entries: dict[str, dict] = {}
    stats = {"exact": 0, "span": 0, "failed": 0}
    for chunk in v3_chunks:
        cid = chunk["chunk_id"]
        doc_id = doc_map.get(chunk["source_id"])
        if doc_id is None:
            entries[cid] = {"mapping": "failed", "gold": [], "reason": "v3 来源无 v4 对应文档（快照缺失）"}
            stats["failed"] += 1
            continue
        children = by_doc.get(doc_id)
        if not children:
            entries[cid] = {"mapping": "failed", "gold": [], "reason": f"v4 文档 {doc_id} 无子块"}
            stats["failed"] += 1
            continue
        mapping, gold, reason = map_excerpt(
            chunk["text"], [c["chunk_id"] for c in children],
            [squash(c["text"]) for c in children])
        entries[cid] = {"mapping": mapping, "gold": gold, "reason": reason}
        stats[mapping] += 1
    return {"stats": stats, "excerpts": entries}


def build_query_gold(v3_queries: list[dict], excerpts: dict[str, dict]) -> dict[str, dict]:
    """每条 v3 查询迁移后的 gold 子块与排除原因。"""
    out: dict[str, dict] = {}
    for query in v3_queries:
        gold: list[str] = []
        failed: list[str] = []
        for cid in query["gold_evidence_ids"]:
            entry = excerpts.get(cid)
            if entry is None or entry["mapping"] == "failed":
                failed.append(cid)
            else:
                gold.extend(entry["gold"])
        out[query["sample_id"]] = {
            "split": query["split"], "bucket": query["bucket"],
            "gold": sorted(dict.fromkeys(gold)),
            "failed_excerpts": failed,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=Path("examples/public_health_v3"))
    parser.add_argument("--v4-sources", type=Path, default=Path("examples/public_health_v4/sources.json"))
    parser.add_argument("--chunks", type=Path, default=Path(".cache/corpus/chunks"),
                        help="分块产物根目录，下含 <strategy>/chunks.jsonl")
    parser.add_argument("--strategies", nargs="+", default=["recursive", "parent_child"])
    parser.add_argument("--out", type=Path, default=Path("examples/public_health_v4/v3_gold_mapping.json"))
    args = parser.parse_args()

    v3_chunks = read_jsonl(args.v3 / "chunks.jsonl")
    v3_queries = read_jsonl(args.v3 / "queries.jsonl")
    doc_map = build_v3_to_v4_doc(args.v3, args.v4_sources)

    result: dict[str, object] = {
        "schema_version": "v3-gold-mapping-v1",
        "note": "v3 摘录 → v4 子块的映射（M6）；只含 ID 与映射类型，不含全文。"
                "exact=单子块逐字包含；span=跨相邻子块，全部计为 gold；failed=映射失败。",
        "strategies": {},
    }
    for strategy in args.strategies:
        chunks_path = args.chunks / strategy / "chunks.jsonl"
        migrated = migrate_strategy(v3_chunks, doc_map, chunks_path)
        queries = build_query_gold(v3_queries, migrated["excerpts"])
        answerable = [q for q in v3_queries if q["gold_evidence_ids"]]
        unmappable = [q["sample_id"] for q in answerable
                      if not queries[q["sample_id"]]["gold"]]
        result["strategies"][strategy] = {  # type: ignore[index]
            "stats": migrated["stats"],
            "answerable_queries": len(answerable),
            "excluded_queries": unmappable,
            "excluded_count": len(unmappable),
            "excerpts": migrated["excerpts"],
            "queries": queries,
        }
        print(strategy, migrated["stats"], f"排除查询 {len(unmappable)} 条")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("映射文件：", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
