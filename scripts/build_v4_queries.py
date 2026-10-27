"""构建 v4 评测查询集（M7 阶段 A：只构造查询，不做相关性标注）。

输入：
- ``examples/public_health_v3/queries.jsonl``（v3 的 168 条，sample_id 与 question 原样保留）；
- ``examples/public_health_v4/queries_authored.jsonl``（新编写约 60 条）；
- Huatuo26M-Lite（FreedomIntelligence，Apache-2.0）在线抽样，只取问题文本与原始 ID。

处理：合并 → 补 ``origin``/``intent``/``expected_filters``/``bucket``/``answerable``
→ 对照解析后全文做泄露检查 → 按 origin × intent 分层、固定种子划分 dev 60 条。

输出：
- ``examples/public_health_v4/queries.jsonl``（阶段 A 草案，不冻结；阶段 B 标注后冻结）；
- ``examples/public_health_v4/huatuo_sample.jsonl``（Huatuo 抽样，附来源与许可字段）。

用法::

    python scripts/build_v4_queries.py                    # 全流程（含联网抽样 Huatuo）
    python scripts/build_v4_queries.py --offline          # 用已有 huatuo_sample.jsonl
    python scripts/build_v4_queries.py --skip-leakage     # 跳过泄露检查（调试用）

说明：datasets-server 的 /filter 与 /search 端点对本数据集返回 500，
因此抽样用 /rows 等距分页取候选，再在本地按科室白名单、问题长度与
PII 规则确定性筛选，不下载数据集文件（单文件 138 MB，超过 100 MB 上限）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
V3_QUERIES = ROOT / "examples" / "public_health_v3" / "queries.jsonl"
AUTHORED = ROOT / "examples" / "public_health_v4" / "queries_authored.jsonl"
OUT_QUERIES = ROOT / "examples" / "public_health_v4" / "queries.jsonl"
OUT_HUATUO = ROOT / "examples" / "public_health_v4" / "huatuo_sample.jsonl"
PARSED_DOCS = ROOT / ".cache" / "corpus" / "parsed" / "documents.jsonl"

sys.path.insert(0, str(ROOT / "src"))
from medidiag.rag.leakage import load_jsonl_records, run_leakage_check_records  # noqa: E402

# ---------------------------------------------------------------- Huatuo 抽样

HUATUO_DATASET = "FreedomIntelligence/Huatuo26M-Lite"
HUATUO_LICENSE = "apache-2.0"
HUATUO_REVISION = "90ce61699e90568db82cdce4c4035c6918d70aa6"  # main，2026-10-27 经 API 核实
HUATUO_TOTAL_ROWS = 177_703
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={HUATUO_DATASET}&config=default&split=train"
    "&offset={offset}&length={length}"
)

# 与 v4 语料主题相关的科室白名单（探查得到的 label 全集见数据卡）
HUATUO_LABEL_ALLOW = ("内科", "儿科", "感染与免疫科", "急诊医学科")
# 抽样参数：等距页起点 + 固定种子
HUATUO_PAGE_STRIDE = 4_100
HUATUO_SEED = 20261027
HUATUO_SAMPLE_N = 30
# 问题长度下限：过短的追问缺少上下文，不适合单轮评测
HUATUO_MIN_QUESTION_CHARS = 25

# 个人身份信息（PII）模式：手机号、身份证、邮箱、病历号/住院号
PII_PATTERNS = [
    re.compile(r"1[3-9]\d{9}"),  # 手机号
    re.compile(r"\d{17}[\dXx]"),  # 身份证号
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # 邮箱
    re.compile(r"(病历号|住院号|门诊号|患者编号)[:：]?\s*\w+"),
]


def _read_all(resp) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = resp.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_rows(opener, offset: int, length: int = 100, retries: int = 6) -> list[dict]:
    import time

    req = urllib.request.Request(
        ROWS_URL.format(offset=offset, length=length),
        headers={"User-Agent": "medcite-m7-v4-queries"},
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with opener.open(req, timeout=180) as resp:
                return json.loads(_read_all(resp))["rows"]
        except Exception as exc:  # noqa: BLE001 网络抖动与限速重试
            last_err = exc
            time.sleep(min(30, 2 ** attempt * 2))
    raise RuntimeError(f"rows 端点 offset={offset} 多次失败: {last_err}")


def _contains_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def sample_huatuo(proxy: str | None, sample_n: int = HUATUO_SAMPLE_N) -> list[dict[str, Any]]:
    """确定性抽样 Huatuo26M-Lite 问题。

    规则（可复现）：
    1. /rows 端点按 ``HUATUO_PAGE_STRIDE`` 等距取页（每页 100 行）覆盖整个数据集；
    2. 过滤：科室在白名单内、问题不少于 ``HUATUO_MIN_QUESTION_CHARS`` 字、
       不含 PII 模式、related_diseases 非空（排除纯情绪倾诉类）；
    3. 候选按 (sha1(id), id) 排序后，用固定种子 ``HUATUO_SEED`` 抽 ``sample_n`` 条；
    4. 只保留 ``id`` 与 ``question``，不保存答案。

    只经 API 分页取行，不下载数据集文件（138 MB，超过 100 MB 上限）。
    """
    handlers: list = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)

    candidates: list[dict[str, Any]] = []
    offset = 0
    seen_ids: set[int] = set()
    while offset < HUATUO_TOTAL_ROWS:
        for item in _fetch_rows(opener, offset):
            row = item["row"]
            q = (row.get("question") or "").strip()
            label = (row.get("label") or "").strip()
            if label not in HUATUO_LABEL_ALLOW:
                continue
            if len(q) < HUATUO_MIN_QUESTION_CHARS:
                continue
            if not (row.get("related_diseases") or "").strip():
                continue
            if _contains_pii(q):
                continue
            rid = int(row["id"])
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            digest = hashlib.sha1(str(rid).encode()).hexdigest()
            candidates.append({"id": rid, "question": q, "label": label, "_digest": digest})
        offset += HUATUO_PAGE_STRIDE
        if offset < HUATUO_TOTAL_ROWS:
            import time

            time.sleep(1.5)  # 限速：与采集约定一致，页间间隔不少于 1.5 秒

    candidates.sort(key=lambda c: (c["_digest"], c["id"]))
    rng = random.Random(HUATUO_SEED)
    picked = rng.sample(candidates, sample_n) if len(candidates) > sample_n else candidates
    picked.sort(key=lambda c: c["id"])
    print(f"Huatuo 候选 {len(candidates)} 条，抽样 {len(picked)} 条")
    return picked


# ---------------------------------------------------------------- 字段补全

INTENT_BY_BUCKET = {
    "direct": "popular_science",
    "paraphrase": "popular_science",
    "multi": "popular_science",
    "uncovered": "popular_science",  # 普通健康科普问题，语料未覆盖
    "near_miss": "popular_science",
    "out_of_scope": "out_of_scope",
}


def prepare_v3(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """v3 查询原样保留 sample_id 与 question，补统一字段。"""
    out = []
    for r in records:
        out.append(
            {
                "sample_id": r["sample_id"],
                "split": r["split"],  # 暂存原划分，统一分层划分时会被重排字段覆盖
                "bucket": r["bucket"],
                "question": r["question"],
                "gold_evidence_ids": list(r.get("gold_evidence_ids", [])),
                "rationale": r.get("rationale", ""),
                "origin": "v3",
                "intent": INTENT_BY_BUCKET.get(r["bucket"], "popular_science"),
                "expected_filters": {},
                "answerable": None,
            }
        )
    return out


def prepare_authored(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        out.append(
            {
                "sample_id": r["sample_id"],
                "split": None,
                "bucket": r["bucket"],
                "question": r["question"],
                "gold_evidence_ids": [],
                "rationale": r.get("rationale", ""),
                "origin": "authored_v4",
                "intent": r["intent"],
                "expected_filters": dict(r.get("expected_filters") or {}),
                "answerable": None,
            }
        )
    return out


def prepare_huatuo(picked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Huatuo 真实问题均为个体病情咨询，路由标签记为 individual_care。"""
    out = []
    for row in picked:
        out.append(
            {
                "sample_id": f"v4_h{row['id']}",
                "split": None,
                "bucket": "individual_care",
                "question": row["question"],
                "gold_evidence_ids": [],
                "rationale": f"Huatuo26M-Lite id={row['id']}（{row['label']}），真实个体咨询，测路由与拒答。",
                "origin": "huatuo",
                "intent": "individual_care",
                "expected_filters": {},
                "answerable": None,
            }
        )
    return out


# ---------------------------------------------------------------- 划分

SPLIT_SEED = 20261027
DEV_N = 60


def stratified_split(records: list[dict[str, Any]]) -> None:
    """按 origin × intent 分层、固定种子抽取 dev 60 条，其余为 test。

    分层内按 sample_id 排序后用固定种子抽样，保证可复现。
    就地更新 ``split`` 字段。
    """
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        strata.setdefault((r["origin"], r["intent"]), []).append(r)
    for members in strata.values():
        members.sort(key=lambda r: r["sample_id"])

    rng = random.Random(SPLIT_SEED)
    total = len(records)
    dev_n = min(DEV_N, total)  # 小样本（如单元测试）时 dev 不超过总量
    # 先按比例取整分配，再按小数余数补足
    allocation: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    assigned = 0
    for key, members in strata.items():
        exact = dev_n * len(members) / total
        base = min(int(exact), len(members))
        allocation[key] = base
        assigned += base
        remainders.append((exact - base, key))
    remainders.sort(key=lambda x: (-x[0], x[1]))
    i = 0
    while assigned < dev_n and i < len(remainders):
        key = remainders[i][1]
        if allocation[key] < len(strata[key]):
            allocation[key] += 1
            assigned += 1
        i += 1

    for key, members in strata.items():
        n = allocation[key]
        chosen = set(rng.sample(range(len(members)), n)) if n < len(members) else set(range(len(members)))
        for idx, r in enumerate(members):
            r["split"] = "dev" if idx in chosen else "test"


# ---------------------------------------------------------------- 泄露检查


def doc_to_chunks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """把解析后的文档摊平为泄露检查可读的 text 记录（与分块无关，用元素级文本）。"""
    records = []
    seq = 0
    for s_i, section in enumerate(doc.get("sections", [])):
        for e in section.get("elements", []):
            for value in [e.get("text", ""), *e.get("items", [])]:
                if value:
                    records.append({"chunk_id": f"{doc['doc_id']}:{s_i}:{seq}", "text": str(value)})
                    seq += 1
            for row in e.get("rows", []):
                records.append(
                    {
                        "chunk_id": f"{doc['doc_id']}:{s_i}:{seq}",
                        "text": " ".join(str(c) for c in row),
                    }
                )
                seq += 1
    return records


def leakage_check(eval_records: list[dict[str, Any]]) -> list[str]:
    docs = load_jsonl_records(PARSED_DOCS)
    kb_records: list[dict[str, Any]] = []
    for doc in docs:
        kb_records.extend(doc_to_chunks(doc))
    return run_leakage_check_records(
        eval_records,
        kb_records,
        {
            "chunk_fields_to_check": [],
            "check_question_text": True,
            "check_answer_key": False,
        },
    )


# ---------------------------------------------------------------- 主流程


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="不联网，使用已有 huatuo_sample.jsonl")
    parser.add_argument("--skip-leakage", action="store_true", help="跳过泄露检查")
    parser.add_argument("--proxy", default=None, help="覆盖 Hugging Face 访问代理（默认读系统代理）")
    args = parser.parse_args()

    v3 = prepare_v3(load_jsonl_records(V3_QUERIES))
    authored = prepare_authored(load_jsonl_records(AUTHORED))

    if args.offline and OUT_HUATUO.exists():
        huatuo_rows = load_jsonl_records(OUT_HUATUO)
        picked = [{"id": int(r["sample_id"].removeprefix("v4_h")), "question": r["question"], "label": r.get("huatuo_label", "")} for r in huatuo_rows]
    else:
        proxy = args.proxy
        if proxy is None:
            import winreg

            try:
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
                )
                enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
                server, _ = winreg.QueryValueEx(key, "ProxyServer")
                proxy = f"http://{server}" if enabled else None
            except OSError:
                proxy = None
        picked = sample_huatuo(proxy)
        write_jsonl(
            OUT_HUATUO,
            [
                {
                    "sample_id": f"v4_h{r['id']}",
                    "question": r["question"],
                    "huatuo_label": r["label"],
                    "source": "FreedomIntelligence/Huatuo26M-Lite",
                    "source_url": "https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite",
                    "revision": HUATUO_REVISION,
                    "license": HUATUO_LICENSE,
                }
                for r in picked
            ],
        )
    huatuo = prepare_huatuo(picked)

    all_records = v3 + authored + huatuo

    if not args.skip_leakage:
        hits = leakage_check(all_records)
        if hits:
            print("泄露检查命中：")
            for h in hits:
                print(" ", h)
            raise SystemExit(1)
        print("泄露检查通过（新查询与解析后全文无 >50 字逐字重合）。")

    stratified_split(all_records)

    dist = Counter((r["origin"], r["intent"], r["split"]) for r in all_records)
    print("origin × intent × split 分布：")
    for (origin, intent, split), n in sorted(dist.items()):
        print(f"  {origin:12s} {intent:16s} {split:4s} {n}")
    print("bucket 分布：", dict(Counter(r["bucket"] for r in all_records)))
    print("dev:", sum(1 for r in all_records if r["split"] == "dev"), "test:", sum(1 for r in all_records if r["split"] == "test"))

    write_jsonl(OUT_QUERIES, all_records)
    print(f"已写入 {OUT_QUERIES}（{len(all_records)} 条，阶段 A 草案，未冻结）")


if __name__ == "__main__":
    main()
