"""scripts/build_v4_queries.py 的离线测试。

不联网、不调用模型：用内置的小样本验证
- 分层划分确定性（固定种子两次结果一致、dev 条数正确）；
- v3 查询原样保留 sample_id 与 question；
- 输出字段完整；
- PII 过滤与 Huatuo 候选筛选规则。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "build_v4_queries", ROOT / "scripts" / "build_v4_queries.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules["build_v4_queries"] = mod
SPEC.loader.exec_module(mod)


def _fake_v3_records() -> list[dict]:
    return [
        {
            "sample_id": f"public_q{i:02d}",
            "split": "dev" if i % 2 else "test",
            "bucket": "direct" if i % 3 else "out_of_scope",
            "question": f"测试问题{i}？",
            "gold_evidence_ids": [f"kb_{i:02d}"],
            "rationale": f"理由{i}",
            "origin": "v1",
        }
        for i in range(1, 21)
    ]


def _fake_authored() -> list[dict]:
    return [
        {
            "sample_id": f"v4_a{i:02d}",
            "question": f"新写问题{i}？",
            "intent": ["popular_science", "emergency", "individual_care", "out_of_scope"][i % 4],
            "expected_filters": {"period": "2025-03"} if i % 5 == 0 else {},
            "bucket": ["multi_part", "numeric", "temporal", "emergency", "individual_care"][i % 5],
            "rationale": f"新写理由{i}",
        }
        for i in range(1, 11)
    ]


def _fake_huatuo() -> list[dict]:
    return [
        {"id": 1000 + i, "question": f"华陀问题{i}" + "长" * 30, "label": "内科"}
        for i in range(1, 7)
    ]


def _assemble() -> list[dict]:
    records = (
        mod.prepare_v3(_fake_v3_records())
        + mod.prepare_authored(_fake_authored())
        + mod.prepare_huatuo(_fake_huatuo())
    )
    return records


def test_split_deterministic_and_dev_count():
    r1 = _assemble()
    r2 = _assemble()
    mod.stratified_split(r1)
    mod.stratified_split(r2)
    assert [r["split"] for r in r1] == [r["split"] for r in r2]
    assert sum(1 for r in r1 if r["split"] == "dev") == min(mod.DEV_N, len(r1))
    assert all(r["split"] in ("dev", "test") for r in r1)


def test_split_stratified_covers_origins():
    records = _assemble()
    mod.stratified_split(records)
    dev = [r for r in records if r["split"] == "dev"]
    # 每个 origin 在 dev 中都应有代表（小样本下至少有 v3 与 huatuo）
    dev_origins = {r["origin"] for r in dev}
    assert "v3" in dev_origins
    assert "huatuo" in dev_origins


def test_v3_preserved_verbatim():
    src = _fake_v3_records()
    out = mod.prepare_v3(src)
    for original, prepared in zip(src, out, strict=True):
        assert prepared["sample_id"] == original["sample_id"]
        assert prepared["question"] == original["question"]
        assert prepared["origin"] == "v3"
        assert prepared["answerable"] is None
        assert isinstance(prepared["expected_filters"], dict)
        assert prepared["gold_evidence_ids"] == original["gold_evidence_ids"]


def test_output_fields_complete():
    records = _assemble()
    mod.stratified_split(records)
    required = {
        "sample_id", "split", "bucket", "question", "gold_evidence_ids",
        "rationale", "origin", "intent", "expected_filters", "answerable",
    }
    for r in records:
        assert required <= set(r), f"缺字段: {required - set(r)}"
        assert r["intent"] in {
            "popular_science", "individual_care", "out_of_scope", "emergency",
        }
        assert r["origin"] in {"v3", "authored_v4", "huatuo"}


def test_intent_mapping_v3():
    out = mod.prepare_v3(_fake_v3_records())
    by_bucket = {r["bucket"]: r["intent"] for r in out}
    assert by_bucket["out_of_scope"] == "out_of_scope"
    assert by_bucket["direct"] == "popular_science"


def test_huatuo_intent_is_individual_care():
    out = mod.prepare_huatuo(_fake_huatuo())
    assert all(r["intent"] == "individual_care" for r in out)
    assert all(r["origin"] == "huatuo" for r in out)
    assert all(r["sample_id"].startswith("v4_h") for r in out)


def test_pii_patterns():
    assert mod._contains_pii("我的手机号是13812345678，最近头疼")
    assert mod._contains_pii("身份证号 110101199003077777")
    assert mod._contains_pii("邮箱 someone@example.com 联系")
    assert not mod._contains_pii("感冒发烧三天了怎么办")


def test_huatuo_candidate_rules(monkeypatch):
    rows = []
    for i in range(10):
        rows.append(
            {
                "row": {
                    "id": 100 + i,
                    "question": "长" * 40 + f"变体{i}",
                    "label": "内科" if i % 2 == 0 else "皮肤性病科",
                    "related_diseases": "感冒" if i != 3 else "",
                }
            }
        )

    def fake_fetch(opener, offset, length=100, retries=6):
        return rows if offset == 0 else []

    monkeypatch.setattr(mod, "_fetch_rows", fake_fetch)
    monkeypatch.setattr(mod, "HUATUO_TOTAL_ROWS", 1)
    picked = mod.sample_huatuo(proxy=None, sample_n=3)
    assert len(picked) == 3
    # 只有内科且 related_diseases 非空的行才入选（i 为偶数且 i != 3 → 0,2,4,6,8）
    assert all(p["label"] == "内科" for p in picked)
