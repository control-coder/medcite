"""M7 v4 句子级标注的离线测试：提示词组装与编号校验、输出校验、句子到块的投影、一致率。

不依赖网络、模型与付费调用；池与检索部分用小型内存语料构造。
"""

from __future__ import annotations

import json
import types
from pathlib import Path

from eval import v4_judgments as vj

DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "public_health_v4"


def _passage(doc_id: str, chunk_id: str, text: str, title: str = "标题") -> vj.PassageState:
    return vj.PassageState(doc_id=doc_id, chunk_id=chunk_id, title=title, text=text)


# ----------------------------------------------------------------- 池与编号

def _pool(*passages: tuple[str, str, str]) -> vj.QueryPool:
    pool = vj.QueryPool("q1")
    for rank, (doc_id, chunk_id, text) in enumerate(passages, 1):
        pool.add(_passage(doc_id, chunk_id, text), "parent_child_dense", rank, "pool")
    vj.assign_requests(pool)
    return pool


def test_pool_dedups_by_chunk_id_and_keeps_same_doc_chunks() -> None:
    pool = vj.QueryPool("q1")
    pool.add(_passage("d1", "d1:0:pc:0", "句子一。"), "parent_child_dense", 1, "pool")
    pool.add(_passage("d1", "d1:1:pc:0", "句子二。"), "parent_child_bm25", 2, "pool")  # 同文档另一块
    pool.add(_passage("d1", "d1:0:pc:0", "句子一。"), "gold_parent_child", 0, "gold_pc")  # 同块再次命中
    assert [p.chunk_id for p in pool.passages] == ["d1:0:pc:0", "d1:1:pc:0"]
    assert [s["via"] for s in pool.sources["d1:0:pc:0"]] == ["pool", "gold_pc"]
    assert [s["method"] for s in pool.sources["d1:1:pc:0"]] == ["parent_child_bm25"]


def test_duplicate_sentences_are_sent_once_with_sequential_ids() -> None:
    pool = _pool(("d1", "d1:0:rc:0", "流感由病毒引起。勤洗手可以预防。"),
                 ("d1", "d1:0:pc:0", "勤洗手可以预防。"),  # 与上一块重复的句子
                 ("d2", "d2:0:pc:0", "另一段的句子。"))
    assert [p.ids for p in pool.passages] == [{0: "p1s1", 1: "p1s2"}, {}, {0: "p2s1"}]
    request = vj.build_judge_request(pool, "问题？", 1, model="mimo-v2.5")
    payload = json.loads(request.messages[1]["content"])
    assert payload["question"] == "问题？"
    assert [p["passage"] for p in payload["passages"]] == [1, 2]  # 无新句子的段不送
    assert [s["id"] for p in payload["passages"] for s in p["sentences"]] == ["p1s1", "p1s2", "p2s1"]
    assert vj.valid_sentence_ids(pool, 1) == {"p1s1", "p1s2", "p2s1"}
    assert request.prompt_version == vj.PROMPT_VERSION
    # 重复句子的标签随哈希落到两个块上
    digest = vj.sentence_hash("勤洗手可以预防。")
    assert pool.request_ids[1]["p1s2"] == digest
    rows = vj.judgment_rows(None, {"pool": pool, "labels": {digest: "relevant"}, "responses": []})
    assert {(r["chunk_id"], r["label"]) for r in rows if r["sentence_hash"] == digest} == {
        ("d1:0:rc:0", "relevant")}  # 同 doc 同句只出一行
    assert len(rows) == 3


def test_assign_requests_splits_long_pools_without_breaking_passages() -> None:
    pool = vj.QueryPool("q1")
    pool.add(_passage("d1", "d1:0:rc:0", "".join(f"甲{i}句。" for i in range(600))), "recursive_bm25", 1, "pool")
    pool.add(_passage("d2", "d2:0:rc:0", "".join(f"乙{i}句。" for i in range(600))), "recursive_dense", 1, "pool")
    assert vj.assign_requests(pool) == 2
    # 同一段不拆到两次请求；编号在各请求内从 p1 重新开始
    assert [p.request_no for p in pool.passages] == [1, 2]
    assert [p.ids[0] for p in pool.passages] == ["p1s1", "p1s1"]


# ----------------------------------------------------------------- 输出校验与纠错重试

def test_label_issues_and_correction_request_are_deterministic() -> None:
    valid = {"p1s1", "p1s2"}
    assert vj.label_issues({"relevant": ["p1s1"], "partial": ["p1s2"]}, valid) == []
    issues = vj.label_issues({"relevant": ["p9s9", "p1s1"], "partial": ["p1s1"]}, valid)
    assert issues == ["非法编号：['p9s9']", "重复编号：['p1s1']"]
    assert vj.label_issues("x", valid) == ["输出不是 JSON 对象"]

    pool = _pool(("d1", "d1:0:pc:0", "句子一。句子二。"))
    request = vj.build_judge_request(pool, "问题？", 1)
    retry = vj.build_correction_request(request, issues)
    assert retry.messages[:2] == request.messages
    assert len(retry.messages) == 3 and "p9s9" in retry.messages[2]["content"]
    assert retry.messages == vj.build_correction_request(request, issues).messages  # 同输入同请求体


class _FakeProvider:
    """按调用顺序返回预设输出，记录收到的请求。"""

    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.requests: list[object] = []

    def generate(self, request: object, **_: object) -> object:
        self.requests.append(request)
        parsed = self.outputs.pop(0)
        return types.SimpleNamespace(parsed_json=parsed, usage={"input_tokens": 10, "output_tokens": 2})


def test_invalid_output_retries_once_then_succeeds() -> None:
    pool = _pool(("d1", "d1:0:pc:0", "句子一。句子二。"))
    request = vj.build_judge_request(pool, "问题？", 1)
    provider = _FakeProvider([{"relevant": ["p7s7"]}, {"relevant": ["p1s2"]}])
    result = vj.judge_requests(provider, pool, "q1", [request], workers=1)
    assert result["responses"][0]["status"] == "ok" and result["responses"][0]["retried"] == 1
    assert result["responses"][0]["usage"] == {"input_tokens": 20, "output_tokens": 4}
    assert result["labels"] == {vj.sentence_hash("句子二。"): "relevant"}
    assert len(provider.requests[1].messages) == len(provider.requests[0].messages) + 1


def test_invalid_after_retry_is_judge_failed_and_all_irrelevant() -> None:
    pool = _pool(("d1", "d1:0:pc:0", "句子一。句子二。"))
    request = vj.build_judge_request(pool, "问题？", 1)
    provider = _FakeProvider([{"relevant": ["p7s7"]}, {"partial": ["p8s8"]}])
    result = vj.judge_requests(provider, pool, "q1", [request], workers=1)
    assert result["responses"][0]["status"] == "judge_failed"
    assert result["labels"] == {}
    assert vj.pool_rows(None, result)["judge_failed"] is True
    assert {r["label"] for r in vj.judgment_rows(None, result)} == {"irrelevant"}


# ----------------------------------------------------------------- parse_labels

def test_parse_labels_accepts_valid_output() -> None:
    pool = _pool(("d1", "d1:0:pc:0", "句子一。句子二。"))
    valid = vj.valid_sentence_ids(pool, 1)
    first, second = sorted(valid)
    labels = vj.parse_labels({"relevant": [first], "partial": [second]}, valid)
    assert labels == {first: "relevant", second: "partial"}


def test_parse_labels_rejects_unknown_ids_and_duplicates() -> None:
    valid = {"p1s3", "p8s2"}
    assert vj.parse_labels({"relevant": ["p99s1"]}, valid) is None  # 编号不属于本次请求
    assert vj.parse_labels({"relevant": ["p1s3"], "partial": ["p1s3"]}, valid) is None  # 重复
    assert vj.parse_labels({"relevant": "p1s3"}, valid) is None  # 不是列表
    assert vj.parse_labels({"unknown": ["p1s3"]}, valid) is None  # 未知键
    assert vj.parse_labels({"relevant": [1]}, valid) is None  # 非字符串
    assert vj.parse_labels("not-a-dict", valid) is None


def test_parse_labels_allows_empty_lists() -> None:
    assert vj.parse_labels({"relevant": [], "partial": []}, {"p1s3"}) == {}
    assert vj.parse_labels({}, {"p1s3"}) == {}


# ----------------------------------------------------------------- 句子到块的投影

def test_chunk_relevance_takes_highest_overlapping_label() -> None:
    # 生产路径的 doc_labels 键是「去空白句子文本」（由 sentence_texts 替换哈希后传入）
    doc_labels = {"流感由病毒引起。": "relevant", "多休息。": "partial"}
    chunk = "流感由病毒引起。多休息。多喝水。"
    assert vj.chunk_relevance(chunk, doc_labels) == "relevant"
    # 块 ⊆ 句子（子块只覆盖长句的一部分）
    long_sentence = "流感由病毒引起也可能合并细菌感染"
    doc_labels = {long_sentence: "partial"}
    assert vj.chunk_relevance("流感由病毒引起", doc_labels) == "partial"
    # 空白差异不影响匹配
    assert vj.chunk_relevance("流感由病毒引起 也可能合并 细菌感染", doc_labels) == "partial"
    # 无重叠
    assert vj.chunk_relevance("完全无关的内容", doc_labels) == "irrelevant"


def test_chunk_relevance_matches_whole_chunk_by_hash_without_texts() -> None:
    # 只有 judgments.jsonl（键为 sha256）时：整句块按哈希命中，跨句组合块无法匹配
    text = "流感由病毒引起。"
    digest = vj.sentence_hash(text)
    assert vj.chunk_relevance(text, {digest: "relevant"}) == "relevant"
    assert vj.chunk_relevance("流感由病毒引起。多休息。", {digest: "relevant"}) == "irrelevant"


def test_chunk_labels_for_query_projects_by_doc_id() -> None:
    chunks = [
        {"chunk_id": "d1:0:pc:0", "source_id": "d1", "text": "句子一。",
         "metadata": {"doc_id": "d1"}},
        {"chunk_id": "d2:0:pc:0", "source_id": "d2", "text": "句子一。",
         "metadata": {"doc_id": "d2"}},
    ]
    digest = vj.sentence_hash("句子一。")
    judgments = {"q1": {"d1": {digest: "partial"}}}
    texts = {"q1": {digest: "句子一。"}}
    labels = vj.chunk_labels_for_query("q1", chunks, judgments, texts)
    assert labels == {"d1:0:pc:0": "partial", "d2:0:pc:0": "irrelevant"}


def test_recover_sentence_texts_from_chunks() -> None:
    chunks = {"parent_child": [
        {"chunk_id": "d1:0:pc:0", "source_id": "d1", "text": "句子一。句子二。",
         "metadata": {"doc_id": "d1"}},
    ]}
    digest = vj.sentence_hash("句子二。")
    judgments = {"q1": {"d1": {digest: "relevant"}}}
    texts = vj.recover_sentence_texts(chunks, judgments)
    assert texts == {"q1": {digest: "句子二。"}}


# ----------------------------------------------------------------- 一致率

def test_cohen_kappa_basic() -> None:
    # 完美一致（两个类别都出现）kappa=1；单个类别的完美一致 kappa 无定义（pe=1）
    assert vj.cohen_kappa([(True, True), (False, False)]) == 1.0
    assert vj.cohen_kappa([(True, True), (True, True)]) is None
    # 全部分歧且两组各半：po=0，pe=0.5，kappa=-1
    assert vj.cohen_kappa([(True, False), (False, True)]) == -1.0
    assert vj.cohen_kappa([(True, False)] * 50 + [(False, True)] * 50) == -1.0
    # 多数一致、少数分歧：kappa 介于 0 与 1
    assert 0 < vj.cohen_kappa([(True, True)] * 80 + [(False, False)] * 15 + [(True, False)] * 5) < 1
    assert vj.cohen_kappa([]) is None


def test_query_answerable_uses_sentence_labels() -> None:
    judgments = {"q1": {"d1": {"a" * 64: "partial"}}, "q2": {"d1": {"b" * 64: "relevant"}}}
    assert vj.query_answerable(judgments, "q1") == (False, True)
    assert vj.query_answerable(judgments, "q2") == (True, False)
    assert vj.query_answerable(judgments, "q3") == (False, False)


def test_v3_agreement_only_counts_excerpts_in_pool() -> None:
    chunks = [
        {"chunk_id": "d1:0:pc:0", "source_id": "d1", "source": "t", "text": "甲句。", "metadata": {"doc_id": "d1"}},
        {"chunk_id": "d1:1:pc:0", "source_id": "d1", "source": "t", "text": "乙句。", "metadata": {"doc_id": "d1"}},
        {"chunk_id": "d2:0:pc:0", "source_id": "d2", "source": "t", "text": "丙句。", "metadata": {"doc_id": "d2"}},
    ]
    corpus = types.SimpleNamespace(
        chunks={"parent_child": chunks},
        chunk_by_id={"parent_child": {c["chunk_id"]: c for c in chunks}},
        mapping={"strategies": {"parent_child": {"excerpts": {
            "e_gold": {"mapping": "exact", "gold": ["d1:0:pc:0"]},    # v3 正例，池内，LLM 判正
            "e_neg": {"mapping": "exact", "gold": ["d1:1:pc:0"]},     # 池内负例，LLM 判负
            "e_outside": {"mapping": "exact", "gold": ["d2:0:pc:0"]},  # 池外：不参与
            "e_failed": {"mapping": "failed", "gold": []}}}}},
        query_excerpt_ids=lambda qid: ["e_gold"])
    judgments = {"q1": {"d1": {vj.sentence_hash("甲句。"): "relevant", vj.sentence_hash("乙句。"): "irrelevant"}}}
    texts = {"q1": {vj.sentence_hash("甲句。"): "甲句。", vj.sentence_hash("乙句。"): "乙句。"}}
    result = vj.v3_agreement(corpus, judgments, [{"sample_id": "q1"}], texts)
    assert result["n_pairs"] == 2
    assert result["confusion"] == {"v3_pos_llm_pos": 1, "v3_pos_llm_neg": 0,
                                   "v3_neg_llm_pos": 0, "v3_neg_llm_neg": 1}
    assert result["agreement"] == 1.0 and result["failed_mapping_excerpts"] == 1


# ----------------------------------------------------------------- 真实产物的一致性（离线）

def test_judgments_file_matches_frozen_queries() -> None:
    """已冻结产物自检：judgments.jsonl 投影出的 gold 与 queries.jsonl 一致。"""
    queries_path = DATA_DIR / "queries.jsonl"
    judgments_path = DATA_DIR / "judgments.jsonl"
    if not judgments_path.exists():
        return  # 标注尚未运行时跳过
    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(q["answerable"] is None for q in queries):
        return  # 尚未冻结
    judgments = vj.load_judgments(judgments_path)
    chunks_dir = Path(__file__).resolve().parent.parent / ".cache" / "corpus" / "chunks"
    pc_chunks = [json.loads(line)
                 for line in (chunks_dir / "parent_child" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    all_chunks = {strategy: [json.loads(line)
                             for line in (chunks_dir / strategy / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                             if line.strip()]
                  for strategy in ("recursive", "parent_child")}
    texts = vj.recover_sentence_texts(all_chunks, judgments)
    for query in queries:
        gold = vj.gold_chunks(query["sample_id"], pc_chunks, judgments, texts)
        assert gold == sorted(query["gold_evidence_ids"]), query["sample_id"]
        labels = set(vj.chunk_labels_for_query(query["sample_id"], pc_chunks, judgments, texts).values())
        assert query["answerable"] == ("relevant" in labels)
        assert query["partial_only"] == ("relevant" not in labels and "partial" in labels)
