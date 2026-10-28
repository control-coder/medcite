"""M13 指标纯函数与评审检查样本构造的离线用例。"""

from __future__ import annotations

import pytest

from eval.entailment_check import (
    change_number,
    delete_negation,
    insert_negation,
    production_flag,
    punct_variant,
    rules_catch,
)
from eval.generation_quality import (
    EVIDENCE_REF,
    build_items,
    citation_metrics,
    citation_relevance,
    context_precision,
    faithfulness,
    mean_cosine,
    parse_nli,
    plan_batches,
)

E, P, N = "entailed", "partial", "not_entailed"


def test_citation_relevance_follows_alce_rules() -> None:
    assert citation_relevance(E, [E]) == [True]
    assert citation_relevance(N, [N]) == [False]
    assert citation_relevance(P, [E, E]) == [False, False]  # 句未被蕴含：引用都不计相关
    assert citation_relevance(E, [E, N]) == [True, False]  # 第二条单独不蕴含，去掉后第一条仍蕴含
    assert citation_relevance(E, [N, N]) == [True, True]  # 只有合起来才蕴含：都相关
    assert citation_relevance(E, [E, E]) == [True, True]


def test_citation_metrics_skip_unjudged_sentences() -> None:
    sentences = [{"all": E, "alone": [E, N]}, {"all": N, "alone": [N]}, {"all": None, "alone": [None]}]
    result = citation_metrics(sentences)
    assert result["n_sentences"] == 2
    assert result["recall"]["count"] == "1/2"
    assert result["precision"]["count"] == "1/3"  # 3 条引用里只有第一句的第一条相关


def test_faithfulness_counts_entailed_share() -> None:
    result = faithfulness([E, E, P, N, None])
    assert result["faithfulness"]["count"] == "2/4"
    assert result["partial_share"]["count"] == "1/4"


def test_context_precision_weights_by_rank() -> None:
    assert context_precision(["a", "b", "c"], {"a"}) == pytest.approx(1.0)
    assert context_precision(["x", "a", "y"], {"a"}) == pytest.approx(0.5)
    assert context_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx((1 + 2 / 3) / 2)
    assert context_precision(["x", "y"], {"a"}) == 0.0
    assert context_precision([], {"a"}) == 0.0


def test_mean_cosine() -> None:
    assert mean_cosine([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(0.5)


def test_build_items_and_batch_split() -> None:
    sentences = [{"text": "甲", "citations": [{"quote": "q1"}, {"quote": "q2"}]}, {"text": "乙", "citations": [{"quote": "q3"}]}]
    items = build_items(sentences)
    assert [i["kind"] for i in items] == ["all", "alone", "alone", "faith", "all", "faith"]
    assert [i["id"] for i in items] == list(range(6))
    assert sum(i["premise"] == EVIDENCE_REF for i in items) == 2
    assert len(plan_batches({"c": "证据"}, items, "m")) == 1
    long_evidence = {"c": "证" * 11500}  # 超出输入上限：引用整份证据的项与短 quote 项拆开
    batches = plan_batches(long_evidence, items, "m")
    assert len(batches) >= 2
    assert sorted(i["id"] for _, group in batches for i in group) == list(range(6))


def test_parse_nli_requires_exact_ids_and_labels() -> None:
    ok = {"verdicts": [{"id": 0, "label": E}, {"id": 1, "label": N}]}
    assert parse_nli(ok, [0, 1]) == {0: E, 1: N}
    assert parse_nli(ok, [0]) is None
    assert parse_nli({"verdicts": [{"id": 0, "label": "yes"}]}, [0]) is None
    assert parse_nli({"verdicts": [{"id": 0, "label": E}, {"id": 0, "label": E}]}, [0]) is None


def test_constructed_transforms() -> None:
    assert change_number("成人每天应摄入5克盐") == "成人每天应摄入11克盐"
    assert change_number("胰岛素分泌不足") is None
    assert change_number("增加0.5倍") == "增加2.0倍"
    assert insert_negation("儿童可以接种该疫苗") == "儿童不可以接种该疫苗"
    assert insert_negation("儿童不可以接种该疫苗") is None
    assert delete_negation("不要让儿童留在车内") == "要让儿童留在车内"
    assert delete_negation("儿童可以接种") is None
    changed = punct_variant("发热时，应多饮水；必要时就医。")
    assert changed is not None and "，" not in changed and not changed.endswith("。")
    assert punct_variant("没有标点的句子") is None


def test_production_flag_and_rules() -> None:
    assert production_flag("该药不能与酒同服", "该药可与酒同服使用")
    assert not production_flag("该药不能与酒同服", "该药不能与酒同服使用")
    chunk = "盐摄入量应少于5克以降低血压风险。"
    pair = {"sentence": "盐摄入量应少于11克", "quote": "盐摄入量应少于5克以降低血压风险", "quote_chunk_id": "c1"}
    assert rules_catch(pair, chunk)  # 数值 11 不在 quote 中，确定性校验会拦下
    assert not rules_catch({**pair, "sentence": "盐摄入量应少于5克"}, chunk)
