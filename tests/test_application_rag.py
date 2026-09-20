"""应用检索的轻量回归和指标定义，不替代历史研究评测。"""

from scripts.verify_rag import make_retriever, retrieval_metrics, run_regression


def test_hit_is_not_multi_relevant_recall():
    assert retrieval_metrics(["a"], ["a", "b"], 1) == {"hit": 1, "recall": 0.5}
    assert retrieval_metrics([], [], 3) == {"hit": None, "recall": None}


def test_lexical_mode_never_builds_embedding(monkeypatch):
    retriever = make_retriever()

    def forbidden(*args):
        raise AssertionError("词法模式不得隐式下载或调用向量模型")

    monkeypatch.setattr(retriever, "_get_embedding_scores", forbidden)
    config = {"use_bm25": True, "use_embedding": False, "use_term_normalization": True,
              "use_evidence_weighting": False, "use_rerank": False}
    assert retriever.search("不存在的主题", experiment_config=config) == []
    assert retriever.search("pyrexia", experiment_config=config)[0].chunk_id == "sim_01"


def test_fixed_fixture_and_reference_ids():
    report = run_regression()
    assert report["query_count"] == 20
    known = {chunk.chunk_id for chunk in make_retriever().chunks}
    for run in report["runs"].values():
        for query in run["queries"]:
            assert set(query["retrieved_ids"]) <= known
            assert set(query["relevant_chunk_ids"]) <= known
    optimized = report["runs"]["bm25_term_normalized"]
    assert "q_20" in optimized["miss_ids_at_3"]
    assert optimized["queries"][18]["retrieved_ids"] == []
