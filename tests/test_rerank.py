"""交叉编码器重排的离线回归：用假打分函数替代模型，不读模型缓存、不联网。"""
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from medidiag.errors import MediDiagError
from medidiag.llm.cassette import Cassette, ReplayTransport, request_key
from medidiag.llm.models import ACTIVE_MIMO_MODEL
from medidiag.rag.rerank import RERANK_MODEL, Reranker, load_local_reranker
from medidiag.rag.runtime import RuntimeMedicalRAG
from medidiag.workflow.application import MAX_RERANK_CANDIDATES, load_application_config
from medidiag.workflow.retrieval_agent import (
    RetrievalAgentWorkflowProvider,
    merge_hits,
    run_search_agent,
)
from tests.test_retrieval_agent import ScriptedLLM, _hits, _reply

ROOT = Path(__file__).resolve().parents[1]
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
BASE_CONFIG = ROOT / "configs/application_agent.yaml"  # v2 语料 178 段，BM25 能给出 10 个以上候选
QUESTION = "如何预防感染"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str


def by_text(scores: dict[str, float]):
    """按片段原文查表打分；记录每次收到的（问题，片段）对。"""
    seen: list[list[tuple[str, str]]] = []

    def score(pairs: list[tuple[str, str]]) -> list[float]:
        seen.append(pairs)
        return [scores.get(text, 0.0) for _, text in pairs]

    score.seen = seen  # type: ignore[attr-defined]
    return score


def reranker(scores: dict[str, float], candidates: int = 10) -> Reranker:
    return Reranker(by_text(scores), model_name=RERANK_MODEL, model_revision=REVISION, candidates=candidates)


# ----------------------------------------------------------------- Reranker


def test_rerank_orders_by_score_dedupes_and_breaks_ties_by_chunk_id() -> None:
    items = [Hit("c", "丙"), Hit("a", "甲"), Hit("b", "乙"), Hit("a", "甲（第二次检索）")]
    ranked = reranker({"甲": 0.1, "乙": 0.9, "丙": 0.9}).rerank("问题", items, top_n=3)
    assert [(h.chunk_id, s) for h, s in ranked] == [("b", 0.9), ("c", 0.9), ("a", 0.1)]
    assert ranked[2][0].text == "甲"  # 重复的片段只打一次分，保留先出现的那一个


def test_rerank_scores_each_unique_candidate_once_against_the_question() -> None:
    scorer = by_text({})
    Reranker(scorer, model_name=RERANK_MODEL, model_revision=REVISION, candidates=5).rerank(
        "问题", [Hit("a", "甲"), Hit("a", "甲"), Hit("b", "乙")], top_n=1)
    assert scorer.seen == [[("问题", "甲"), ("问题", "乙")]]  # type: ignore[attr-defined]


def test_rerank_without_candidates_does_not_call_the_model() -> None:
    scorer = by_text({})
    assert Reranker(scorer, model_name=RERANK_MODEL, model_revision=REVISION, candidates=5).rerank("问题", [], 3) == []
    assert scorer.seen == []  # type: ignore[attr-defined]


def test_rerank_rejects_a_score_count_that_does_not_match_the_candidates() -> None:
    bad = Reranker(lambda pairs: [1.0], model_name=RERANK_MODEL, model_revision=REVISION, candidates=5)
    with pytest.raises(MediDiagError):
        bad.rerank("问题", [Hit("a", "甲"), Hit("b", "乙")], 2)


def test_local_reranker_never_downloads(monkeypatch) -> None:
    import sentence_transformers

    calls = []

    def fake_cross_encoder(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("not in cache")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_cross_encoder)
    with pytest.raises(MediDiagError, match="不会自动下载"):
        load_local_reranker(RERANK_MODEL, REVISION)
    assert calls[0]["local_files_only"] is True and calls[0]["revision"] == REVISION


# ----------------------------------------------------------------- 运行时 RAG


def _rerank_config(candidates: int = 10) -> dict:
    config = copy.deepcopy(load_application_config(BASE_CONFIG))
    config["retrieval"]["rerank"] = {"model": RERANK_MODEL, "revision": REVISION, "local_files_only": True,
                                     "candidates": candidates}
    return config


def _rag_with(scores: dict[str, float], candidates: int = 10) -> RuntimeMedicalRAG:
    return RuntimeMedicalRAG.from_config(_rerank_config(candidates), root=ROOT,
                                         reranker_factory=lambda config: reranker(scores, candidates))


def test_rerank_config_without_a_factory_is_refused() -> None:
    with pytest.raises(MediDiagError, match="重排"):
        RuntimeMedicalRAG.from_config(_rerank_config(), root=ROOT)


def test_fixed_search_takes_more_candidates_and_keeps_the_reranked_top_k() -> None:
    plain = RuntimeMedicalRAG.from_config(load_application_config(BASE_CONFIG), root=ROOT)
    question = QUESTION
    candidates = plain.search_evidence(question, top_k=10)
    assert len(candidates) == 10
    last = candidates[-1]  # BM25 排在第 10 的片段，重排后应排到第一
    rag = _rag_with({last.text: 5.0})
    top = rag.search_evidence(question)
    assert len(top) == int(rag.retrieval_config["top_k"]) and top[0].chunk_id == last.chunk_id
    assert top[0].final_score == 5.0
    assert [i.chunk_id for i in rag.search_evidence(question, top_k=10)] == [i.chunk_id for i in candidates]  # 指定深度时不重排


def test_frozen_bundle_records_which_reranker_ordered_the_evidence() -> None:
    rag = _rag_with({})
    result = rag.freeze_evidence(QUESTION, rag.search_evidence(QUESTION))
    assert result["evidence_bundle"]["retrieval_provenance"]["reranker"] == {
        "model": RERANK_MODEL, "revision": REVISION, "candidates": 10}
    plain = RuntimeMedicalRAG.from_config(load_application_config(BASE_CONFIG), root=ROOT)
    bundle = plain.freeze_evidence(QUESTION, plain.search_evidence(QUESTION))["evidence_bundle"]
    assert "reranker" not in bundle["retrieval_provenance"]


# ----------------------------------------------------------------- 应用配置


@pytest.mark.parametrize("change", [
    {"model": "BAAI/bge-reranker-base"},
    {"revision": "main"},
    {"local_files_only": False},
    {"candidates": 2},  # 少于最终段数 top_k=3
    {"candidates": MAX_RERANK_CANDIDATES + 1},
    {"candidates": True},
    {"candidates": 10.0},
    {"extra": 1},
])
def test_rerank_config_accepts_only_the_pinned_local_model(tmp_path, change) -> None:
    config = _rerank_config()
    config["retrieval"]["rerank"].update(change)
    path = tmp_path / "rerank.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError):
        load_application_config(path)


def test_rerank_config_with_pinned_model_is_accepted(tmp_path) -> None:
    path = tmp_path / "rerank.yaml"
    path.write_text(yaml.safe_dump(_rerank_config(), allow_unicode=True), encoding="utf-8")
    assert load_application_config(path)["retrieval"]["rerank"]["candidates"] == 10


# ----------------------------------------------------------------- 检索 Agent


def test_agent_sees_only_observe_chunks_while_merge_uses_all_candidates() -> None:
    llm = ScriptedLLM(_reply("规范说法"), _reply())
    results = {"规范说法": _hits("x", "y", "z", "w")}
    merged_from: list[list[str]] = []

    def merge(lists):
        merged_from.append([h.chunk_id for hits in lists for h in hits])
        return [h for hits in reversed(lists) for h in hits]

    outcome = run_search_agent(llm, lambda q: results.get(q, []), "屋里太闷", _hits("a", "b", "c", "d"),
                               ACTIVE_MIMO_MODEL, observe=3, merge=merge)
    first = json.loads(llm.requests[0].messages[1]["content"])["first_search"]
    assert [o["chunk_id"] for o in first] == ["a", "b", "c"]
    assert [o["chunk_id"] for o in json.loads(llm.requests[1].messages[-1]["content"])] == ["x", "y", "z"]
    assert merged_from == [["a", "b", "c", "d", "x", "y", "z", "w"]]
    assert [h.chunk_id for h in outcome.chunks] == ["x", "y", "z"]
    assert outcome.steps[-1] == {"action": "merge", "method": "rerank", "candidates": 8}
    assert [s["chunk_ids"] for s in outcome.steps if s["action"] == "search"] == [["a", "b", "c"], ["x", "y", "z"]]


def test_agent_requests_with_observe_match_requests_without_extra_candidates() -> None:
    """多取候选只影响合并：模型收到的请求与只取 3 段时逐字相同，已有调用记录可直接复用。"""
    plain, deep = ScriptedLLM(_reply("规范说法"), _reply()), ScriptedLLM(_reply("规范说法"), _reply())
    run_search_agent(plain, lambda q: _hits("x", "y", "z"), "屋里太闷", _hits("a", "b", "c"), ACTIVE_MIMO_MODEL)
    run_search_agent(deep, lambda q: _hits("x", "y", "z", "w", "v"), "屋里太闷", _hits("a", "b", "c", "d"),
                     ACTIVE_MIMO_MODEL, observe=3, merge=lambda lists: list(lists[0]))
    assert [r.messages for r in plain.requests] == [r.messages for r in deep.requests]


def test_merge_failure_falls_back_to_rrf_and_is_recorded() -> None:
    def broken(lists):
        raise MediDiagError("RAG_INDEX_BUILD_FAILED")

    outcome = run_search_agent(ScriptedLLM(_reply()), lambda q: [], "屋里太闷", _hits("a", "b", "c"),
                               ACTIVE_MIMO_MODEL, merge=broken)
    assert outcome.chunks == merge_hits([_hits("a", "b", "c")]) and outcome.status == "ok"
    assert outcome.steps[-1] == {"action": "merge", "method": "rrf", "reason": "rerank_failed",
                                 "code": "RAG_INDEX_BUILD_FAILED"}


def test_agent_provider_reranks_the_union_against_the_original_question() -> None:
    plain = RuntimeMedicalRAG.from_config(load_application_config(BASE_CONFIG), root=ROOT)
    question = QUESTION
    tenth = plain.search_evidence(question, top_k=10)[-1]
    rag = _rag_with({tenth.text: 5.0})
    provider = RetrievalAgentWorkflowProvider(rag, llm=ScriptedLLM(_reply()))  # type: ignore[arg-type]
    result = provider.retrieve(question)
    assert result["chunks"][0]["chunk_id"] == tenth.chunk_id  # 模型没看到第 10 段，重排仍能把它排上来
    steps = result["search_agent"]["steps"]
    assert len(steps[0]["chunk_ids"]) == 3 and steps[-1]["method"] == "rerank"
    assert result["evidence_bundle"]["retrieval_provenance"]["reranker"]["candidates"] == 10


# ----------------------------------------------------------------- 调用记录


def test_cassette_reads_the_base_but_writes_only_its_own_file(tmp_path) -> None:
    body = {"model": "mimo-v2.5", "messages": [{"role": "user", "content": "旧"}], "max_tokens": 8}
    old = tmp_path / "old.jsonl"
    Cassette(old).put(request_key(body), {"id": "r1", "choices": [{"message": {"content": "旧回复"}}]}, 1)
    before = old.read_bytes()
    new = Cassette(tmp_path / "new.jsonl", base=Cassette(old))
    assert new.get(request_key(body))["body"]["id"] == "r1" and len(new) == 0
    new.put(request_key(body, 1), {"id": "r2", "choices": [{"message": {"content": "新回复"}}]}, 1)
    assert old.read_bytes() == before and len(new) == 1
    response = ReplayTransport(new)("https://api.xiaomimimo.com/v1/chat/completions", headers={}, json=body, timeout=5)
    assert response.json()["id"] == "r1"
