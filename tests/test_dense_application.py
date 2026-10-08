"""向量检索方案与“输出不合格后补救一次”的离线回归：用假的向量编码器和假的传输，不下载模型、不联网。"""
import copy
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from medidiag.errors import MediDiagError
from medidiag.llm.openai_compatible import OpenAICompatibleProvider
from medidiag.llm.profiles import get_provider_profile
from medidiag.rag.dense import BGE_ZH_QUERY_PREFIX, DenseRetriever
from medidiag.schemas import KnowledgeChunk
from medidiag.workflow.application import load_application_config, retrieval_profile
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.xiaomimimo.com/v1/chat/completions"


def _chunk(chunk_id: str, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=chunk_id, source="测试来源", source_id=chunk_id, text=text)


def _encoder(seen: list[str]):
    table = {"甲": [1.0, 0.0], "乙": [0.0, 1.0], "丙": [0.0, 1.0]}

    def encode(texts: list[str]):
        seen.extend(texts)
        return np.array([table[t[-1]] for t in texts])
    return encode


def test_dense_retriever_ranks_by_cosine_with_prefix_and_stable_ties() -> None:
    seen: list[str] = []
    retriever = DenseRetriever([_chunk("c1", "甲"), _chunk("c3", "丙"), _chunk("c2", "乙")],
                               encoder=_encoder(seen), model_name="m", model_revision="r")
    retriever.build_index()
    results = retriever.search("问乙", top_k=3)
    assert [r.chunk_id for r in results] == ["c2", "c3", "c1"]  # 同分按 chunk_id 排序
    assert results[0].embedding_score == pytest.approx(1.0) and results[2].embedding_score == pytest.approx(0.0)
    assert seen[-1] == BGE_ZH_QUERY_PREFIX + "问乙"  # 查询带检索前缀，文档不带
    assert len(retriever.search("问甲", top_k=2)) == 2


def test_dense_retriever_requires_built_index() -> None:
    retriever = DenseRetriever([_chunk("c1", "甲")], encoder=_encoder([]), model_name="m", model_revision="r")
    with pytest.raises(MediDiagError, match="RAG_INDEX_BUILD_FAILED"):
        retriever.search("甲")


def test_dense_profile_is_accepted_and_bm25_profile_unchanged() -> None:
    assert retrieval_profile(load_application_config(ROOT / "configs/application_dense.yaml")) == "dense"
    assert retrieval_profile(load_application_config(ROOT / "configs/application.yaml")) == "bm25"


@pytest.mark.parametrize("change", ["floating_revision", "downloadable", "other_model", "mixed_weights",
                                    "bm25_too", "rerank", "gate", "bad_on_invalid"])
def test_unsafe_dense_config_rejected(tmp_path, change) -> None:
    config = copy.deepcopy(load_application_config(ROOT / "configs/application_dense.yaml"))
    flags = config["experiments"]["rag"]["rag_full"]["config"]
    if change == "floating_revision":
        config["embedding"]["revision"] = "main"
    elif change == "downloadable":
        config["embedding"]["local_files_only"] = False
    elif change == "other_model":
        config["embedding"]["model"] = "someone/else"
    elif change == "mixed_weights":
        config["retrieval"]["weights"]["w1_bm25"] = 0.5
    elif change == "bm25_too":
        flags["use_bm25"] = True
    elif change == "rerank":
        flags["use_rerank"] = True
    elif change == "gate":
        config["leakage_check"]["check_question_text"] = False
    else:
        config["generation"]["on_invalid"] = "retry_forever"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_application_config(path)


# ---- 输出不合格后的补救：最多多问一次 ----

class _FakeRag:
    chunks = [_chunk("c1", "完整原文一")]


def _reply(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"id": "rid", "model": "mimo-v2.5", "usage": {"total_tokens": 3},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]})


def _provider(monkeypatch, replies, on_invalid):
    monkeypatch.setenv("MIMO_API_KEY", "test-placeholder-not-a-real-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.5")
    bodies: list[dict] = []

    def post(url, **kwargs):
        bodies.append(kwargs["json"])
        return replies[min(len(bodies), len(replies)) - 1]
    llm = OpenAICompatibleProvider(get_provider_profile("mimo_v25"), post=post, max_retries=0, max_structured_retries=0)
    return MimoGroundedWorkflowProvider(_FakeRag(), llm=llm, on_invalid=on_invalid), bodies  # type: ignore[arg-type]


BAD = {"status": "sufficient", "claims": [{"text": "改写过的原文", "citation_chunk_ids": ["c1"]}]}
GOOD = {"status": "sufficient", "claims": [{"text": "完整原文一", "citation_chunk_ids": ["c1"]}]}
RETRIEVAL = {"chunks": [{"chunk_id": "c1", "text": "完整原文一"}]}


@pytest.mark.parametrize("mode,extra_message", [("resample", False), ("feedback", True)])
def test_invalid_output_is_retried_once(monkeypatch, mode, extra_message) -> None:
    provider, bodies = _provider(monkeypatch, [_reply(BAD), _reply(GOOD)], mode)
    result = provider.generate("问题", RETRIEVAL, {})
    assert result.payload["claims"][0]["text"] == "完整原文一"
    assert len(bodies) == 2
    assert (len(bodies[1]["messages"]) == 3) is extra_message
    if extra_message:  # 告知上次被拒原因，但不重复上次的错误输出
        assert "摘录不是本轮完整原文" in bodies[1]["messages"][2]["content"]


@pytest.mark.parametrize("mode", ["none", "resample", "feedback"])
def test_never_more_than_one_extra_call(monkeypatch, mode) -> None:
    provider, bodies = _provider(monkeypatch, [_reply(BAD)], mode)
    with pytest.raises(MediDiagError, match="STRUCTURED_OUTPUT_INVALID"):
        provider.generate("问题", RETRIEVAL, {})
    assert len(bodies) == (1 if mode == "none" else 2)


def test_only_output_contract_errors_are_retried(monkeypatch) -> None:
    provider, bodies = _provider(monkeypatch, [httpx.Response(429, json={})], "feedback")
    with pytest.raises(MediDiagError):
        provider.generate("问题", RETRIEVAL, {})
    assert len(bodies) == 1
