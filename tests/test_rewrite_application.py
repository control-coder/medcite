"""BM25 + 查询改写方案的离线回归：假的传输，不联网。"""
import copy
import json
from pathlib import Path

import httpx
import pytest
import yaml

from medidiag.llm.openai_compatible import OpenAICompatibleProvider
from medidiag.llm.profiles import get_provider_profile
from medidiag.workflow.application import load_application_config, retrieval_profile
from medidiag.workflow.mimo_grounded import MimoGroundedWorkflowProvider
from medidiag.workflow.query_rewrite import MAX_REWRITE_CHARS, parse_rewrite

ROOT = Path(__file__).resolve().parents[1]


class _Rag:
    chunks: list = []

    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> dict:
        self.queries.append(query)
        return {"query": query, "chunks": [], "top_k": 0, "config_hash": "h", "corpus_hash": "c"}


def _reply(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"id": "rid", "model": "mimo-v2.5", "usage": {"total_tokens": 3},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}]})


def _provider(monkeypatch, replies, rewrite=True):
    monkeypatch.setenv("MIMO_API_KEY", "test-placeholder-not-a-real-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    monkeypatch.setenv("MIMO_MODEL", "mimo-v2.5")
    bodies: list[dict] = []

    def post(url, **kwargs):
        bodies.append(kwargs["json"])
        return replies[min(len(bodies), len(replies)) - 1]
    llm = OpenAICompatibleProvider(get_provider_profile("mimo_v25"), post=post, max_retries=0, max_structured_retries=0)
    rag = _Rag()
    return MimoGroundedWorkflowProvider(rag, llm=llm, rewrite=rewrite), rag, bodies  # type: ignore[arg-type]


def test_rewrite_is_appended_to_the_original_question(monkeypatch) -> None:
    provider, rag, bodies = _provider(monkeypatch, [_reply({"query": "通风与改善室内空气质量"})])
    result = provider.retrieve("屋里太闷怎样透透气？")
    assert rag.queries == ["屋里太闷怎样透透气？ 通风与改善室内空气质量"]
    assert result["query_rewrite"]["status"] == "ok" and len(bodies) == 1
    assert result["execution_mode"] == "mimo_grounded"


@pytest.mark.parametrize("reply", [httpx.Response(429, json={}), _reply({"query": ""}),
                                   _reply({"query": "长" * (MAX_REWRITE_CHARS + 1)}), _reply({"other": "x"})])
def test_failed_rewrite_falls_back_to_original_and_says_so(monkeypatch, reply) -> None:
    provider, rag, bodies = _provider(monkeypatch, [reply])
    result = provider.retrieve("原问题")
    assert rag.queries == ["原问题"]
    assert result["query_rewrite"]["status"] == "fallback" and result["query_rewrite"]["code"]
    assert len(bodies) == 1  # 改写失败不重试


def test_rewrite_disabled_makes_no_model_call(monkeypatch) -> None:
    provider, rag, bodies = _provider(monkeypatch, [_reply({"query": "不该被调用"})], rewrite=False)
    result = provider.retrieve("原问题")
    assert rag.queries == ["原问题"] and bodies == [] and "query_rewrite" not in result


def test_parse_rewrite_strips_and_rejects_bad_shapes() -> None:
    assert parse_rewrite({"query": "  规范表述 "}) == "规范表述"
    assert parse_rewrite(None) is None and parse_rewrite({"query": 3}) is None


def test_rewrite_profile_is_bm25_and_unsafe_combinations_rejected(tmp_path) -> None:
    config = load_application_config(ROOT / "configs/application_rewrite.yaml")
    assert retrieval_profile(config) == "bm25" and config["generation"]["rewrite"] is True
    dense = copy.deepcopy(load_application_config(ROOT / "configs/application_dense.yaml"))
    dense["generation"]["rewrite"] = True  # 向量检索下改写没有收益，不接受
    bad_type = copy.deepcopy(config)
    bad_type["generation"]["rewrite"] = "yes"
    for name, content in (("dense", dense), ("type", bad_type)):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
        with pytest.raises(ValueError):
            load_application_config(path)
