"""Unit coverage for the live DeepSeek-compatible minimal demo adapter."""

from __future__ import annotations

import json

import httpx
import pytest

from medidiag.agents.llm_client import LLMClient
from medidiag.errors import MediDiagError
from medidiag.workflow.deepseek_provider import DeepSeekWorkflowProvider
from medidiag.workflow.provider_runtime import ProviderCallRunner


def _response(payload: dict, *, request_id: str = "req-live-1") -> httpx.Response:
    request = httpx.Request("POST", "https://www.dogapi.cc/v1/chat/completions")
    return httpx.Response(200, json=payload, headers={"x-request-id": request_id}, request=request)


def test_llm_client_uses_dogapi_openai_endpoint_and_preserves_request_id() -> None:
    captured: dict = {}

    def fake_post(url: str, **kwargs) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return _response({"choices": [{"message": {"content": "hello"}}]})

    client = LLMClient(
        api_key="test-key",
        base_url="https://www.dogapi.cc/v1/",
        model="demo-model",
        post=fake_post,
    )
    completion = client.complete("draft", system_prompt="system")

    assert captured["url"] == "https://www.dogapi.cc/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "demo-model"
    assert captured["json"]["messages"][0] == {"role": "system", "content": "system"}
    assert completion.content == "hello"
    assert completion.request_id == "req-live-1"


def test_live_provider_generates_schema_bound_draft_and_audits_request_id() -> None:
    def fake_post(url: str, **kwargs) -> httpx.Response:
        prompt = kwargs["json"]["messages"][-1]["content"]
        assert "live_demo_evidence" in prompt
        content = json.dumps(
            {
                "claims": [
                    {
                        "text": "现有演示证据不足以形成确定性医学结论，需由合格临床人员复核。",
                        "citation_chunk_ids": ["live_demo_evidence"],
                        "confidence": 0.2,
                    }
                ],
                "uncertainty": "本输出受本地演示证据限制，不构成医疗建议。",
                "risk_flags": ["qualified_clinician_review_required"],
            }
        )
        return _response({"choices": [{"message": {"content": content}}]})

    provider = DeepSeekWorkflowProvider(
        LLMClient(api_key="test-key", model="demo-model", post=fake_post)
    )
    retrieval = {
        "chunks": [
            {
                "chunk_id": "live_demo_evidence",
                "source": "fixture",
                "text": "Fixture evidence for a non-diagnostic engineering demonstration.",
            }
        ]
    }
    outcome = ProviderCallRunner(sleep=lambda _: None).call(
        "generation",
        lambda: provider.generate("Deidentified demo input.", retrieval, provider.plan("q", retrieval)),
    )

    assert outcome.request_id == "req-live-1"
    assert outcome.payload["agents"][0]["agent_name"] == "live_evidence_drafter"
    assert outcome.payload["claims"][0]["citation_chunk_ids"] == ["live_demo_evidence"]
    review = provider.review(outcome.payload, provider.arbitrate(outcome.payload, retrieval))
    assert review["verdict"] == "APPROVED"
    assert review["citation_verdicts"][0]["method"] == "demo_structure_binding_not_nli"


def test_live_provider_rejects_unknown_citation_id() -> None:
    def fake_post(url: str, **kwargs) -> httpx.Response:
        content = json.dumps(
            {
                "claims": [
                    {
                        "text": "该断言不应通过未知的引用标识符。",
                        "citation_chunk_ids": ["unknown"],
                        "confidence": 0.1,
                    }
                ],
                "uncertainty": "不确定。",
                "risk_flags": [],
            }
        )
        return _response({"choices": [{"message": {"content": content}}]})

    provider = DeepSeekWorkflowProvider(LLMClient(api_key="test-key", post=fake_post))
    retrieval = {
        "chunks": [
            {
                "chunk_id": "allowed",
                "source": "fixture",
                "text": "Local demonstration fixture.",
            }
        ]
    }
    with pytest.raises(MediDiagError, match="LLM_JSON_INVALID"):
        provider.generate("demo", retrieval, provider.plan("demo", retrieval))
