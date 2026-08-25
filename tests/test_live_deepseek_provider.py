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
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return httpx.Response(200, json=payload, headers={"x-request-id": request_id}, request=request)


def test_llm_client_uses_official_openai_endpoint_and_preserves_request_id() -> None:
    captured: dict = {}

    def fake_post(url: str, **kwargs) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return _response({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "total_tokens": 105,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
            },
        })

    client = LLMClient(
        api_key="test-key",
        base_url="https://api.deepseek.com/",
        model="demo-model",
        post=fake_post,
    )
    completion = client.complete("draft", system_prompt="system")

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "demo-model"
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["messages"][0] == {"role": "system", "content": "system"}
    assert completion.content == "hello"
    assert completion.request_id == "req-live-1"
    assert completion.usage["prompt_cache_hit_tokens"] == 80
    assert completion.usage["prompt_cache_miss_tokens"] == 20
    assert completion.usage["prompt_cache_hit_rate"] == 800000
    assert client.usage_summary()["prompt_cache_hit_tokens"] == 80
    assert client.usage_summary()["prompt_cache_hit_rate"] == 800000


def test_llm_client_retries_transient_connection_error() -> None:
    """临时网络错误按退避策略重试，并保留请求标识。"""
    calls = 0
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError("temporary failure", request=request)
        return _response({"id": "chatcmpl-retry-id", "choices": [{"message": {"content": "hello"}}]})

    completion = LLMClient(
        api_key="test-key",
        max_retries=1,
        retry_backoff_seconds=0.25,
        post=fake_post,
        sleep=sleeps.append,
    ).complete("draft")

    assert calls == 2
    assert sleeps == [0.25]
    assert completion.request_id == "req-live-1"


def test_llm_client_uses_response_body_id_when_headers_are_absent() -> None:
    """响应头缺失时使用响应体 id 作为调用标识。"""
    def fake_post(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-body-id",
                "choices": [{"message": {"content": "hello"}}],
            },
            request=request,
        )

    completion = LLMClient(api_key="test-key", post=fake_post).complete("draft")

    assert completion.request_id == "chatcmpl-body-id"


def test_llm_client_accepts_missing_request_id_outside_formal_provenance() -> None:
    """开发模式可保留成功响应，即使供应商未提供调用标识。"""

    def fake_post(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
            request=request,
        )

    completion = LLMClient(api_key="test-key", post=fake_post).complete("draft")

    assert completion.request_id is None


def test_llm_client_retries_missing_request_id_when_formal_provenance_requires_it() -> None:
    """正式 response-id 模式仅接受具有真实调用标识的成功响应。"""

    calls = 0
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs) -> httpx.Response:
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", url)
        payload = {"choices": [{"message": {"content": "hello"}}]}
        if calls == 2:
            payload["id"] = "chatcmpl-second-response"
        return httpx.Response(200, json=payload, request=request)

    completion = LLMClient(
        api_key="test-key",
        require_request_id=True,
        max_retries=1,
        retry_backoff_seconds=0.25,
        post=fake_post,
        sleep=sleeps.append,
    ).complete("draft")

    assert calls == 2
    assert sleeps == [0.25]
    assert completion.request_id == "chatcmpl-second-response"


def test_llm_client_rejects_missing_request_id_after_formal_retries() -> None:
    """正式 response-id 模式不能把无调用标识的成功响应写入评测结果。"""

    calls = 0

    def fake_post(url: str, **kwargs) -> httpx.Response:
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello"}}]},
            request=request,
        )

    client = LLMClient(
        api_key="test-key",
        require_request_id=True,
        max_retries=1,
        retry_backoff_seconds=0,
        post=fake_post,
        sleep=lambda _: None,
    )

    with pytest.raises(MediDiagError, match="PROVIDER_RESPONSE_ID_MISSING") as exc_info:
        client.complete("draft")

    assert calls == 2
    assert exc_info.value.code == "PROVIDER_RESPONSE_ID_MISSING"



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
    assert outcome.metadata["generation_model"] == "demo-model"
    assert outcome.metadata["prompt_cache_hit_tokens"] == 0
    assert outcome.payload["agents"][0]["agent_name"] == "live_evidence_drafter"
    assert outcome.payload["claims"][0]["citation_chunk_ids"] == ["live_demo_evidence"]
    review = provider.review(
        outcome.payload, provider.arbitrate(outcome.payload, retrieval), retrieval
    )
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
