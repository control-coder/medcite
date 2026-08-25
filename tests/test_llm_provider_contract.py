"""LLM Provider Layer 的离线 contract tests。"""

from __future__ import annotations

import json

import httpx
import pytest

from medidiag.errors import MediDiagError
from medidiag.llm import FakeProvider, LLMRequest, OpenAICompatibleProvider, get_provider_profile
from medidiag.workflow.openai_provider import OpenAICompatibleWorkflowProvider
from medidiag.workflow.provider import DeterministicWorkflowProvider


def _response(
    payload: dict,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", "https://provider.invalid/chat/completions")
    return httpx.Response(status, json=payload, headers=headers, request=request)


def test_fake_provider_is_deterministic_and_network_free() -> None:
    provider = FakeProvider()
    request = LLMRequest(
        messages=[{"role": "user", "content": "公开脱敏输入"}],
        response_format={"type": "json_object"},
        prompt_version="test-v1",
    )

    first = provider.generate(request, timeout_s=1, idempotency_key="same-key")
    second = provider.generate(request, timeout_s=1, idempotency_key="same-key")

    assert first == second
    assert first.response_id == second.response_id
    assert first.response_id is not None and first.response_id.startswith("fake_")
    assert first.provenance_mode == "system_fingerprint"
    assert first.parsed_json == {
        "summary": "离线确定性 Provider 仅用于工程测试，不构成医疗建议。",
        "prompt_version": "test-v1",
    }


def test_mimo_profile_filters_unsupported_sampling_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://mimo.invalid/v1")
    captured: dict = {}

    def fake_post(url: str, **kwargs) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return _response(
            {
                "id": "mimo-response-1",
                "model": "mimo-v2.5",
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": '{"ok": true}'}],
                            "reasoning_content": "不写入用户报告的推理字段",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            }
        )

    provider = OpenAICompatibleProvider(
        get_provider_profile("mimo_v25"),
        post=fake_post,
        max_retries=0,
    )
    result = provider.generate(
        LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
            temperature=0.3,
            top_p=0.8,
            reasoning_mode="enabled",
        ),
        timeout_s=3,
        idempotency_key="case:stage:1",
    )

    assert captured["url"] == "https://mimo.invalid/v1/chat/completions"
    assert captured["headers"]["Idempotency-Key"] == "case:stage:1"
    assert "temperature" not in captured["json"]
    assert "top_p" not in captured["json"]
    assert captured["json"]["thinking"] == {"type": "enabled"}
    assert result.filtered_parameters == ("temperature", "top_p")
    assert result.parsed_json == {"ok": True}
    assert result.reasoning_content == "不写入用户报告的推理字段"
    assert result.usage == {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12}
    assert result.provenance_mode == "provider_response_id"


def test_deepseek_profile_preserves_fingerprint_tool_calls_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_post(url: str, **kwargs) -> httpx.Response:
        return _response(
            {
                "id": "chatcmpl-1",
                "model": "deepseek-test",
                "system_fingerprint": "fp-1",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 3},
            }
        )

    provider = OpenAICompatibleProvider(get_provider_profile("deepseek_default"), post=fake_post)
    result = provider.generate(
        LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice="auto",
        ),
        timeout_s=3,
        idempotency_key="tool-call",
    )

    assert result.tool_calls[0]["id"] == "call-1"
    assert result.system_fingerprint == "fp-1"
    assert result.provenance_mode == "system_fingerprint"
    assert result.usage["prompt_cache_hit_tokens"] == 5


def test_openai_compatible_provider_retries_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = 0
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(
                {"error": {"message": "rate limited"}},
                status=429,
                headers={"Retry-After": "1.5"},
            )
        return _response({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    provider = OpenAICompatibleProvider(
        get_provider_profile("deepseek_default"),
        post=fake_post,
        sleep=sleeps.append,
        max_retries=1,
        retry_backoff_seconds=0.25,
    )
    result = provider.generate(
        LLMRequest(messages=[{"role": "user", "content": "test"}]),
        timeout_s=3,
        idempotency_key="retry",
    )

    assert calls == 2
    assert sleeps == [1.5]
    assert result.retry_count == 1


@pytest.mark.parametrize(
    ("status", "body", "expected_code"),
    [
        (401, {"error": {"message": "invalid api key"}}, "PROVIDER_AUTH_FAILED"),
        (400, {"error": {"message": "maximum context length exceeded"}}, "PROVIDER_CONTEXT_LIMIT"),
        (422, {"error": {"message": "content policy blocked"}}, "PROVIDER_CONTENT_BLOCKED"),
        (403, {"error": {"message": "safety policy blocked"}}, "PROVIDER_CONTENT_BLOCKED"),
    ],
)
def test_openai_compatible_provider_maps_non_retryable_errors(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: dict,
    expected_code: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = OpenAICompatibleProvider(
        get_provider_profile("deepseek_default"),
        post=lambda *args, **kwargs: _response(body, status=status),
        max_retries=0,
    )

    with pytest.raises(MediDiagError) as caught:
        provider.generate(
            LLMRequest(messages=[{"role": "user", "content": "test"}]),
            timeout_s=3,
            idempotency_key="error",
        )

    assert caught.value.code == expected_code


def test_missing_api_key_maps_to_provider_auth_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(get_provider_profile("deepseek_default"), max_retries=0)

    with pytest.raises(MediDiagError) as caught:
        provider.generate(
            LLMRequest(messages=[{"role": "user", "content": "test"}]),
            timeout_s=3,
            idempotency_key="missing-key",
        )

    assert caught.value.code == "PROVIDER_AUTH_FAILED"


@pytest.mark.parametrize(
    ("transport_error", "expected_code"),
    [
        (httpx.ReadTimeout("读取超时"), "LLM_TIMEOUT"),
        (httpx.ConnectError("连接失败"), "PROVIDER_NETWORK_ERROR"),
    ],
)
def test_transport_errors_are_mapped_after_retry_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: httpx.RequestError,
    expected_code: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = 0

    def failing_post(url: str, **kwargs) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise transport_error

    provider = OpenAICompatibleProvider(
        get_provider_profile("deepseek_default"),
        post=failing_post,
        sleep=lambda _: None,
        max_retries=1,
    )

    with pytest.raises(MediDiagError) as caught:
        provider.generate(
            LLMRequest(messages=[{"role": "user", "content": "test"}]),
            timeout_s=3,
            idempotency_key="transport-error",
        )

    assert calls == 2
    assert caught.value.code == expected_code


def test_profile_defaults_are_applied_to_request_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    captured: dict = {}

    def fake_post(url: str, **kwargs) -> httpx.Response:
        captured.update(kwargs)
        return _response({"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        get_provider_profile("deepseek_default"),
        post=fake_post,
        max_retries=0,
    )
    provider.generate(
        LLMRequest(messages=[{"role": "user", "content": "test"}]),
        timeout_s=3,
        idempotency_key="profile-defaults",
    )

    assert captured["json"]["temperature"] == 0
    assert captured["json"]["max_tokens"] == 1200
    assert captured["json"]["thinking"] == {"type": "disabled"}


def test_structured_output_invalid_is_not_coerced_to_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    provider = OpenAICompatibleProvider(
        get_provider_profile("deepseek_default"),
        post=lambda *args, **kwargs: _response(
            {"choices": [{"message": {"content": "not-json"}}]}
        ),
        max_retries=0,
    )

    with pytest.raises(MediDiagError) as caught:
        provider.generate(
            LLMRequest(
                messages=[{"role": "user", "content": "test"}],
                response_format={"type": "json_object"},
            ),
            timeout_s=3,
            idempotency_key="json",
        )

    assert caught.value.code == "STRUCTURED_OUTPUT_INVALID"


def test_workflow_assembly_switches_profiles_without_agent_code_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_workflow = DeterministicWorkflowProvider()
    assert fake_workflow.version == "deterministic-provider-v1"

    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    monkeypatch.setenv("MIMO_BASE_URL", "https://mimo.invalid/v1")
    workflow = OpenAICompatibleWorkflowProvider(
        OpenAICompatibleProvider(
            get_provider_profile("mimo_v25"),
            post=lambda *args, **kwargs: _response(
                {
                    "id": "mimo-workflow-1",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "claims": [
                                            {
                                                "text": "现有演示证据不足以形成确定性医学结论。",
                                                "citation_chunk_ids": ["allowed"],
                                                "confidence": 0.2,
                                            }
                                        ],
                                        "uncertainty": "仅为工程演示，需由专业人员复核。",
                                        "risk_flags": [],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                }
            ),
        )
    )
    retrieval = {"chunks": [{"chunk_id": "allowed", "source": "fixture", "text": "demo"}]}
    generated = workflow.generate("脱敏输入", retrieval, workflow.plan("q", retrieval))

    assert generated.metadata["provider_profile"] == "mimo_v25"
    assert generated.request_id == "mimo-workflow-1"
