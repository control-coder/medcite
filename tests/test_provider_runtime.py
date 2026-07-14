"""Provider timeout, HTTP error, retry, and schema boundary tests."""

from __future__ import annotations

import httpx
import pytest

from medidiag.errors import MediDiagError
from medidiag.workflow.provider_runtime import ProviderCallRunner, ProviderResponse


def _http_error(status: int, request_id: str = "req-test") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.invalid/v1/call")
    response = httpx.Response(
        status, request=request, headers={"x-request-id": request_id}
    )
    return httpx.HTTPStatusError("provider failure", request=request, response=response)


def test_rate_limit_retries_and_preserves_request_ids() -> None:
    calls = 0
    sleeps: list[float] = []
    audit = []

    def operation():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _http_error(429, f"req-rate-{calls}")
        return ProviderResponse(
            {"objective": "use retrieved evidence first"},
            request_id="req-success",
        )

    outcome = ProviderCallRunner(sleep=sleeps.append).call(
        "plan", operation, on_attempt=audit.append
    )

    assert calls == 3
    assert sleeps == [0.05, 0.1]
    assert outcome.request_id == "req-success"
    assert outcome.retry_count == 2
    assert [item.retry_decision for item in audit] == [
        "retry", "retry", "not_needed"
    ]
    assert [item.request_id for item in audit] == [
        "req-rate-1", "req-rate-2", "req-success"
    ]


@pytest.mark.parametrize(
    ("stage", "expected_code"),
    [
        ("retrieval", "RAG_TIMEOUT"),
        ("generation", "LLM_TIMEOUT"),
        ("review", "JUDGE_TIMEOUT"),
    ],
)
def test_timeout_mapping_exhausts_bounded_retries(
    stage: str, expected_code: str
) -> None:
    calls = 0
    audit = []

    def operation():
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    runner = ProviderCallRunner(max_attempts=2, backoff_seconds=(0,), sleep=lambda _: None)
    with pytest.raises(MediDiagError) as caught:
        runner.call(stage, operation, on_attempt=audit.append)

    assert caught.value.code == expected_code
    assert calls == 2
    assert [item.retry_decision for item in audit] == ["retry", "exhausted"]


def test_transient_5xx_retries_then_exhausts() -> None:
    audit = []
    runner = ProviderCallRunner(max_attempts=2, backoff_seconds=(0,), sleep=lambda _: None)

    def operation():
        raise _http_error(503, "req-down")

    with pytest.raises(MediDiagError) as caught:
        runner.call("generation", operation, on_attempt=audit.append)

    assert caught.value.code == "PROVIDER_UNAVAILABLE"
    assert audit[-1].http_status == 503
    assert audit[-1].request_id == "req-down"
    assert audit[-1].retry_decision == "exhausted"


def test_invalid_schema_is_not_retried() -> None:
    calls = 0
    audit = []

    def operation():
        nonlocal calls
        calls += 1
        return {"unexpected": "shape"}

    with pytest.raises(MediDiagError) as caught:
        ProviderCallRunner(sleep=lambda _: None).call(
            "review", operation, on_attempt=audit.append
        )

    assert caught.value.code == "PROVIDER_SCHEMA_INVALID"
    assert calls == 1
    assert audit[0].retryable is False
    assert audit[0].retry_decision == "rejected"


def test_non_retryable_http_error_is_rejected_immediately() -> None:
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise _http_error(400, "req-invalid")

    with pytest.raises(MediDiagError) as caught:
        ProviderCallRunner(sleep=lambda _: None).call("plan", operation)

    assert caught.value.code == "PROVIDER_REQUEST_REJECTED"
    assert calls == 1
