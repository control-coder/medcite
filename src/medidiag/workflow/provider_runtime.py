"""Reliable execution boundary for external workflow providers."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from medidiag.errors import MediDiagError, get_error_spec


class _StagePayload(BaseModel):
    model_config = ConfigDict(extra="allow")


class _NormalizePayload(_StagePayload):
    normalized_query: str = Field(min_length=1)


class _RetrievalPayload(_StagePayload):
    query: str
    chunks: list[dict[str, Any]]


class _PlanPayload(_StagePayload):
    objective: str = Field(min_length=1)


class _GenerationPayload(_StagePayload):
    agents: list[dict[str, Any]]
    claims: list[dict[str, Any]]


class _ArbitrationPayload(_StagePayload):
    verdict: str = Field(min_length=1)


class _ReviewPayload(_StagePayload):
    verdict: Literal["APPROVED", "REVISION_REQUIRED", "ESCALATED"]
    citation_verdicts: list[dict[str, Any]]
    compliance_status: str = Field(min_length=1)


class _ReportPayload(_StagePayload):
    case_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    disclaimer: str = Field(min_length=1)


_STAGE_SCHEMAS: dict[str, type[_StagePayload]] = {
    "normalize": _NormalizePayload,
    "retrieval": _RetrievalPayload,
    "plan": _PlanPayload,
    "generation": _GenerationPayload,
    "arbitration": _ArbitrationPayload,
    "review": _ReviewPayload,
    "report": _ReportPayload,
}

_TIMEOUT_CODES = {
    "normalize": "RAG_TIMEOUT",
    "retrieval": "RAG_TIMEOUT",
    "review": "JUDGE_TIMEOUT",
}


@dataclass(frozen=True)
class ProviderResponse:
    """Provider payload plus transport metadata safe for audit logs."""

    payload: dict[str, Any]
    request_id: str | None = None


@dataclass(frozen=True)
class ProviderAttempt:
    stage: str
    attempt: int
    latency_ms: int
    request_id: str | None
    error_code: str | None
    retryable: bool
    retry_decision: Literal["not_needed", "retry", "exhausted", "rejected"]
    http_status: int | None = None

    def to_event_detail(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provider_attempt": self.attempt,
            "latency_ms": self.latency_ms,
            "provider_request_id": self.request_id,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "retry_decision": self.retry_decision,
            "http_status": self.http_status,
        }


@dataclass(frozen=True)
class ProviderCallOutcome:
    payload: dict[str, Any]
    request_id: str | None
    attempts: tuple[ProviderAttempt, ...]
    elapsed_ms: int

    @property
    def retry_count(self) -> int:
        return max(0, len(self.attempts) - 1)


class ProviderCallRunner:
    """Execute one provider stage with bounded, auditable retries."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        backoff_seconds: Sequence[float] = (0.05, 0.1),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if len(backoff_seconds) < max_attempts - 1:
            raise ValueError("backoff_seconds must cover every retry")
        self.max_attempts = max_attempts
        self.backoff_seconds = tuple(backoff_seconds)
        self.sleep = sleep

    def call(
        self,
        stage: str,
        operation: Callable[[], dict[str, Any] | ProviderResponse],
        *,
        on_attempt: Callable[[ProviderAttempt], None] | None = None,
    ) -> ProviderCallOutcome:
        if stage not in _STAGE_SCHEMAS:
            raise ValueError(f"unsupported provider stage: {stage}")

        call_started = time.perf_counter()
        attempts: list[ProviderAttempt] = []
        for number in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                raw = operation()
                response = raw if isinstance(raw, ProviderResponse) else ProviderResponse(raw)
                payload = self._validate(stage, response.payload)
            except (MediDiagError, httpx.TimeoutException, httpx.HTTPStatusError, ValidationError) as exc:
                error_code, request_id, http_status = self._classify(stage, exc)
                spec = get_error_spec(error_code)
                will_retry = spec.retryable and number < self.max_attempts
                decision: Literal["retry", "exhausted", "rejected"]
                if will_retry:
                    decision = "retry"
                elif spec.retryable:
                    decision = "exhausted"
                else:
                    decision = "rejected"
                attempt = ProviderAttempt(
                    stage=stage,
                    attempt=number,
                    latency_ms=self._latency_ms(started),
                    request_id=request_id,
                    error_code=error_code,
                    retryable=spec.retryable,
                    retry_decision=decision,
                    http_status=http_status,
                )
                attempts.append(attempt)
                if on_attempt:
                    on_attempt(attempt)
                if will_retry:
                    self.sleep(self.backoff_seconds[number - 1])
                    continue
                raise MediDiagError(
                    error_code,
                    detail=f"provider stage {stage} failed on attempt {number}",
                    context={"provider_attempt": attempt.to_event_detail()},
                ) from exc

            attempt = ProviderAttempt(
                stage=stage,
                attempt=number,
                latency_ms=self._latency_ms(started),
                request_id=response.request_id,
                error_code=None,
                retryable=False,
                retry_decision="not_needed",
            )
            attempts.append(attempt)
            if on_attempt:
                on_attempt(attempt)
            return ProviderCallOutcome(
                payload,
                response.request_id,
                tuple(attempts),
                self._latency_ms(call_started),
            )

        raise AssertionError("provider retry loop exited unexpectedly")

    @staticmethod
    def _validate(stage: str, payload: Any) -> dict[str, Any]:
        model = _STAGE_SCHEMAS[stage].model_validate(payload)
        return model.model_dump(mode="json")

    @staticmethod
    def _classify(
        stage: str,
        exc: MediDiagError | httpx.TimeoutException | httpx.HTTPStatusError | ValidationError,
    ) -> tuple[str, str | None, int | None]:
        if isinstance(exc, MediDiagError):
            return exc.code, None, None
        if isinstance(exc, ValidationError):
            return "PROVIDER_SCHEMA_INVALID", None, None
        if isinstance(exc, httpx.TimeoutException):
            return _TIMEOUT_CODES.get(stage, "LLM_TIMEOUT"), None, None

        response = exc.response
        request_id = next(
            (
                response.headers[name]
                for name in ("x-request-id", "request-id", "x-correlation-id")
                if name in response.headers
            ),
            None,
        )
        if response.status_code == 429:
            return "PROVIDER_RATE_LIMITED", request_id, response.status_code
        if 500 <= response.status_code < 600:
            return "PROVIDER_UNAVAILABLE", request_id, response.status_code
        return "PROVIDER_REQUEST_REJECTED", request_id, response.status_code

    @staticmethod
    def _latency_ms(started: float) -> int:
        return max(0, round((time.perf_counter() - started) * 1000))
