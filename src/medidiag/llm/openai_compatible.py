"""通用 OpenAI-compatible Chat Completions adapter。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

import httpx

from medidiag.errors import MediDiagError
from medidiag.llm.contracts import LLMRequest, ProviderCapabilities, ProviderResult
from medidiag.llm.profiles import ProviderProfile

PostCallable = Callable[..., httpx.Response]
SleepCallable = Callable[[float], None]


class OpenAICompatibleProvider:
    """把兼容 Chat Completions 的差异归一化为统一 ProviderResult。"""

    def __init__(
        self,
        profile: ProviderProfile,
        *,
        post: PostCallable | None = None,
        sleep: SleepCallable = time.sleep,
        max_retries: int = 2,
        max_structured_retries: int = 1,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if profile.adapter != "openai_compatible":
            raise ValueError(f"profile {profile.profile_id} 不是 openai_compatible adapter")
        if profile.api_style != "chat_completions":
            raise ValueError("一期 OpenAICompatibleProvider 只实现 chat_completions")
        self.profile = profile
        self.provider_id = profile.provider_id
        self.profile_id = profile.profile_id
        self.model = profile.model
        self._post = post or httpx.post
        self._sleep = sleep
        self._max_retries = max(0, max_retries)
        self._max_structured_retries = max(0, min(1, max_structured_retries))
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def capabilities(self) -> ProviderCapabilities:
        return self.profile.capabilities

    @property
    def is_configured(self) -> bool:
        return bool(self.profile.base_url and self.profile.model and self.profile.api_key)

    def generate(
        self,
        request: LLMRequest,
        *,
        timeout_s: float,
        idempotency_key: str,
    ) -> ProviderResult:
        if not self.profile.base_url:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail=f"profile {self.profile_id} 缺少 base_url")
        if not self.profile.model:
            raise MediDiagError("PROVIDER_REQUEST_REJECTED", detail=f"profile {self.profile_id} 缺少 model")
        api_key = self.profile.api_key
        if not api_key:
            env_name = self.profile.api_key_env or "未配置的环境变量"
            raise MediDiagError("PROVIDER_AUTH_FAILED", detail=f"profile {self.profile_id} 需要环境变量 {env_name}")

        body, filtered = self._request_body(request)
        started = time.perf_counter()
        last_error: Exception | None = None
        attempt = 0
        network_retries = 0
        structured_retries = 0
        request_idempotency_key = idempotency_key
        while True:
            try:
                response = self._post(
                    f"{self.profile.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Idempotency-Key": request_idempotency_key,
                    },
                    json=body,
                    timeout=timeout_s,
                )
                if response.status_code >= 400:
                    self._raise_http_error(response)
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise MediDiagError("PROVIDER_SCHEMA_INVALID", detail="provider 响应不是 JSON") from exc
                if not isinstance(payload, dict):
                    raise MediDiagError("PROVIDER_SCHEMA_INVALID", detail="provider 响应必须是 JSON object")
                result = self._normalize_response(
                    payload,
                    response=response,
                    filtered=filtered,
                    retry_count=attempt,
                    latency_ms=_latency_ms(started),
                    response_format=request.response_format,
                )
                if self.profile.require_response_id and not result.response_id:
                    raise MediDiagError("PROVIDER_RESPONSE_ID_MISSING")
                return result
            except (httpx.TimeoutException, httpx.RequestError, MediDiagError) as exc:
                last_error = exc
                if (
                    isinstance(exc, MediDiagError)
                    and exc.code == "STRUCTURED_OUTPUT_INVALID"
                    and structured_retries < self._max_structured_retries
                ):
                    # JSON 契约失败属于一次新的受控生成，使用派生幂等键避免复用坏响应。
                    structured_retries += 1
                    attempt += 1
                    request_idempotency_key = f"{idempotency_key}:structured-retry-1"
                    self._sleep(self._retry_delay(exc, attempt - 1))
                    continue
                if self._retryable(exc) and network_retries < self._max_retries:
                    network_retries += 1
                    attempt += 1
                    self._sleep(self._retry_delay(exc, attempt - 1))
                    continue
                mapped = self._mapped_exception(exc)
                if mapped is exc:
                    raise
                raise mapped from exc

        raise MediDiagError("PROVIDER_UNAVAILABLE", detail=str(last_error))

    def _request_body(self, request: LLMRequest) -> tuple[dict[str, Any], tuple[str, ...]]:
        capabilities = self.capabilities()
        body: dict[str, Any] = {
            "model": request.model or self.profile.model,
            "messages": request.messages,
        }
        filtered: list[str] = []
        temperature = request.temperature if request.temperature is not None else self.profile.temperature
        top_p = request.top_p if request.top_p is not None else self.profile.top_p
        max_tokens = request.max_tokens if request.max_tokens is not None else self.profile.max_tokens
        self._optional_parameter(body, filtered, "temperature", temperature, capabilities.temperature)
        self._optional_parameter(body, filtered, "top_p", top_p, capabilities.top_p)
        self._optional_parameter(body, filtered, "max_tokens", max_tokens, capabilities.max_output_tokens)
        self._optional_parameter(body, filtered, "response_format", request.response_format, capabilities.structured_output)
        self._optional_parameter(body, filtered, "tools", request.tools, capabilities.tool_calling)
        self._optional_parameter(body, filtered, "tool_choice", request.tool_choice, capabilities.tool_calling)
        reasoning_mode = request.reasoning_mode or self.profile.reasoning_mode
        if reasoning_mode not in (None, "provider_default"):
            if capabilities.reasoning_mode:
                body["thinking"] = {"type": reasoning_mode}
            else:
                filtered.append("reasoning_mode")
        return body, tuple(filtered)

    @staticmethod
    def _optional_parameter(
        body: dict[str, Any],
        filtered: list[str],
        name: str,
        value: Any,
        supported: bool,
    ) -> None:
        if value is None:
            return
        if supported:
            body[name] = value
        else:
            filtered.append(name)

    def _normalize_response(
        self,
        payload: dict[str, Any],
        *,
        response: httpx.Response,
        filtered: tuple[str, ...],
        retry_count: int,
        latency_ms: int,
        response_format: dict[str, Any] | None,
    ) -> ProviderResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise MediDiagError("PROVIDER_SCHEMA_INVALID", detail="provider 响应缺少 choices[0]")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise MediDiagError("PROVIDER_SCHEMA_INVALID", detail="provider 响应缺少 message")
        content = _normalize_content(message.get("content"))
        tool_calls = _normalize_tool_calls(message.get("tool_calls"))
        if not content and not tool_calls:
            raise MediDiagError("PROVIDER_SCHEMA_INVALID", detail="provider 返回空 content 且无 tool_calls")
        parsed_json = _parse_structured_content(content) if response_format is not None else None
        response_id = _response_id(response, payload)
        fingerprint = payload.get("system_fingerprint")
        system_fingerprint = fingerprint.strip() if isinstance(fingerprint, str) and fingerprint.strip() else None
        reasoning = message.get("reasoning_content")
        reasoning_content = reasoning if isinstance(reasoning, str) and reasoning.strip() else None
        provenance_mode: Literal[
            "system_fingerprint", "provider_response_id", "unverified"
        ] = (
            "system_fingerprint"
            if system_fingerprint
            else "provider_response_id"
            if response_id
            else "unverified"
        )
        model = payload.get("model")
        return ProviderResult(
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            model=model if isinstance(model, str) and model else self.profile.model,
            response_id=response_id,
            system_fingerprint=system_fingerprint,
            content=content,
            parsed_json=parsed_json,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=_normalize_usage(payload.get("usage")),
            finish_reason=choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
            raw_error_code=None,
            latency_ms=latency_ms,
            retry_count=retry_count,
            provenance_mode=provenance_mode,
            filtered_parameters=filtered,
        )

    @staticmethod
    def _raise_http_error(response: httpx.Response) -> None:
        status = response.status_code
        error_text = response.text.lower()
        context: dict[str, Any] = {"http_status": status}
        if status in {400, 403, 422} and any(
            token in error_text for token in ("content policy", "safety", "moderation")
        ):
            raise MediDiagError("PROVIDER_CONTENT_BLOCKED", context=context)
        if status in {400, 413, 422} and any(
            token in error_text for token in ("context", "token limit", "maximum context")
        ):
            raise MediDiagError("PROVIDER_CONTEXT_LIMIT", context=context)
        if status in {401, 403}:
            raise MediDiagError("PROVIDER_AUTH_FAILED", context=context)
        if status == 429:
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                context["retry_after_seconds"] = retry_after
            raise MediDiagError("PROVIDER_RATE_LIMITED", context=context)
        if 500 <= status < 600:
            raise MediDiagError("PROVIDER_UNAVAILABLE", context=context)
        raise MediDiagError("PROVIDER_REQUEST_REJECTED", context=context)

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        if isinstance(exc, MediDiagError):
            retry_after = exc.context.get("retry_after_seconds")
            if isinstance(retry_after, (int, float)) and not isinstance(retry_after, bool):
                return max(0.0, float(retry_after))
        return float(self._retry_backoff_seconds * (2**attempt))

    @staticmethod
    def _mapped_exception(exc: Exception) -> Exception:
        if isinstance(exc, httpx.TimeoutException):
            return MediDiagError("LLM_TIMEOUT", detail=str(exc))
        if isinstance(exc, httpx.RequestError):
            return MediDiagError("PROVIDER_NETWORK_ERROR", detail=str(exc))
        return exc

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.RequestError)):
            return True
        return isinstance(exc, MediDiagError) and exc.spec.retryable


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


def _normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return ""


def _normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _parse_structured_content(content: str) -> dict[str, Any]:
    candidate = content.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediDiagError("STRUCTURED_OUTPUT_INVALID", detail="provider 结构化输出不是 JSON object") from exc
    if not isinstance(parsed, dict):
        raise MediDiagError("STRUCTURED_OUTPUT_INVALID", detail="provider 结构化输出必须是 JSON object")
    return parsed


def _normalize_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    aliases = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "total_tokens": "total_tokens",
        "prompt_cache_hit_tokens": "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens": "prompt_cache_miss_tokens",
    }
    result: dict[str, int] = {}
    for source, target in aliases.items():
        raw = value.get(source)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            result[target] = raw
    return result


def _response_id(response: httpx.Response, payload: dict[str, Any]) -> str | None:
    for name in ("x-request-id", "request-id", "x-correlation-id"):
        value = response.headers.get(name)
        if value:
            return str(value)
    body_id = payload.get("id")
    return body_id.strip() if isinstance(body_id, str) and body_id.strip() else None


def _latency_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
