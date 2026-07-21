"""OpenAI-compatible DeepSeek client used by the live demo provider.

The client intentionally exposes a small synchronous boundary because the single-machine
worker is synchronous. It sends no case data to a provider until the caller explicitly
selects the live provider and supplies ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from medidiag.config import get_settings
from medidiag.errors import MediDiagError


@dataclass(frozen=True)
class LLMCompletion:
    """A completion payload plus the provider request identifier for trace correlation."""

    content: str
    request_id: str | None
    model: str


class LLMClient:
    """Minimal DeepSeek/OpenAI-compatible chat-completions client.

    ``post`` is injectable so tests can assert request construction without performing
    external network calls. The API key is intentionally never included in returned data.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        seed: int = 42,
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.deepseek_api_key
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.model = model or settings.deepseek_model
        self.timeout = timeout if timeout is not None else settings.deepseek_timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._post = post or httpx.post

    @property
    def is_configured(self) -> bool:
        """Whether a non-empty API key is available."""
        return bool(self.api_key.strip())

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMCompletion:
        """Call ``/chat/completions`` and preserve a safe request identifier.

        HTTP and timeout exceptions are deliberately propagated for ``ProviderCallRunner``
        to classify, retry, and audit. Malformed completion bodies are normalized to the
        existing ``LLM_JSON_INVALID`` dependency error.
        """
        if not self.is_configured:
            raise MediDiagError(
                "PROVIDER_REQUEST_REJECTED",
                detail="DEEPSEEK_API_KEY is required for the live DeepSeek provider",
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
                "seed": self.seed,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MediDiagError(
                "LLM_JSON_INVALID",
                detail="chat completion response does not contain choices[0].message.content",
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise MediDiagError(
                "LLM_JSON_INVALID",
                detail="chat completion content is empty or not text",
            )
        return LLMCompletion(
            content=content,
            request_id=_request_id(response, data),
            model=self.model,
        )

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Compatibility helper returning only the generated text."""
        return self.complete(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ).content

    @staticmethod
    def parse_json_object(content: str) -> dict[str, Any]:
        """Parse a JSON object, tolerating one Markdown code fence from a provider."""
        candidate = content.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
            candidate = candidate.rsplit("```", 1)[0].strip()
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MediDiagError(
                "LLM_JSON_INVALID",
                detail="live DeepSeek output is not a JSON object",
            ) from exc
        if not isinstance(payload, dict):
            raise MediDiagError(
                "LLM_JSON_INVALID",
                detail="live DeepSeek output must be a JSON object",
            )
        return payload


def _request_id(
    response: httpx.Response, payload: dict[str, Any] | None = None
) -> str | None:
    request_id = next(
        (
            response.headers[name]
            for name in ("x-request-id", "request-id", "x-correlation-id")
            if name in response.headers
        ),
        None,
    )
    if request_id:
        return request_id
    body_request_id = payload.get("id") if payload else None
    return body_request_id.strip() if isinstance(body_request_id, str) else None
