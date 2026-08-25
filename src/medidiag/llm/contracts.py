"""LLM Provider 的统一数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ProviderCapabilities:
    """描述一个 profile 实际支持的协议和可选参数。"""

    chat_completions: bool = True
    responses: bool = False
    structured_output: bool = False
    tool_calling: bool = False
    reasoning_mode: bool = False
    temperature: bool = True
    top_p: bool = True
    streaming: bool = False
    max_output_tokens: bool = True
    response_id: bool = True
    system_fingerprint: bool = False


@dataclass(frozen=True)
class LLMRequest:
    """与供应商无关的生成请求。"""

    messages: list[Message]
    model: str | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning_mode: Literal["disabled", "enabled", "provider_default"] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = "unspecified"


@dataclass(frozen=True)
class ProviderResult:
    """归一化后的生成结果；不保存 API key 或完整原始响应。"""

    provider_id: str
    profile_id: str
    model: str
    response_id: str | None
    system_fingerprint: str | None
    content: str
    parsed_json: dict[str, Any] | None
    tool_calls: list[dict[str, Any]]
    reasoning_content: str | None
    usage: dict[str, int]
    finish_reason: str | None
    raw_error_code: str | None
    latency_ms: int
    retry_count: int
    provenance_mode: Literal["system_fingerprint", "provider_response_id", "unverified"]
    filtered_parameters: tuple[str, ...] = ()


class LLMProvider(Protocol):
    """Agent 与 workflow 只依赖该接口，不依赖供应商 SDK。"""

    provider_id: str
    profile_id: str

    def generate(
        self,
        request: LLMRequest,
        *,
        timeout_s: float,
        idempotency_key: str,
    ) -> ProviderResult: ...

    def capabilities(self) -> ProviderCapabilities: ...
