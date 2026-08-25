"""确定性离线 LLM Provider。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from medidiag.llm.contracts import LLMRequest, ProviderCapabilities, ProviderResult


class FakeProvider:
    """用于单元测试和离线演示；不访问网络，不生成医学诊断。"""

    provider_id = "fake"

    def __init__(self, *, profile_id: str = "fake_offline", model: str = "deterministic-v1") -> None:
        self.profile_id = profile_id
        self.model = model

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            structured_output=True,
            tool_calling=True,
            response_id=True,
            system_fingerprint=True,
        )

    def generate(
        self,
        request: LLMRequest,
        *,
        timeout_s: float,
        idempotency_key: str,
    ) -> ProviderResult:
        del timeout_s
        payload: dict[str, Any] = {
            "summary": "离线确定性 Provider 仅用于工程测试，不构成医疗建议。",
            "prompt_version": request.prompt_version,
        }
        content = json.dumps(payload, ensure_ascii=False)
        response_id = "fake_" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:16]
        return ProviderResult(
            provider_id=self.provider_id,
            profile_id=self.profile_id,
            model=request.model or self.model,
            response_id=response_id,
            system_fingerprint="fake-deterministic-v1",
            content=content,
            parsed_json=payload,
            tool_calls=[],
            reasoning_content=None,
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            raw_error_code=None,
            latency_ms=0,
            retry_count=0,
            provenance_mode="system_fingerprint",
        )
