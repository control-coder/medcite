"""共享脱敏契约。

`trace_exporter` 与结构化日志必须遵守同一套规则，否则两条输出通道会各自演化出
不同的敏感字段定义。这里是唯一定义处：新增敏感键名或模式只改本模块。

规则本身是兜底，不是许可。调用方仍然不得把病例正文、证据文本、prompt、完整
provider 响应或 API key 传进来——脱敏能挡住已知形态，挡不住未知形态。
"""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "authorization",
        "idempotency_key",
        "password",
        "prompt",
        "question",
        "secret",
        "token",
    }
)

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
CN_ID = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")

REDACTED = "[REDACTED]"


def is_sensitive_key(key: str) -> bool:
    """键名是否属于必须整体屏蔽的字段。"""
    normalized = key.lower().replace("-", "_")
    return (
        normalized in SENSITIVE_KEYS
        or normalized.endswith(("_api_key", "_password", "_secret", "_token"))
        or "authorization" in normalized
        or normalized.endswith("question")
        or normalized == "prompt"
    )


def redact_text(value: str) -> str:
    """屏蔽字符串中的 bearer token、邮箱、手机号和身份证号。"""
    value = BEARER.sub("Bearer [REDACTED]", value)
    value = EMAIL.sub("[REDACTED_EMAIL]", value)
    value = PHONE.sub("[REDACTED_PHONE]", value)
    return CN_ID.sub("[REDACTED_ID]", value)


def sanitize(value: Any, key: str | None = None) -> Any:
    """递归脱敏任意 JSON 兼容结构。"""
    if key and is_sensitive_key(key):
        return REDACTED
    if isinstance(value, dict):
        return {item_key: sanitize(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
