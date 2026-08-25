"""Provider profile 加载与环境变量解析。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import yaml

from medidiag.llm.contracts import ProviderCapabilities


@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    adapter: str
    provider_id: str
    model: str
    base_url: str | None
    api_key_env: str | None
    api_style: str
    timeout_s: float
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    reasoning_mode: str | None
    require_response_id: bool
    capabilities: ProviderCapabilities

    @property
    def api_key(self) -> str:
        """只在构造 adapter 时读取密钥，不将密钥写回 profile。"""
        if not self.api_key_env:
            return ""
        return os.getenv(self.api_key_env, "").strip()


def load_provider_profiles() -> tuple[str, dict[str, ProviderProfile]]:
    """从包内 YAML 加载 profile，避免在业务代码中写死供应商参数。"""
    resource = files("medidiag.llm").joinpath("profiles.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
        raise ValueError("llm profiles 配置必须包含 profiles 映射")

    default_profile = str(raw.get("default_profile", "fake_offline"))
    profiles: dict[str, ProviderProfile] = {}
    for profile_id, value in raw["profiles"].items():
        if not isinstance(value, dict):
            raise ValueError(f"profile {profile_id} 必须是映射")
        capability_raw = value.get("capabilities", {})
        if not isinstance(capability_raw, dict):
            raise ValueError(f"profile {profile_id}.capabilities 必须是映射")
        capabilities = ProviderCapabilities(**capability_raw)
        profiles[str(profile_id)] = ProviderProfile(
            profile_id=str(profile_id),
            adapter=str(value["adapter"]),
            provider_id=str(value.get("provider_id", profile_id)),
            model=_resolve_env(str(value.get("model", ""))),
            base_url=_optional_env(value.get("base_url")),
            api_key_env=_optional_text(value.get("api_key_env")),
            api_style=str(value.get("api_style", "chat_completions")),
            timeout_s=float(value.get("timeout_s", 60)),
            temperature=_optional_float(value.get("temperature")),
            top_p=_optional_float(value.get("top_p")),
            max_tokens=_optional_int(value.get("max_tokens")),
            reasoning_mode=_optional_text(value.get("reasoning_mode")),
            require_response_id=bool(value.get("require_response_id", False)),
            capabilities=capabilities,
        )
    if default_profile not in profiles:
        raise ValueError(f"默认 profile 不存在: {default_profile}")
    return default_profile, profiles


def get_provider_profile(profile_id: str | None = None) -> ProviderProfile:
    default_profile, profiles = load_provider_profiles()
    selected = profile_id or os.getenv("MEDIDIAG_LLM_PROFILE") or default_profile
    try:
        return profiles[selected]
    except KeyError as exc:
        raise ValueError(f"未知 LLM profile: {selected}") from exc


def _resolve_env(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        expression = value[2:-1]
        if ":-" in expression:
            env_name, default = expression.split(":-", 1)
            return os.getenv(env_name, default).strip()
        return os.getenv(expression, "").strip()
    return value


def _optional_env(value: Any) -> str | None:
    text = _optional_text(value)
    return _resolve_env(text) if text is not None else None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
