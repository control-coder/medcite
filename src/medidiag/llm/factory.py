"""LLM Provider 工厂。"""

from __future__ import annotations

from medidiag.llm.contracts import LLMProvider
from medidiag.llm.fake import FakeProvider
from medidiag.llm.openai_compatible import OpenAICompatibleProvider, PostCallable, SleepCallable
from medidiag.llm.profiles import get_provider_profile


def build_llm_provider(
    profile_id: str | None = None,
    *,
    post: PostCallable | None = None,
    sleep: SleepCallable | None = None,
    max_retries: int = 2,
) -> LLMProvider:
    profile = get_provider_profile(profile_id)
    if profile.adapter == "fake":
        return FakeProvider(profile_id=profile.profile_id, model=profile.model)
    if profile.adapter == "openai_compatible":
        if sleep is None:
            return OpenAICompatibleProvider(
                profile,
                post=post,
                max_retries=max_retries,
            )
        return OpenAICompatibleProvider(
            profile,
            post=post,
            sleep=sleep,
            max_retries=max_retries,
        )
    raise ValueError(f"不支持的 LLM adapter: {profile.adapter}")
