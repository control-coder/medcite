"""MediDiag 的供应商无关 LLM Provider Layer。"""

from medidiag.llm.contracts import LLMProvider, LLMRequest, ProviderCapabilities, ProviderResult
from medidiag.llm.factory import build_llm_provider
from medidiag.llm.fake import FakeProvider
from medidiag.llm.openai_compatible import OpenAICompatibleProvider
from medidiag.llm.profiles import ProviderProfile, get_provider_profile, load_provider_profiles

__all__ = [
    "FakeProvider",
    "LLMProvider",
    "LLMRequest",
    "OpenAICompatibleProvider",
    "ProviderCapabilities",
    "ProviderProfile",
    "ProviderResult",
    "build_llm_provider",
    "get_provider_profile",
    "load_provider_profiles",
]
