"""Factory for creating LLM providers."""

import importlib
from typing import ClassVar

from podcastfy.utils.enums import LLMProvider as LLMProviderEnum

from .base import LLMProvider
from .providers.litellm import LiteLLM
from .providers.llamafile import LlamafileLLM


def detect_llm_provider(model_name: str) -> LLMProviderEnum:
    name = model_name.lower()
    if "gemini" in name:
        return LLMProviderEnum.GEMINI
    if name == "local" or name.startswith("llamafile"):
        return LLMProviderEnum.LLAMAFILE
    return LLMProviderEnum.LITELLM


class LLMProviderFactory:
    _providers: ClassVar[dict[LLMProviderEnum, type[LLMProvider]]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def _ensure_providers(cls) -> None:
        if cls._initialized:
            return
        cls._providers[LLMProviderEnum.LITELLM] = LiteLLM
        cls._providers[LLMProviderEnum.LLAMAFILE] = LlamafileLLM
        try:
            from .providers.gemini import GeminiLLM

            cls._providers[LLMProviderEnum.GEMINI] = GeminiLLM
        except ImportError:
            pass
        cls._initialized = True

    @classmethod
    def create(cls, name: LLMProviderEnum, api_key: str, model: str, **kwargs) -> LLMProvider:
        cls._ensure_providers()
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            msg = f"Unknown LLM provider: {name}. Choose from: {', '.join(p.name for p in cls._providers)}"
            raise ValueError(msg)
        return provider_cls(api_key=api_key, model=model, **kwargs)

    @classmethod
    def register(cls, name: LLMProviderEnum, provider_cls: type[LLMProvider]) -> None:
        cls._providers[name] = provider_cls
