"""Factory for creating LLM providers."""

from typing import ClassVar

from podcastfy.utils.enums import LLMProvider as LLMProviderEnum

from .base import LLMProvider
from .providers.gemini import GeminiLLM
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
    _providers: ClassVar[dict[LLMProviderEnum, type[LLMProvider]]] = {
        LLMProviderEnum.GEMINI: GeminiLLM,
        LLMProviderEnum.LITELLM: LiteLLM,
        LLMProviderEnum.LLAMAFILE: LlamafileLLM,
    }

    @classmethod
    def create(cls, name: LLMProviderEnum, api_key: str, model: str, **kwargs) -> LLMProvider:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            msg = f"Unknown LLM provider: {name}. Choose from: {', '.join(p.name for p in cls._providers)}"
            raise ValueError(msg)
        return provider_cls(api_key=api_key, model=model, **kwargs)

    @classmethod
    def register(cls, name: LLMProviderEnum, provider_cls: type[LLMProvider]) -> None:
        cls._providers[name] = provider_cls
