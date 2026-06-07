"""Factory for creating LLM providers."""

from typing import ClassVar, Optional, Type
from .base import LLMProvider
from .providers.gemini import GeminiLLM
from .providers.litellm import LiteLLM
from .providers.llamafile import LlamafileLLM


def detect_llm_provider(model_name: str) -> str:
    name = model_name.lower()
    if "gemini" in name:
        return "gemini"
    if name == "local" or name.startswith("llamafile"):
        return "llamafile"
    return "litellm"


class LLMProviderFactory:
    _providers: ClassVar[dict[str, Type[LLMProvider]]] = {
        "gemini": GeminiLLM,
        "litellm": LiteLLM,
        "llamafile": LlamafileLLM,
    }

    @classmethod
    def create(cls, name: str, api_key: str, model: str, **kwargs) -> LLMProvider:
        provider_cls = cls._providers.get(name.lower())
        if not provider_cls:
            raise ValueError(
                f"Unknown LLM provider: {name}. "
                f"Choose from: {', '.join(cls._providers.keys())}"
            )
        return provider_cls(api_key=api_key, model=model, **kwargs)

    @classmethod
    def register(cls, name: str, provider_cls: Type[LLMProvider]) -> None:
        cls._providers[name.lower()] = provider_cls