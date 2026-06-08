"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        config_conversation: dict | None = None,
    ) -> str: ...
