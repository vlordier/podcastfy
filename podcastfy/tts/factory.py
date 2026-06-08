"""Factory for creating TTS providers."""

from podcastfy.utils.enums import TTSProvider as TTSProviderEnum

from .base import TTSProvider
from .providers.edge import EdgeTTS
from .providers.elevenlabs import ElevenLabsTTS
from .providers.gemini import GeminiTTS
from .providers.geminimulti import GeminiMultiTTS
from .providers.openai import OpenAITTS


class TTSProviderFactory:
    """Factory class for creating TTS providers."""

    _providers: dict[TTSProviderEnum, type[TTSProvider]] = {
        TTSProviderEnum.ELEVENLABS: ElevenLabsTTS,
        TTSProviderEnum.OPENAI: OpenAITTS,
        TTSProviderEnum.EDGE: EdgeTTS,
        TTSProviderEnum.GEMINI: GeminiTTS,
        TTSProviderEnum.GEMINI_MULTI: GeminiMultiTTS,
    }

    @classmethod
    def create(cls, provider_name: str, api_key: str | None = None, model: str | None = None) -> TTSProvider:
        """
        Create a TTS provider instance.

        Args:
            provider_name: Name of the provider to create
            api_key: Optional API key for the provider
            model: Optional model name for the provider

        Returns:
            TTSProvider instance

        Raises:
            ValueError: If provider_name is not supported
        """
        try:
            provider_enum = TTSProviderEnum(provider_name.lower())
        except ValueError:
            msg = f"Unsupported provider: {provider_name}. Choose from: {', '.join(p.value for p in TTSProviderEnum)}"
            raise ValueError(
                msg
            )

        provider_class = cls._providers.get(provider_enum)
        if not provider_class:
            msg = f"Unsupported provider: {provider_name}. Choose from: {', '.join(p.value for p in TTSProviderEnum)}"
            raise ValueError(
                msg
            )

        return provider_class(api_key, model) if api_key else provider_class(model=model)

    @classmethod
    def register_provider(cls, name: TTSProviderEnum, provider_class: type[TTSProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name] = provider_class
