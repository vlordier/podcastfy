"""Factory for creating TTS providers."""

from typing import Type
from .base import TTSProvider
from .providers.elevenlabs import ElevenLabsTTS
from .providers.openai import OpenAITTS
from .providers.edge import EdgeTTS
from .providers.gemini import GeminiTTS
from .providers.geminimulti import GeminiMultiTTS
from podcastfy.utils.enums import TTSProvider as TTSProviderEnum


class TTSProviderFactory:
    """Factory class for creating TTS providers."""
    
    _providers: dict[TTSProviderEnum, Type[TTSProvider]] = {
        TTSProviderEnum.ELEVENLABS: ElevenLabsTTS,
        TTSProviderEnum.OPENAI: OpenAITTS,
        TTSProviderEnum.EDGE: EdgeTTS,
        TTSProviderEnum.GEMINI: GeminiTTS,
        TTSProviderEnum.GEMINI_MULTI: GeminiMultiTTS
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
            raise ValueError(f"Unsupported provider: {provider_name}. "
                           f"Choose from: {', '.join(p.value for p in TTSProviderEnum)}")
                            
        provider_class = cls._providers.get(provider_enum)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider_name}. "
                           f"Choose from: {', '.join(p.value for p in TTSProviderEnum)}")
                            
        return provider_class(api_key, model) if api_key else provider_class(model=model)
    
    @classmethod
    def register_provider(cls, name: TTSProviderEnum, provider_class: Type[TTSProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name] = provider_class