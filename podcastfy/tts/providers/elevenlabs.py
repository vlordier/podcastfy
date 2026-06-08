"""ElevenLabs TTS provider implementation."""

from elevenlabs import client as elevenlabs_client

from podcastfy.utils.constants import ELEVENLABS_TTS_MODEL

from ..base import TTSProvider


class ElevenLabsTTS(TTSProvider):
    def __init__(self, api_key: str, model: str = ELEVENLABS_TTS_MODEL) -> None:
        """
        Initialize ElevenLabs TTS provider.

        Args:
            api_key (str): ElevenLabs API key
            model (str): Model name to use. Defaults to "eleven_multilingual_v2"
        """
        self.client = elevenlabs_client.ElevenLabs(api_key=api_key)
        self.model = model

    def generate_audio(self, text: str, voice: str, model: str, voice2: str | None = None) -> bytes:
        """Generate audio using ElevenLabs API."""
        audio = self.client.generate(text=text, voice=voice, model=model)
        return b"".join(chunk for chunk in audio if chunk)

    def get_supported_tags(self) -> list[str]:
        """Get supported SSML tags."""
        return super().get_supported_tags()
