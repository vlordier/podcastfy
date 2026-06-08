"""Google Cloud Text-to-Speech provider implementation for single speaker."""

import logging

from google.cloud import texttospeech_v1

from podcastfy.utils.constants import COMMON_SSML_TAGS, GEMINI_TTS_VOICE

from ..base import TTSProvider

logger = logging.getLogger(__name__)


class GeminiTTS(TTSProvider):
    """Google Cloud Text-to-Speech provider for single speaker."""

    def __init__(self, api_key: str | None = None, model: str = GEMINI_TTS_VOICE):
        """
        Initialize Google Cloud TTS provider.

        Args:
            api_key (str): Google Cloud API key
            model (str): Default voice model to use
        """
        self.model = model
        try:
            self.client = texttospeech_v1.TextToSpeechClient(client_options={"api_key": api_key} if api_key else None)
        except Exception as e:
            logger.error(f"Failed to initialize Google TTS client: {str(e)}")
            raise

    def generate_audio(
        self, text: str, voice: str = GEMINI_TTS_VOICE, model: str | None = None, voice2: str | None = None, **kwargs
    ) -> bytes:
        """
        Generate audio using Google Cloud TTS API.

        Args:
            text (str): Text to convert to speech
            voice (str): Voice ID/name to use (format: "{language-code}-{name}-{gender}")
            model (str): Optional model override

        Returns:
            bytes: Audio data

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If audio generation fails
        """
        self.validate_parameters(text, voice, model or self.model)

        try:
            # Create synthesis input
            synthesis_input = texttospeech_v1.SynthesisInput(text=text)

            # Parse language code from voice ID (e.g., "en-IN" from "en-IN-Journey-D")
            language_code = "-".join(voice.split("-")[:2])

            voice_params = texttospeech_v1.VoiceSelectionParams(
                language_code=language_code,
                name=voice,
            )

            # Set audio config
            audio_config = texttospeech_v1.AudioConfig(audio_encoding=texttospeech_v1.AudioEncoding.MP3)

            # Generate speech
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_config
            )

            return response.audio_content

        except Exception as e:
            logger.error(f"Failed to generate audio: {str(e)}")
            msg = f"Failed to generate audio: {str(e)}"
            raise RuntimeError(msg) from e

    def get_supported_tags(self) -> list[str]:
        """Get supported SSML tags."""
        return list(COMMON_SSML_TAGS)

    def validate_parameters(self, text: str, voice: str, model: str) -> None:
        """
        Validate input parameters before generating audio.

        Args:
            text (str): Input text
            voice (str): Voice ID/name
            model (str): Model name

        Raises:
            ValueError: If parameters are invalid
        """
        super().validate_parameters(text, voice, model)
