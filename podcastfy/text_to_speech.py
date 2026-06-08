"""
Text-to-Speech Module for converting text into speech using various providers.

This module provides functionality to convert text into speech using various TTS models.
It supports ElevenLabs, Google, OpenAI and Edge TTS services and handles the conversion process,
including cleaning of input text and merging of audio files.
"""

import io
import logging
import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from pydub import AudioSegment

from .tts.factory import TTSProviderFactory
from .utils.config_conversation import load_conversation_config_model, TTSProviderConfig, ConversationConfigModel
from .tts.base import QAPair
from .utils.enums import TTSProvider, ApiKeyLabel
from podcastfy.utils.constants import (
    GEMINI_MULTI_TTS_MODEL,
    GEMINI_MULTI_VOICE1,
    GEMINI_MULTI_VOICE2,
    DEFAULT_BITRATE,
    DEFAULT_CODEC,
)

logger = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        conversation_config: Optional[ConversationConfigModel] = None,
    ):
        if isinstance(conversation_config, ConversationConfigModel):
            self.conversation_config = conversation_config
        else:
            self.conversation_config = load_conversation_config_model(conversation_config)
        self.tts_config = self.conversation_config.text_to_speech

        # Get API key from config if not provided
        if not api_key:
            _api_key_map = {
                TTSProvider.OPENAI: ApiKeyLabel.OPENAI,
                TTSProvider.ELEVENLABS: ApiKeyLabel.ELEVENLABS,
                TTSProvider.GEMINI: ApiKeyLabel.GEMINI,
                TTSProvider.GEMINI_MULTI: ApiKeyLabel.GEMINI,
            }
            api_key_label = _api_key_map.get(TTSProvider(model.lower()))
            api_key = os.environ.get(api_key_label.value, None) if api_key_label else None

        # Initialize provider using factory
        self.provider = TTSProviderFactory.create(
            provider_name=model, api_key=api_key, model=model
        )

        # Setup directories and config
        self._setup_directories()
        self.audio_format = self.conversation_config.audio_format
        self.ending_message = self.conversation_config.ending_message

    def _get_provider_config(self) -> TTSProviderConfig:
        """Get provider-specific configuration."""
        provider_name = self.provider.__class__.__name__.lower().replace("tts", "")
        provider_config = self.tts_config.get(provider_name)

        if provider_config is None:
            provider_config = TTSProviderConfig(
                model=None,
                default_voices={
                    "question": None,
                    "answer": None,
                },
            )

        return provider_config

    def convert_to_speech(self, text: str, output_file: str) -> None:
        """
        Convert input text to speech and save as an audio file.

        Args:
                text (str): Input text to convert to speech.
                output_file (str): Path to save the output audio file.

        Raises:
            ValueError: If the input text is not properly formatted
        """
        # Validate transcript format

        cleaned_text = text

        try:

            if self.provider.multi_speaker:
                audio_data = self.provider.generate_audio(
                    cleaned_text,
                    voice=GEMINI_MULTI_VOICE1,
                    model=GEMINI_MULTI_TTS_MODEL,
                    voice2=GEMINI_MULTI_VOICE2,
                    ending_message=self.ending_message,
                )
                try:
                    if not audio_data:
                        raise ValueError("No audio data produced")
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    segment = AudioSegment.from_file(io.BytesIO(audio_data))
                    segment.export(
                        output_file,
                        format=self.audio_format,
                        codec=DEFAULT_CODEC,
                        bitrate=DEFAULT_BITRATE
                    )
                except Exception as e:
                    logger.error(f"Error during audio processing: {str(e)}")
                    raise
            else:
                with tempfile.TemporaryDirectory(dir=self.temp_audio_dir) as temp_dir:
                    audio_segments = self._generate_audio_segments(
                        cleaned_text, temp_dir
                    )
                    self._merge_audio_files(audio_segments, output_file)
                    logger.info(f"Audio saved to {output_file}")

        except Exception as e:
            logger.error(f"Error converting text to speech: {str(e)}")
            raise

    def _generate_audio_segments(self, text: str, temp_dir: str) -> List[str]:
        """Generate audio segments for each Q&A pair."""
        qa_pairs = self.provider.split_qa(
            text, self.ending_message, self.provider.get_supported_tags()
        )
        audio_files = []
        provider_config = self._get_provider_config()

        for idx, (question, answer) in enumerate(qa_pairs, 1):
            for speaker_type, content in [("question", question), ("answer", answer)]:
                temp_file = os.path.join(
                    temp_dir, f"{idx}_{speaker_type}.{self.audio_format}"
                )
                voices = provider_config.default_voices or {}
                voice = voices.get(speaker_type)
                model = provider_config.model

                audio_data = self.provider.generate_audio(content, voice, model)
                with open(temp_file, "wb") as f:
                    f.write(audio_data)
                audio_files.append(temp_file)

        return audio_files

    def _merge_audio_files(self, audio_files: List[str], output_file: str) -> None:
        """
        Merge the provided audio files sequentially, ensuring questions come before answers.

        Args:
                audio_files: List of paths to audio files to merge
                output_file: Path to save the merged audio file
        """
        try:

            def get_sort_key(file_path: str) -> Tuple[int, int]:
                """
                Create sort key from filename that puts questions before answers.
                Example filenames: "1_question.mp3", "1_answer.mp3"
                """
                basename = os.path.basename(file_path)
                # Extract the index number and type (question/answer)
                idx = int(basename.split("_")[0])
                is_answer = basename.split("_")[1].startswith("answer")
                return (
                    idx,
                    1 if is_answer else 0,
                )  # Questions (0) come before answers (1)

            # Sort files by index and type (question/answer)
            audio_files.sort(key=get_sort_key)

            # Create empty audio segment
            combined = AudioSegment.empty()

            # Add each audio file to the combined segment
            for file_path in audio_files:
                combined += AudioSegment.from_file(file_path, format=self.audio_format)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Export the combined audio
            combined.export(output_file, format=self.audio_format)
            logger.info(f"Merged audio saved to {output_file}")

        except Exception as e:
            logger.error(f"Error merging audio files: {str(e)}")
            raise

    def _setup_directories(self) -> None:
        """Setup required directories for audio processing."""
        self.output_directories = self.conversation_config.output_directories.model_dump()
        temp_dir = self.conversation_config.temp_audio_dir.rstrip("/").split("/")
        self.temp_audio_dir = os.path.join(*temp_dir)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.temp_audio_dir = os.path.join(base_dir, self.temp_audio_dir)

        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Create directories if they don't exist
        for dir_path in [
            self.conversation_config.output_directories.transcripts,
            self.conversation_config.output_directories.audio,
            self.temp_audio_dir,
        ]:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
