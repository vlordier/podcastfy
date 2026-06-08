"""
Conversation Configuration Module

This module handles the loading and management of conversation configuration settings
for the Podcastfy application. It uses a YAML file for conversation-specific configuration settings.
"""

import os
import sys
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from podcastfy.utils.constants import DEFAULT_CREATIVITY, DEFAULT_MAX_NUM_CHUNKS, DEFAULT_MIN_CHUNK_SIZE
from podcastfy.utils.enums import AudioFormat, TTSProvider
from podcastfy.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_conversation_config_path(config_file: str = "conversation_config.yaml") -> str | None:
    """
    Get the path to the conversation_config.yaml file.

    Returns:
            str: The path to the conversation_config.yaml file.
    """
    try:
        # Check if the script is running in a PyInstaller bundle
        base_path = sys._MEIPASS if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))  # noqa: SLF001

        # Look for conversation_config.yaml in the same directory as the script
        config_path = os.path.join(base_path, config_file)
        if os.path.exists(config_path):
            return config_path

        # If not found, look in the parent directory (package root)
        config_path = os.path.join(os.path.dirname(base_path), config_file)
        if os.path.exists(config_path):
            return config_path

        # If still not found, look in the current working directory
        config_path = os.path.join(os.getcwd(), config_file)
        if os.path.exists(config_path):
            return config_path

        msg = f"{config_file} not found"
        raise FileNotFoundError(msg)

    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"Error locating {config_file}: {e}")
        return None


class TTSProviderConfig(BaseModel):
    """Pydantic model for TTS provider configuration."""

    default_voices: dict[str, str] = Field(default_factory=dict)
    model: str | None = None


class OutputDirectories(BaseModel):
    transcripts: str = "./data/transcripts"
    audio: str = "./data/audio"


class ConversationConfigModel(BaseModel):
    """Pydantic model for conversation configuration."""

    conversation_style: list[str] = Field(default_factory=lambda: ["engaging", "fast-paced", "enthusiastic"])
    roles_person1: str = "main summarizer"
    roles_person2: str = "questioner/clarifier"
    dialogue_structure: list[str] = Field(
        default_factory=lambda: ["Introduction", "Main Content Summary", "Conclusion"]
    )
    podcast_name: str = "PODCASTIFY"
    podcast_tagline: str = "Your Personal Generative AI Podcast"
    output_language: str = "English"
    engagement_techniques: list[str] = Field(
        default_factory=lambda: ["rhetorical questions", "anecdotes", "analogies", "humor"]
    )
    creativity: float = Field(default=DEFAULT_CREATIVITY, ge=0, le=2)
    user_instructions: str = ""
    max_num_chunks: int = Field(default=DEFAULT_MAX_NUM_CHUNKS, ge=1, le=50)
    min_chunk_size: int = Field(default=DEFAULT_MIN_CHUNK_SIZE, ge=50)
    text_to_speech: dict[str, TTSProviderConfig] = Field(default_factory=dict)
    default_tts_model: TTSProvider = TTSProvider.OPENAI
    audio_format: AudioFormat = AudioFormat.MP3
    temp_audio_dir: str = "data/audio/tmp/"
    ending_message: str = "See You Next Time!"
    output_directories: OutputDirectories = Field(default_factory=OutputDirectories)

    @field_validator("conversation_style")
    @classmethod
    def no_empty_styles(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            msg = "conversation_style entries cannot be empty"
            raise ValueError(msg)
        return v

    @field_validator("engagement_techniques")
    @classmethod
    def no_empty_techniques(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            msg = "engagement_techniques entries cannot be empty"
            raise ValueError(msg)
        return v


def load_conversation_config_model(config_data: str | dict[str, Any] | None = None) -> ConversationConfigModel:
    if config_data is None:
        config_path = get_conversation_config_path()
        if config_path:
            with open(config_path, "r") as file:
                raw = yaml.safe_load(file) or {}
        else:
            raw = {}
    elif isinstance(config_data, str):
        with open(config_data, "r") as file:
            raw = yaml.safe_load(file) or {}
    else:
        raw = config_data

    # Parse TTS provider configs
    tts_raw = raw.get("text_to_speech", {})
    tts_providers = {}
    for provider_name, provider_cfg in tts_raw.items():
        if isinstance(provider_cfg, dict):
            if provider_name in (
                "output_directories",
                "audio_format",
                "temp_audio_dir",
                "ending_message",
                "default_tts_model",
            ):
                continue
            tts_providers[provider_name] = TTSProviderConfig(**provider_cfg)

    output_dirs_raw = tts_raw.get("output_directories", {}) if isinstance(tts_raw, dict) else {}
    if not output_dirs_raw and isinstance(raw, dict) and "output_directories" in raw:
        output_dirs_raw = raw["output_directories"]

    kwargs = {k: v for k, v in raw.items() if k != "text_to_speech"}
    return ConversationConfigModel(
        **kwargs,
        text_to_speech=tts_providers,
        output_directories=OutputDirectories(**output_dirs_raw) if output_dirs_raw else OutputDirectories(),
    )
