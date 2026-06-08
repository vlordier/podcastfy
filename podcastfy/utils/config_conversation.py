"""
Conversation Configuration Module

This module handles the loading and management of conversation configuration settings
for the Podcastfy application. It uses a YAML file for conversation-specific configuration settings.
"""

import os
import sys
from typing import Any, Dict, Optional
import yaml

from pydantic import BaseModel, Field, field_validator

from podcastfy.utils.enums import TTSProvider, AudioFormat
from podcastfy.utils.constants import DEFAULT_MAX_NUM_CHUNKS, DEFAULT_MIN_CHUNK_SIZE, DEFAULT_CREATIVITY
from podcastfy.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_conversation_config_path(config_file: str = 'conversation_config.yaml') -> Optional[str]:
	"""
	Get the path to the conversation_config.yaml file.
	
	Returns:
		str: The path to the conversation_config.yaml file.
	"""
	try:
		# Check if the script is running in a PyInstaller bundle
		if getattr(sys, 'frozen', False):
			base_path = sys._MEIPASS
		else:
			base_path = os.path.dirname(os.path.abspath(__file__))
		
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
		
		raise FileNotFoundError(f"{config_file} not found")
	
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
    dialogue_structure: list[str] = Field(default_factory=lambda: ["Introduction", "Main Content Summary", "Conclusion"])
    podcast_name: str = "PODCASTIFY"
    podcast_tagline: str = "Your Personal Generative AI Podcast"
    output_language: str = "English"
    engagement_techniques: list[str] = Field(default_factory=lambda: ["rhetorical questions", "anecdotes", "analogies", "humor"])
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
            raise ValueError("conversation_style entries cannot be empty")
        return v

    @field_validator("engagement_techniques")
    @classmethod
    def no_empty_techniques(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("engagement_techniques entries cannot be empty")
        return v


def load_conversation_config_model(config_conversation: Optional[Dict[str, Any]] = None) -> ConversationConfigModel:
	"""
	Load and return a ConversationConfigModel instance (Pydantic-validated).

	Args:
		config_conversation (Optional[Dict[str, Any]]): Configuration dictionary to use.
			If None, default config will be loaded from conversation_config.yaml.

	Returns:
		ConversationConfigModel: An instance of the Pydantic model.
	"""
	if config_conversation is None:
		config_path = get_conversation_config_path()
		if config_path:
			with open(config_path, 'r') as file:
				raw = yaml.safe_load(file)
		else:
			raw = {}
	else:
		raw = config_conversation

	# Parse TTS provider configs
	tts_raw = raw.get("text_to_speech", {})
	tts_providers = {}
	for provider_name, provider_cfg in tts_raw.items():
		if isinstance(provider_cfg, dict):
			# Skip non-provider keys
			if provider_name in ("output_directories", "audio_format", "temp_audio_dir", "ending_message", "default_tts_model"):
				continue
			tts_providers[provider_name] = TTSProviderConfig(**provider_cfg)

	output_dirs_raw = tts_raw.get("output_directories", {}) if isinstance(tts_raw, dict) else {}
	# Also support flat structure (from model_dump round-trip)
	if not output_dirs_raw and isinstance(raw, dict) and "output_directories" in raw:
		output_dirs_raw = raw["output_directories"]

	return ConversationConfigModel(
		conversation_style=raw.get("conversation_style", ["engaging", "fast-paced", "enthusiastic"]),
		roles_person1=raw.get("roles_person1", "main summarizer"),
		roles_person2=raw.get("roles_person2", "questioner/clarifier"),
		dialogue_structure=raw.get("dialogue_structure", ["Introduction", "Main Content Summary", "Conclusion"]),
		podcast_name=raw.get("podcast_name", "PODCASTIFY"),
		podcast_tagline=raw.get("podcast_tagline", "Your Personal Generative AI Podcast"),
		output_language=raw.get("output_language", "English"),
		engagement_techniques=raw.get("engagement_techniques", ["rhetorical questions", "anecdotes", "analogies", "humor"]),
		creativity=float(raw.get("creativity", DEFAULT_CREATIVITY)),
		user_instructions=raw.get("user_instructions", ""),
		max_num_chunks=raw.get("max_num_chunks", DEFAULT_MAX_NUM_CHUNKS),
		min_chunk_size=raw.get("min_chunk_size", DEFAULT_MIN_CHUNK_SIZE),
		text_to_speech=tts_providers,
		default_tts_model=tts_raw.get("default_tts_model", raw.get("default_tts_model", TTSProvider.OPENAI)),
		audio_format=tts_raw.get("audio_format", raw.get("audio_format", AudioFormat.MP3)),
		temp_audio_dir=tts_raw.get("temp_audio_dir", raw.get("temp_audio_dir", "data/audio/tmp/")),
		ending_message=tts_raw.get("ending_message", raw.get("ending_message", "See You Next Time!")),
		output_directories=OutputDirectories(**output_dirs_raw) if output_dirs_raw else OutputDirectories(),
	)