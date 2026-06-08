"""
Configuration Module

This module handles the loading and management of configuration settings for the Podcastfy application.
It uses a YAML file for non-sensitive configuration settings.
"""

import os
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field

from podcastfy.utils.constants import DEFAULT_GEMINI_LLM, DEFAULT_MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS


def get_config_path(config_file: str = 'config.yaml') -> Optional[str]:
	"""
	Get the path to the config.yaml file.
	
	Returns:
		str: The path to the config.yaml file.
	"""
	try:
		base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		
		# Look for config.yaml in the package root
		config_path = os.path.join(base_path, config_file)
		if os.path.exists(config_path):
			return config_path
		
		# If not found, look in the current working directory
		config_path = os.path.join(os.getcwd(), config_file)
		if os.path.exists(config_path):
			return config_path
		
		raise FileNotFoundError(f"{config_file} not found")
	
	except (FileNotFoundError, PermissionError, OSError) as e:
		print(f"Error locating {config_file}: {e}")
		return None


def _load_yaml_config() -> dict:
	"""Load the raw YAML configuration as a dictionary."""
	config_path = get_config_path()
	if config_path:
		with open(config_path, 'r') as file:
			return yaml.safe_load(file)
	return {}


class ContentGeneratorConfigModel(BaseModel):
    llm_model: str = DEFAULT_GEMINI_LLM
    meta_llm_model: str = DEFAULT_GEMINI_LLM
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=MIN_OUTPUT_TOKENS, le=MAX_OUTPUT_TOKENS)
    prompt_template: str = "souzatharsis/podcastfy_multimodal_cleanmarkup"
    prompt_commit: str = "b2365f11"
    longform_prompt_template: str = "souzatharsis/podcastfy_longform"
    longform_prompt_commit: str = "acfdbc91"
    cleaner_prompt_template: str = "souzatharsis/podcastfy_longform_clean"
    cleaner_prompt_commit: str = "8c110a0b"
    rewriter_prompt_template: str = "souzatharsis/podcast_rewriter"
    rewriter_prompt_commit: str = "8ee296fb"


class ContentExtractorConfigModel(BaseModel):
    youtube_url_patterns: list[str] = Field(default_factory=lambda: ["youtube.com", "youtu.be"])


class AppConfigModel(BaseModel):
    content_generator: ContentGeneratorConfigModel = Field(default_factory=ContentGeneratorConfigModel)
    content_extractor: ContentExtractorConfigModel = Field(default_factory=ContentExtractorConfigModel)
    main: Optional[Dict[str, Any]] = None


def load_app_config_model() -> AppConfigModel:
    raw = _load_yaml_config()
    cg_raw = raw.get("content_generator", {})
    ce_raw = raw.get("content_extractor", {})
    return AppConfigModel(
        content_generator=ContentGeneratorConfigModel(**cg_raw),
        content_extractor=ContentExtractorConfigModel(**ce_raw),
        main=raw.get("main"),
    )