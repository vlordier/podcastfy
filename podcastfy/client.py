"""
Podcastfy CLI

This module provides a command-line interface for generating podcasts or transcripts
from URLs or existing transcript files. It orchestrates the content extraction,
generation, and text-to-speech conversion processes.
"""

import os
import uuid
import typer
from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.content_generator import ContentGenerator
from podcastfy.text_to_speech import TextToSpeech
from podcastfy.utils.config import load_app_config_model, AppConfigModel
from podcastfy.utils.config_conversation import load_conversation_config_model
from podcastfy.utils.logger import setup_logger
from podcastfy.utils.enums import TTSProvider, ApiKeyLabel
from podcastfy.utils.constants import DEFAULT_TRANSCRIPTS_DIR, DEFAULT_AUDIO_DIR
from typing import List, Optional, Dict, Any

import logging

logger = setup_logger(__name__)

app = typer.Typer()

os.environ["LANGCHAIN_TRACING_V2"] = "False"


def process_content(
    urls: Optional[List[str]] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    generate_audio: bool = True,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False
):
    """
    Process URLs, a transcript file, image paths, or raw text to generate a podcast or transcript.
    """
    try:
        if config is None:
            app_config = load_app_config_model()

        # Load default conversation config
        conv_config = load_conversation_config_model()

        # Update with provided config if any
        if conversation_config:
            known_fields = {}
            for k, v in conversation_config.items():
                if k in type(conv_config).model_fields and k != "text_to_speech":
                    known_fields[k] = v
            # Also extract known top-level fields from nested text_to_speech (old structure)
            tts = conversation_config.get("text_to_speech", {})
            if isinstance(tts, dict):
                for nested_key in ("output_directories", "temp_audio_dir", "ending_message", "audio_format", "default_tts_model"):
                    if nested_key in tts and nested_key not in known_fields:
                        known_fields[nested_key] = tts[nested_key]
            if known_fields:
                # Convert nested dict to OutputDirectories if needed
                od = known_fields.get("output_directories")
                if od is not None and isinstance(od, dict):
                    from podcastfy.utils.config_conversation import OutputDirectories as OD
                    known_fields["output_directories"] = OD(**od)
                conv_config = conv_config.model_copy(update=known_fields)
        # Get output directories from conversation config
        tts_config = conv_config.text_to_speech
        output_directories = conv_config.output_directories

        if transcript_file:
            logger.info(f"Using transcript file: {transcript_file}")
            with open(transcript_file, "r") as file:
                qa_content = file.read()
        else:
            # Initialize content_extractor if needed
            content_extractor = None
            if urls or topic or (text and longform and len(text.strip()) < 100):
                content_extractor = ContentExtractor()

            content_generator = ContentGenerator(
                is_local=is_local,
                model_name=model_name,
                api_key_label=api_key_label,
                conversation_config=conv_config.model_dump()
            )

            combined_content = ""
            
            if urls:
                logger.info(f"Processing {len(urls)} links")
                contents = [content_extractor.extract_content(link) for link in urls]
                combined_content += "\n\n".join(contents)

            if text:
                if longform and len(text.strip()) < 100:
                    logger.info("Text too short for direct long-form generation. Extracting context...")
                    expanded_content = content_extractor.generate_topic_content(text)
                    combined_content += f"\n\n{expanded_content}"
                else:
                    combined_content += f"\n\n{text}"

            if topic:
                topic_content = content_extractor.generate_topic_content(topic)
                combined_content += f"\n\n{topic_content}"

            # Generate Q&A content using output directory from conversation config
            random_filename = f"transcript_{uuid.uuid4().hex}.txt"
            transcripts_dir = output_directories.transcripts or DEFAULT_TRANSCRIPTS_DIR if output_directories else DEFAULT_TRANSCRIPTS_DIR
            transcript_filepath = os.path.join(
                transcripts_dir,
                random_filename,
            )
            qa_content = content_generator.generate_qa_content(
                combined_content,
                image_file_paths=image_paths or [],
                output_filepath=transcript_filepath,
                longform=longform
            )

        if generate_audio:
            api_key = None
            if tts_model != TTSProvider.EDGE:
                api_key_label = {
                    TTSProvider.OPENAI: ApiKeyLabel.OPENAI,
                    TTSProvider.ELEVENLABS: ApiKeyLabel.ELEVENLABS,
                    TTSProvider.GEMINI: ApiKeyLabel.GEMINI,
                    TTSProvider.GEMINI_MULTI: ApiKeyLabel.GEMINI,
                }.get(TTSProvider(tts_model))
                api_key = os.environ.get(api_key_label.value, "") if api_key_label else ""

            text_to_speech = TextToSpeech(
                model=tts_model,
                api_key=api_key,
                conversation_config=conv_config.model_dump(),
            )

            random_filename = f"podcast_{uuid.uuid4().hex}.mp3"
            audio_dir = output_directories.audio or DEFAULT_AUDIO_DIR if output_directories else DEFAULT_AUDIO_DIR
            audio_file = os.path.join(
                audio_dir, random_filename
            )
            text_to_speech.convert_to_speech(qa_content, audio_file)
            logger.info(f"Podcast generated successfully using {tts_model} TTS model")
            return audio_file
        else:
            logger.info(f"Transcript generated successfully: {transcript_filepath}")
            return transcript_filepath

    except Exception as e:
        logger.error(f"An error occurred in the process_content function: {str(e)}")
        raise


@app.command()
def main(
    urls: list[str] = typer.Option(None, "--url", "-u", help="URLs to process"),
    file: typer.FileText = typer.Option(
        None, "--file", "-f", help="File containing URLs, one per line"
    ),
    transcript: typer.FileText = typer.Option(
        None, "--transcript", "-t", help="Path to a transcript file"
    ),
    tts_model: str = typer.Option(
        None,
        "--tts-model",
        "-tts",
        help="TTS model to use (openai, elevenlabs, edge, or gemini)",
    ),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", help="Generate only a transcript without audio"
    ),
    conversation_config_path: str = typer.Option(
        None,
        "--conversation-config",
        "-cc",
        help="Path to custom conversation configuration YAML file",
    ),
    image_paths: List[str] = typer.Option(
        None, "--image", "-i", help="Paths to image files to process"
    ),
    is_local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use a local LLM instead of a remote one (http://localhost:8080)",
    ),
    text: str = typer.Option(
        None, "--text", "-txt", help="Raw text input to be processed"
    ),
    llm_model_name: str = typer.Option(
        None, "--llm-model-name", "-m", help="LLM model name for transcript generation"
    ),
    api_key_label: str = typer.Option(
        None, "--api-key-label", "-k", help="Environment variable name for LLMAPI key"
    ),
    topic: str = typer.Option(
        None, "--topic", "-tp", help="Topic to generate podcast about"
    ),
    longform: bool = typer.Option(
        False, 
        "--longform", 
        "-lf", 
        help="Generate long-form content (only available for text input without images)"
    ),
):
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, image files, or raw text.
    """
    try:
        app_config = load_app_config_model()
        main_config = app_config.main or {}

        conversation_config = None
        # Load conversation config if provided
        if conversation_config_path:
            with open(conversation_config_path, "r") as f:
                conversation_config: Dict[str, Any] | None = yaml.safe_load(f)

        # Use default TTS model from conversation config if not specified
        if tts_model is None:
            default_conv = load_conversation_config_model()
            tts_model = default_conv.default_tts_model
            if isinstance(tts_model, TTSProvider):
                tts_model = tts_model.value

        if transcript:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            final_output = process_content(
                transcript_file=transcript.name,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                conversation_config=conversation_config,
                config=main_config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )
        else:
            urls_list = urls or []
            if file:
                urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise typer.BadParameter(
                    "No input provided. Use --url, --file, --transcript, --image, --text, or --topic."
                )

            final_output = process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=main_config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )

        if transcript_only:
            typer.echo(f"Transcript generated successfully: {final_output}")
        else:
            typer.echo(
                f"Podcast generated successfully using {tts_model} TTS model: {final_output}"
            )

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


def generate_podcast(
    urls: Optional[List[str]] = None,
    url_file: Optional[str] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    transcript_only: bool = False,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
) -> Optional[str]:
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, or image files.

    Args:
        urls (Optional[List[str]]): List of URLs to process.
        url_file (Optional[str]): Path to a file containing URLs, one per line.
        transcript_file (Optional[str]): Path to a transcript file.
        tts_model (Optional[str]): TTS model to use ('openai' [default], 'elevenlabs', 'edge', or 'gemini').
        transcript_only (bool): Generate only a transcript without audio. Defaults to False.
        config (Optional[Dict[str, Any]]): User-provided configuration dictionary.
        conversation_config (Optional[Dict[str, Any]]): User-provided conversation configuration dictionary.
        image_paths (Optional[List[str]]): List of image file paths to process.
        is_local (bool): Whether to use a local LLM. Defaults to False.
        text (Optional[str]): Raw text input to be processed.
        llm_model_name (Optional[str]): LLM model name for content generation.
        api_key_label (Optional[str]): Environment variable name for LLM API key.
        topic (Optional[str]): Topic to generate podcast about.

    Returns:
        Optional[str]: Path to the final podcast audio file, or None if only generating a transcript.
    """
    try:
        print("Generating podcast...")
        # Load default config
        app_config = load_app_config_model()

        # Update config if provided
        if config:
            if isinstance(config, dict):
                for key in [e.value for e in ApiKeyLabel]:
                    if key in config:
                        os.environ[key] = config[key]

        if not conversation_config:
            conversation_config = load_conversation_config_model().model_dump()

        main_config = app_config.main or {}

        # Use provided tts_model if specified, otherwise use the one from config
        if tts_model is None:
            tts_model_value = conversation_config.get("default_tts_model", TTSProvider.OPENAI.value)
            tts_model = tts_model_value.value if isinstance(tts_model_value, TTSProvider) else tts_model_value

        if transcript_file:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            return process_content(
                transcript_file=transcript_file,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=app_config,
                conversation_config=conversation_config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )
        else:
            urls_list = urls or []
            if url_file:
                with open(url_file, "r") as file:
                    urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise ValueError(
                    "No input provided. Please provide either 'urls', 'url_file', "
                    "'transcript_file', 'image_paths', 'text', or 'topic'."
                )

            return process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=app_config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise