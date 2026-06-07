"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import yaml
import re
import time
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from ..client import generate_podcast
import uvicorn

from podcastfy.utils.constants import MAX_URLS, TEMP_FILE_MAX_AGE_SECONDS, DEFAULT_PORT, TEMP_DIR_NAME
from podcastfy.utils.enums import TTSProvider, ApiKeyLabel


logger = logging.getLogger(__name__)


def load_base_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "podcastfy" / "conversation_config.yaml"
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load base config: {e}")
        return {}

def merge_configs(base_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user configuration with base configuration, preferring user values."""
    merged = base_config.copy()
    
    # Handle special cases for nested dictionaries
    if 'text_to_speech' in merged and 'text_to_speech' in user_config:
        merged['text_to_speech'].update(user_config.get('text_to_speech', {}))
    
    # Update top-level keys
    for key, value in user_config.items():
        if key != 'text_to_speech':  # Skip text_to_speech as it's handled above
            if value is not None:  # Only update if value is not None
                merged[key] = value
                
    return merged

class GenerateRequest(BaseModel):
    urls: List[str] = Field(default_factory=list, max_length=MAX_URLS)
    tts_model: Optional[str] = Field(default=None, pattern=r"^(openai|elevenlabs|edge|gemini|geminimulti)?$")  # Values from TTSProvider enum
    user_instructions: Optional[str] = None
    creativity: Optional[float] = Field(default=None, ge=0, le=2)
    openai_key: Optional[str] = None
    google_key: Optional[str] = None
    elevenlabs_key: Optional[str] = None
    is_long_form: bool = False
    user_config: Optional[dict] = None
    conversation_config: Optional[dict] = None


async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    expected_key = os.getenv("PODCASTFY_API_KEY")
    if expected_key:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing API key"
            )
    return x_api_key


app = FastAPI()

TEMP_DIR = os.path.join(os.path.dirname(__file__), TEMP_DIR_NAME)
os.makedirs(TEMP_DIR, exist_ok=True)

# Clean audio files older than 1 hour
now = time.time()
for f in os.listdir(TEMP_DIR):
    fp = os.path.join(TEMP_DIR, f)
    if os.path.isfile(fp) and now - os.path.getmtime(fp) > TEMP_FILE_MAX_AGE_SECONDS:
        try:
            os.remove(fp)
        except OSError:
            pass

@app.post("/generate")
def generate_podcast_endpoint(data: GenerateRequest, auth: str = Depends(verify_api_key)):
    """"""
    try:
        # Set environment variables
        os.environ[ApiKeyLabel.OPENAI.value] = data.openai_key
        os.environ[ApiKeyLabel.GEMINI.value] = data.google_key
        os.environ[ApiKeyLabel.ELEVENLABS.value] = data.elevenlabs_key

        # Load base configuration
        base_config = load_base_config()
        
        # Get TTS model and its configuration from base config
        tts_model = data.tts_model or base_config.get('text_to_speech', {}).get('default_tts_model', TTSProvider.OPENAI.value)
        tts_base_config = base_config.get('text_to_speech', {}).get(tts_model, {})
        
        # Get voices (use user-provided voices or fall back to defaults)
        voices = data.voices if hasattr(data, 'voices') else {}
        default_voices = tts_base_config.get('default_voices', {})
        
        # Prepare user configuration
        user_config = {
            'creativity': float(data.creativity if data.creativity is not None else base_config.get('creativity', 0.7)),
            'conversation_style': base_config.get('conversation_style', []),
            'roles_person1': base_config.get('roles_person1'),
            'roles_person2': base_config.get('roles_person2'),
            'dialogue_structure': base_config.get('dialogue_structure', []),
            'podcast_name': base_config.get('podcast_name'),
            'podcast_tagline': base_config.get('podcast_tagline'),
            'output_language': base_config.get('output_language', 'English'),
            'user_instructions': data.user_instructions if data.user_instructions is not None else base_config.get('user_instructions', ''),
            'engagement_techniques': base_config.get('engagement_techniques', []),
            'text_to_speech': {
                'default_tts_model': tts_model,
                'model': tts_base_config.get('model'),
                'default_voices': {
                    'question': voices.get('question', default_voices.get('question')),
                    'answer': voices.get('answer', default_voices.get('answer'))
                }
            }
        }

        # Merge configurations
        conversation_config = merge_configs(base_config, user_config)

        # Generate podcast
        result = generate_podcast(
            urls=data.urls,
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=False,
        )
        # Handle the result
        if isinstance(result, str) and os.path.isfile(result):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result, output_path)
            return {"audioUrl": f"/audio/{filename}"}
        elif hasattr(result, 'audio_path'):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result.audio_path, output_path)
            return {"audioUrl": f"/audio/{filename}"}
        else:
            raise HTTPException(status_code=500, detail="Invalid result format")

    except Exception as e:
        logger.error(f"Podcast generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/audio/{filename}")
def serve_audio(filename: str):
    """ Get File Audio From ther Server"""
    if not re.match(r'^[\w\-\.]+\.mp3$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    resolved = os.path.normpath(os.path.join(TEMP_DIR, filename))
    if not resolved.startswith(os.path.normpath(TEMP_DIR)):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(resolved)

@app.get("/health")
def healthcheck():
    return {"status": "healthy"}

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", DEFAULT_PORT))
    uvicorn.run(app, host=host, port=port)
