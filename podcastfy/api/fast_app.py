"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import FileResponse
import os
import shutil
import re
import time
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from ..client import generate_podcast
import uvicorn

from podcastfy.utils.constants import MAX_URLS, TEMP_FILE_MAX_AGE_SECONDS, DEFAULT_PORT, TEMP_DIR_NAME
from podcastfy.utils.enums import TTSProvider, ApiKeyLabel
from podcastfy.utils.config_conversation import ConversationConfigModel, load_conversation_config_model


logger = logging.getLogger(__name__)


def load_base_config_model() -> ConversationConfigModel:
    try:
        return load_conversation_config_model()
    except Exception as e:
        print(f"Warning: Could not load base config: {e}")
        return ConversationConfigModel()


class GenerateRequest(BaseModel):
    urls: List[str] = Field(default_factory=list, max_length=MAX_URLS)
    tts_model: Optional[str] = Field(default=None, pattern=r"^(openai|elevenlabs|edge|gemini|geminimulti)?$")
    user_instructions: Optional[str] = None
    creativity: Optional[float] = Field(default=None, ge=0, le=2)
    openai_key: Optional[str] = None
    google_key: Optional[str] = None
    elevenlabs_key: Optional[str] = None
    is_long_form: bool = False
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


@app.on_event("startup")
def cleanup_temp_files():
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
    try:
        # Set environment variables
        if data.openai_key:
            os.environ[ApiKeyLabel.OPENAI.value] = data.openai_key
        if data.google_key:
            os.environ[ApiKeyLabel.GEMINI.value] = data.google_key
        if data.elevenlabs_key:
            os.environ[ApiKeyLabel.ELEVENLABS.value] = data.elevenlabs_key

        # Load base conversation config
        if data.conversation_config:
            base_model = ConversationConfigModel(**data.conversation_config)
        else:
            base_model = load_base_config_model()

        # Build override dict from request fields
        update_dict = {}
        if data.creativity is not None:
            update_dict['creativity'] = data.creativity
        if data.user_instructions is not None:
            update_dict['user_instructions'] = data.user_instructions
        if data.tts_model is not None:
            update_dict['default_tts_model'] = data.tts_model

        # Apply overrides using Pydantic's model_copy
        conversation_config_model = base_model.model_copy(update=update_dict)

        # Determine TTS model for direct argument
        tts_model = data.tts_model or conversation_config_model.default_tts_model

        # Convert to dict for downstream compatibility
        conversation_config = conversation_config_model.model_dump()

        # Generate podcast
        result = generate_podcast(
            urls=data.urls,
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=data.is_long_form,
        )
        # Handle the result
        if isinstance(result, str) and os.path.isfile(result):
            filename = f"podcast_{os.urandom(8).hex()}.mp3"
            output_path = os.path.join(TEMP_DIR, filename)
            shutil.copy2(result, output_path)
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
