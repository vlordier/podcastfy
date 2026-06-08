"""
FastAPI implementation for Podcastify podcast generation service.

This module provides REST endpoints for podcast generation and audio serving,
with configuration management and temporary file handling.
"""

import logging
import os
import re
import shutil
import time

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from podcastfy.utils.config_conversation import ConversationConfigModel, load_conversation_config_model
from podcastfy.utils.constants import DEFAULT_PORT, MAX_URLS, TEMP_DIR_NAME, TEMP_FILE_MAX_AGE_SECONDS
from podcastfy.utils.enums import ApiKeyLabel

from ..client import generate_podcast

logger = logging.getLogger(__name__)


def load_base_config_model() -> ConversationConfigModel:
    try:
        return load_conversation_config_model()
    except Exception as e:
        return ConversationConfigModel()


class GenerateRequest(BaseModel):
    urls: list[str] = Field(default_factory=list, max_length=MAX_URLS)
    tts_model: str | None = Field(default=None, pattern=r"^(openai|elevenlabs|edge|gemini|geminimulti)?$")
    user_instructions: str | None = None
    creativity: float | None = Field(default=None, ge=0, le=2)
    openai_key: str | None = None
    google_key: str | None = None
    elevenlabs_key: str | None = None
    is_long_form: bool = False
    conversation_config: dict | None = None


async def verify_api_key(x_api_key: str | None = Header(None)):
    expected_key = os.getenv(ApiKeyLabel.PODCASTFY.value)
    if expected_key:
        if not x_api_key or x_api_key != expected_key:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or missing API key")
    return x_api_key


app = FastAPI()

TEMP_DIR = os.path.join(os.path.dirname(__file__), TEMP_DIR_NAME)


def _ensure_temp_dir() -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    return TEMP_DIR


@app.on_event("startup")
def cleanup_temp_files():
    _ensure_temp_dir()
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
            update_dict["creativity"] = data.creativity
        if data.user_instructions is not None:
            update_dict["user_instructions"] = data.user_instructions
        if data.tts_model is not None:
            update_dict["default_tts_model"] = data.tts_model

        # Apply overrides using Pydantic's model_copy
        conversation_config_model = base_model.model_copy(update=update_dict)

        # Determine TTS model for direct argument
        tts_model = data.tts_model or conversation_config_model.default_tts_model

        # Generate podcast
        result = generate_podcast(
            urls=data.urls,
            conversation_config=conversation_config_model,
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
    """Get File Audio From ther Server"""
    if not re.match(r"^[\w\-\.]+\.mp3$", filename):
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
