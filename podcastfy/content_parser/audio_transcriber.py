import logging
import os
import subprocess
import tempfile
import urllib.parse

from podcastfy.content_parser.extractor_base import ContentExtractor

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"}


class AudioTranscriber(ContentExtractor):
    def __init__(self, model: str = "base", batch_size: int = 12, language: str | None = None):
        self.model_name = model
        self.batch_size = batch_size
        self.language = language

    @classmethod
    def can_handle(cls, source: str) -> bool:
        ext = os.path.splitext(urllib.parse.urlparse(source).path)[1].lower()
        return ext in AUDIO_EXTENSIONS

    def extract(self, source: str) -> str:
        source_path = self._resolve_source(source)
        try:
            wav_path = self._convert_to_wav(source_path)
            return self._transcribe(wav_path)
        finally:
            if source_path != source:
                os.unlink(source_path)

    def _resolve_source(self, source: str) -> str:
        parsed = urllib.parse.urlparse(source)
        if parsed.scheme in ("http", "https"):
            import requests
            logger.info(f"Downloading audio from {source}")
            resp = requests.get(source, stream=True, timeout=300)
            resp.raise_for_status()
            ext = os.path.splitext(parsed.path)[1] or ".mp3"
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.close()
            return tmp.name
        return source

    def _convert_to_wav(self, audio_path: str) -> str:
        wav_path = audio_path.rsplit(".", 1)[0] + "_16k.wav"
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            "-sample_fmt", "s16",
            wav_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return wav_path

    def _transcribe(self, wav_path: str) -> str:
        from lightning_whisper_mlx import LightningWhisperMLX

        whisper = LightningWhisperMLX(
            model=self.model_name,
            batch_size=self.batch_size,
            quant=None,
        )
        logger.info(
            f"Transcribing with model={self.model_name} "
            f"batch_size={self.batch_size} language={self.language}"
        )
        result = whisper.transcribe(
            audio_path=wav_path,
            language=self.language,
        )
        return result["text"]