"""
YouTube Transcriber Module

This module is responsible for extracting and cleaning transcripts from YouTube videos.
It uses the YouTube Transcript API to fetch transcripts and provides functionality
to clean and format the extracted text.
"""

import logging

from youtube_transcript_api import YouTubeTranscriptApi

from podcastfy.utils.config import load_app_config_model

from .extractor_base import ContentExtractor as ContentExtractorABC

logger = logging.getLogger(__name__)


class YouTubeTranscriber(ContentExtractorABC):
    @classmethod
    def can_handle(cls, source: str) -> bool:
        lower = source.lower()
        return "youtube.com" in lower or "youtu.be" in lower

    def extract(self, source: str) -> str:
        return self.extract_transcript(source)

    def __init__(self):
        app_config = load_app_config_model()
        self.remove_phrases = app_config.youtube_transcriber.remove_phrases

    def extract_transcript(self, url: str) -> str:
        """
        Extract transcript from a YouTube video and remove '[music]' tags (case-insensitive).

        Args:
                url (str): YouTube video URL.

        Returns:
                str: Cleaned and extracted transcript.
        """
        try:
            video_id = url.rsplit("v=", maxsplit=1)[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(
                [entry["text"] for entry in transcript if entry["text"].lower() not in self.remove_phrases]
            )
        except Exception as e:
            logger.error(f"Error extracting YouTube transcript: {str(e)}")
            raise
