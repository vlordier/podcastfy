"""
Content Extractor Module

This module provides functionality to extract content from various sources including
websites, YouTube videos, and PDF files. It serves as a central hub for content
extraction, delegating to specialized extractors based on the source type.
"""

import logging

from google import genai
from google.genai import types

from podcastfy.utils.constants import DEFAULT_GEMINI_LLM

from .extractor_factory import ExtractorFactory
from .pdf_extractor import PDFExtractor
from .website_extractor import WebsiteExtractor
from .youtube_transcriber import YouTubeTranscriber

# Register extractors (order matters — more specific first)
# These module-level calls are intentional: the factory pattern requires
# all extractor subclasses to be registered at import time so that
# ExtractorFactory.create() can dispatch to the correct handler.
ExtractorFactory.register(PDFExtractor)
ExtractorFactory.register(YouTubeTranscriber)
ExtractorFactory.register(WebsiteExtractor)

logger = logging.getLogger(__name__)


class ContentExtractor:
    def __init__(self):
        pass

    def extract_content(self, source: str) -> str:
        """
        Extract content from various sources.

        Args:
                source (str): URL or file path of the content source.

        Returns:
                str: Extracted text content.

        Raises:
                ValueError: If the source type is unsupported.
        """
        try:
            extractor = ExtractorFactory.create(source)
            if extractor is None:
                msg = "Unsupported source type"
                raise ValueError(msg)
            return extractor.extract(source)
        except Exception as e:
            logger.error(f"Error extracting content from {source}: {str(e)}")
            raise

    def generate_topic_content(self, topic: str) -> str:
        """
        Generate content based on a given topic using Gemini's Google Search grounding.

        Args:
                topic (str): The topic to generate content for.

        Returns:
                str: Generated content based on the topic.
        """
        try:
            client = genai.Client()

            grounding_tool = types.Tool(google_search=types.GoogleSearch())

            config = types.GenerateContentConfig(tools=[grounding_tool])

            prompt = f"""Search the web for comprehensive, up-to-date information about {topic}.
						Provide a detailed, well-structured overview covering:
						- Key concepts and definitions
						- Recent developments and trends
						- Important facts and statistics
						- Different perspectives or viewpoints
						
						Be thorough, accurate, and cite sources when relevant."""

            logger.info(f"Generating content with Google Search grounding for topic: {topic}")
            response = client.models.generate_content(model=DEFAULT_GEMINI_LLM, contents=prompt, config=config)

            return response.text
        except Exception as e:
            logger.error(f"Error generating content for topic '{topic}': {str(e)}")
            raise
