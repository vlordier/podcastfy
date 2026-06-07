"""
Content Extractor Module

This module provides functionality to extract content from various sources including
websites, YouTube videos, and PDF files. It serves as a central hub for content
extraction, delegating to specialized extractors based on the source type.
"""

import logging
from typing import List
from .youtube_transcriber import YouTubeTranscriber
from .website_extractor import WebsiteExtractor
from .pdf_extractor import PDFExtractor
from .extractor_factory import ExtractorFactory
from google import genai
from google.genai import types
from podcastfy.utils.constants import DEFAULT_GEMINI_LLM, PREVIEW_CHARS

# Register extractors (order matters — more specific first)
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
				raise ValueError("Unsupported source type")
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
			
			grounding_tool = types.Tool(
				google_search=types.GoogleSearch()
			)
			
			config = types.GenerateContentConfig(
				tools=[grounding_tool]
			)
			
			prompt = f"""Search the web for comprehensive, up-to-date information about {topic}. 
						Provide a detailed, well-structured overview covering:
						- Key concepts and definitions
						- Recent developments and trends
						- Important facts and statistics
						- Different perspectives or viewpoints
						
						Be thorough, accurate, and cite sources when relevant."""
			
			logger.info(f"Generating content with Google Search grounding for topic: {topic}")
			response = client.models.generate_content(
				model=DEFAULT_GEMINI_LLM,
				contents=prompt,
				config=config
			)
			
			return response.text
		except Exception as e:
			logger.error(f"Error generating content for topic '{topic}': {str(e)}")
			raise
		

def main(seed: int = 42) -> None:
	"""
	Main function to test the ContentExtractor class.
	"""
	logging.basicConfig(level=logging.INFO)

	# Create an instance of ContentExtractor
	extractor = ContentExtractor()

	# Test sources
	test_sources: List[str] = [
		"www.souzatharsis.com",
		"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
		"path/to/sample.pdf"
	]

	for source in test_sources:
		try:
			logger.info(f"Extracting content from: {source}")
			content = extractor.extract_content(source)

			# Print the first 500 characters of the extracted content
			logger.info(f"Extracted content (first {PREVIEW_CHARS} characters):\n{content[:PREVIEW_CHARS]}...")

			# Print the total length of the extracted content
			logger.info(f"Total length of extracted content: {len(content)} characters")
			logger.info("-" * 50)

		except Exception as e:
			logger.error(f"An error occurred while processing {source}: {str(e)}")

if __name__ == "__main__":
	main()
