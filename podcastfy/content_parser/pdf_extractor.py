"""
PDF Extractor Module

This module provides functionality to extract text content from PDF files.
It handles the reading of PDF files, text extraction, and normalization of
the extracted content, including handling of special characters and accents.
"""

import logging
import unicodedata

import pymupdf

from .extractor_base import ContentExtractor as ContentExtractorABC

logger = logging.getLogger(__name__)


class PDFExtractor(ContentExtractorABC):
    @classmethod
    def can_handle(cls, source: str) -> bool:
        return source.lower().endswith(".pdf")

    def extract(self, source: str) -> str:
        return self.extract_content(source)

    def extract_content(self, file_path: str) -> str:
        """
        Extract text content from a PDF file, handling foreign characters and special characters.
        Accents are removed from the text.

        Args:
                file_path (str): Path to the PDF file.

        Returns:
                str: Extracted text content with accents removed and properly handled characters.
        """
        try:
            doc = pymupdf.open(file_path)
            content = " ".join(page.get_text() for page in doc)
            doc.close()

            # Normalize the text to handle special characters and remove accents
            return unicodedata.normalize("NFKD", content)

        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            raise
