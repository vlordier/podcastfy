"""
Website Extractor Module

This module is responsible for extracting clean text content from websites using
Playwright to retrieve rendered HTML and BeautifulSoup for local parsing.
"""

import html
import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from podcastfy.content_parser.pdf_extractor import PDFExtractor
from podcastfy.content_parser.youtube_transcriber import YouTubeTranscriber
from podcastfy.utils.config import load_app_config_model
from podcastfy.utils.constants import POST_NAVIGATION_WAIT_MS

from .extractor_base import ContentExtractor as ContentExtractorABC

logger = logging.getLogger(__name__)


class WebsiteExtractor(ContentExtractorABC):
    @classmethod
    def can_handle(cls, source: str) -> bool:
        if PDFExtractor.can_handle(source) or YouTubeTranscriber.can_handle(source):
            return False
        try:
            if not source.startswith(("http://", "https://")):
                source = "https://" + source
            result = urlparse(source)
            return all([result.scheme, result.netloc]) and "." in result.netloc
        except ValueError:
            return False

    def extract(self, source: str) -> str:
        return self.extract_content(source)

    def __init__(self):
        """
        Initialize the WebsiteExtractor.
        """
        app_config = load_app_config_model()
        cfg = app_config.website_extractor
        self.unwanted_tags = cfg.unwanted_tags
        self.user_agent = cfg.user_agent
        self.timeout = cfg.timeout
        self.remove_patterns = cfg.markdown_cleaning.get("remove_patterns", [])

    def extract_content(self, url: str) -> str:
        """
        Extract clean text content from a website using BeautifulSoup.

        Args:
                url (str): Website URL.

        Returns:
                str: Extracted clean text content.

        Raises:
                Exception: If there's an error in extracting the content.
        """
        try:
            # Normalize the URL
            normalized_url = self.normalize_url(url)

            # Fetch the page HTML using Playwright (handles bot detection and JS rendering)
            html_content = self.fetch_with_playwright(normalized_url)

            # Parse the page content with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove unwanted elements
            self.remove_unwanted_elements(soup)

            # Extract and clean the text content
            raw_text = soup.get_text(separator="\n")  # Get all text content
            return self.clean_content(raw_text)

        except requests.RequestException as e:
            logger.error(f"Failed to extract content from {url}: {str(e)}")
            msg = f"Failed to extract content from {url}: {str(e)}"
            raise Exception(msg)
        except Exception as e:
            logger.error(f"An unexpected error occurred while extracting content from {url}: {str(e)}")
            raise

    def fetch_with_playwright(self, url: str) -> str:
        """
        Use Playwright to navigate to the URL and return the rendered HTML.

        Args:
                url (str): The URL to fetch.

        Returns:
                str: The page HTML after network is idle.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=self.user_agent,
                    ignore_https_errors=True,
                )
                page = context.new_page()
                # Extra headers to mimic a real browser
                page.set_extra_http_headers(
                    {
                        "Accept-Language": "en-US,en;q=0.9",
                    }
                )
                page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                # Optionally wait for DOM to be ready
                page.wait_for_timeout(POST_NAVIGATION_WAIT_MS)
                html_content = page.content()
                context.close()
                browser.close()
                return html_content
        except Exception as e:
            if "asyncio loop" in str(e).lower() or "async" in str(e).lower():
                return self.fetch_with_requests(url)
            msg = f"An unexpected error occurred while extracting content from {url}: {str(e)}"
            raise Exception(msg)

    def fetch_with_requests(self, url: str) -> str:
        """
        Fallback method using requests when Playwright fails in async contexts.
        """
        logger.warning(f"Playwright failed in async context, using requests: {url}")
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(url, headers=headers, timeout=self.timeout)
        return response.text

    def normalize_url(self, url: str) -> str:
        """
        Normalize the given URL by adding scheme if missing and ensuring it's a valid URL.

        Args:
                url (str): The URL to normalize.

        Returns:
                str: The normalized URL.

        Raises:
                ValueError: If the URL is invalid after normalization attempts.
        """
        # If the URL doesn't start with a scheme, add 'https://'
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Parse the URL
        parsed = urlparse(url)

        # Ensure the URL has a valid scheme and netloc
        if not all([parsed.scheme, parsed.netloc]):
            msg = f"Invalid URL: {url}"
            raise ValueError(msg)

        return parsed.geturl()

    def remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """
        Remove unwanted elements from the BeautifulSoup object.

        Args:
                soup (BeautifulSoup): The BeautifulSoup object to clean.
        """
        for tag in self.unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()

    def clean_content(self, content: str) -> str:
        """
        Clean the extracted content by removing unnecessary whitespace and applying
        custom cleaning patterns.

        Args:
                content (str): The content to clean.

        Returns:
                str: Cleaned text content.
        """
        # Decode HTML entities
        cleaned_content = html.unescape(content)

        # Remove extra whitespace
        cleaned_content = re.sub(r"\s+", " ", cleaned_content)

        # Remove extra newlines
        cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)

        # Apply custom cleaning patterns from config
        for pattern in self.remove_patterns:
            cleaned_content = re.sub(pattern, "", cleaned_content)

        return cleaned_content.strip()
