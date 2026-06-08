import unittest

import pytest
from bs4 import BeautifulSoup

from podcastfy.content_parser.website_extractor import WebsiteExtractor


def _make_extractor(unwanted_tags=None, remove_patterns=None):
    ex = WebsiteExtractor.__new__(WebsiteExtractor)
    ex.unwanted_tags = unwanted_tags or ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ex.remove_patterns = remove_patterns or [
        r"!\[.*?\]\(.*?\)",
        r"\[([^\]]+)\]\([^\)]+\)",
        r"https?://\S+|www\.\S+",
    ]
    return ex


class TestWebsiteExtractorCanHandle(unittest.TestCase):
    def test_https_url_returns_true(self):
        assert WebsiteExtractor.can_handle("https://example.com/page") is True

    def test_http_url_returns_true(self):
        assert WebsiteExtractor.can_handle("http://example.com/page") is True

    def test_file_url_returns_false(self):
        assert WebsiteExtractor.can_handle("file:///tmp/doc.pdf") is False

    def test_pdf_extension_returns_false(self):
        assert WebsiteExtractor.can_handle("document.pdf") is False
        assert WebsiteExtractor.can_handle("/path/to/file.PDF") is False

    def test_youtube_url_returns_false(self):
        assert WebsiteExtractor.can_handle("https://youtube.com/watch?v=abc123") is False
        assert WebsiteExtractor.can_handle("https://youtu.be/abc123") is False

    def test_plain_domain_returns_true(self):
        assert WebsiteExtractor.can_handle("example.com") is True


class TestWebsiteExtractorNormalizeUrl(unittest.TestCase):
    def test_adds_https_scheme(self):
        ex = _make_extractor()
        result = ex.normalize_url("example.com")
        assert result == "https://example.com"

    def test_keeps_existing_https(self):
        ex = _make_extractor()
        result = ex.normalize_url("https://example.com")
        assert result == "https://example.com"

    def test_keeps_existing_http(self):
        ex = _make_extractor()
        result = ex.normalize_url("http://example.com")
        assert result == "http://example.com"

    def test_raises_on_invalid_url(self):
        ex = _make_extractor()
        with pytest.raises(ValueError, match="Invalid URL:"):
            ex.normalize_url("")


class TestWebsiteExtractorRemoveUnwantedElements(unittest.TestCase):
    def test_removes_unwanted_tags(self):
        ex = _make_extractor(unwanted_tags=["script", "style"])
        html = "<div>Hello</div><script>bad</script><style>.cls{}</style>"
        soup = BeautifulSoup(html, "html.parser")
        ex.remove_unwanted_elements(soup)
        result = soup.get_text(strip=True)
        assert result == "Hello"
        assert "bad" not in result
        assert ".cls" not in result

    def test_preserves_wanted_tags(self):
        ex = _make_extractor(unwanted_tags=["script"])
        html = "<div>Keep this</div><p>Also this</p>"
        soup = BeautifulSoup(html, "html.parser")
        ex.remove_unwanted_elements(soup)
        result = soup.get_text(strip=True)
        assert "Keep this" in result
        assert "Also this" in result

    def test_empty_html_no_error(self):
        ex = _make_extractor()
        soup = BeautifulSoup("", "html.parser")
        ex.remove_unwanted_elements(soup)
        result = soup.get_text(strip=True)
        assert result == ""


class TestWebsiteExtractorCleanContent(unittest.TestCase):
    def test_removes_unwanted_patterns(self):
        ex = _make_extractor(
            remove_patterns=[
                r"https?://\S+",
                r"!\[.*?\]\(.*?\)",
            ]
        )
        content = "Some text with https://example.com link and ![image](img.png)"
        result = ex.clean_content(content)
        assert "https://example.com" not in result
        assert "![image](img.png)" not in result

    def test_collapses_whitespace(self):
        ex = _make_extractor()
        content = "Hello     world\n\n\nnewlines"
        result = ex.clean_content(content)
        assert "Hello" in result
        assert "world" in result
        assert "newlines" in result

    def test_decodes_html_entities(self):
        ex = _make_extractor()
        content = "Hello &amp; world"
        result = ex.clean_content(content)
        assert "&amp;" not in result
        assert "&" in result

    def test_empty_content_returns_empty(self):
        ex = _make_extractor()
        result = ex.clean_content("")
        assert result == ""
