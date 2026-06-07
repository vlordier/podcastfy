import unittest
import podcastfy.content_parser.content_extractor  # noqa: F401 - triggers extractor registration
from podcastfy.content_parser.extractor_base import ContentExtractor
from podcastfy.content_parser.extractor_factory import ExtractorFactory
from podcastfy.content_parser.pdf_extractor import PDFExtractor
from podcastfy.content_parser.website_extractor import WebsiteExtractor
from podcastfy.content_parser.youtube_transcriber import YouTubeTranscriber


class TestExtractorABC(unittest.TestCase):
    def test_abc_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            ContentExtractor()


class TestCanHandle(unittest.TestCase):
    def test_pdf_handles_pdf_extension(self):
        self.assertTrue(PDFExtractor.can_handle("document.pdf"))
        self.assertTrue(PDFExtractor.can_handle("/path/to/file.PDF"))

    def test_pdf_rejects_non_pdf(self):
        self.assertFalse(PDFExtractor.can_handle("document.txt"))
        self.assertFalse(PDFExtractor.can_handle("https://example.com"))

    def test_youtube_handles_youtube_urls(self):
        self.assertTrue(YouTubeTranscriber.can_handle("https://youtube.com/watch?v=abc123"))
        self.assertTrue(YouTubeTranscriber.can_handle("https://youtu.be/abc123"))

    def test_youtube_rejects_non_youtube(self):
        self.assertFalse(YouTubeTranscriber.can_handle("https://example.com"))

    def test_website_handles_urls(self):
        self.assertTrue(WebsiteExtractor.can_handle("https://example.com/page"))
        self.assertTrue(WebsiteExtractor.can_handle("example.com"))

    def test_website_rejects_pdf_and_youtube(self):
        self.assertFalse(WebsiteExtractor.can_handle("file.pdf"))
        self.assertFalse(WebsiteExtractor.can_handle("https://youtube.com/watch?v=abc"))


class TestFactory(unittest.TestCase):
    def test_factory_returns_pdf_extractor(self):
        extractor = ExtractorFactory.create("document.pdf")
        self.assertIsInstance(extractor, PDFExtractor)

    def test_factory_returns_youtube_extractor(self):
        extractor = ExtractorFactory.create("https://youtube.com/watch?v=abc123")
        self.assertIsInstance(extractor, YouTubeTranscriber)

    def test_factory_returns_website_extractor(self):
        extractor = ExtractorFactory.create("https://example.com")
        self.assertIsInstance(extractor, WebsiteExtractor)

    def test_factory_returns_none_for_unknown(self):
        extractor = ExtractorFactory.create("some_random_string")
        self.assertIsNone(extractor)


if __name__ == "__main__":
    unittest.main()