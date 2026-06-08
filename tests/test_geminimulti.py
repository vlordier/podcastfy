import unittest

from podcastfy.tts.providers.geminimulti import GeminiMultiTTS
from podcastfy.utils.constants import DEFAULT_CHUNK_BYTES, DEFAULT_TURN_CHARS


def _make_tts():
    return GeminiMultiTTS.__new__(GeminiMultiTTS)


class TestChunkText(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        tts = _make_tts()
        text = "<Person1>Hello</Person1>"
        chunks = tts.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "<Person1>Hello</Person1>"

    def test_text_longer_than_max_bytes_is_split(self):
        tts = _make_tts()
        content = "A" * (DEFAULT_CHUNK_BYTES // 2)
        text = f"<Person1>{content}</Person1><Person2>{content}</Person2><Person1>{content}</Person1>"
        chunks = tts.chunk_text(text, max_bytes=500)
        assert len(chunks) >= 2

    def test_chunks_dont_exceed_max_bytes(self):
        tts = _make_tts()
        content = "A" * 300
        text = f"<Person1>{content}</Person1><Person2>{content}</Person2>"
        max_bytes = 400
        chunks = tts.chunk_text(text, max_bytes=max_bytes)
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= max_bytes

    def test_single_section_within_limit(self):
        tts = _make_tts()
        text = "<Person1>Short text</Person1>"
        chunks = tts.chunk_text(text, max_bytes=1000)
        assert len(chunks) == 1
        assert "Short text" in chunks[0]

    def test_empty_text_returns_empty_list(self):
        tts = _make_tts()
        chunks = tts.chunk_text("")
        assert chunks == []


class TestSplitTurnText(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        tts = _make_tts()
        result = tts.split_turn_text("Hello world.")
        assert result == ["Hello world."]

    def test_text_split_at_sentence_boundaries(self):
        tts = _make_tts()
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = tts.split_turn_text(text, max_chars=25)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 25

    def test_respects_max_chars(self):
        tts = _make_tts()
        text = ("word " * (DEFAULT_TURN_CHARS * 3)).strip()
        result = tts.split_turn_text(text, max_chars=DEFAULT_TURN_CHARS)
        for chunk in result:
            assert len(chunk) <= DEFAULT_TURN_CHARS

    def test_empty_input_returns_empty_string(self):
        tts = _make_tts()
        result = tts.split_turn_text("")
        assert result == [""]

    def test_single_long_sentence_split_at_words(self):
        tts = _make_tts()
        text = ("word " * 200).strip()
        result = tts.split_turn_text(text, max_chars=50)
        for chunk in result:
            assert len(chunk) <= 50
        assert len(result) > 1
