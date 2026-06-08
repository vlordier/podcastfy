import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import Runnable

from podcastfy.llm.base import LLMProvider
from podcastfy.llm.factory import LLMProviderFactory, detect_llm_provider


class TestLLMProviderABC(unittest.TestCase):
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMProvider()  # type: ignore[abstract]

    def test_detect_gemini(self):
        assert detect_llm_provider("gemini-2.5-flash") == "gemini"

    def test_detect_litellm(self):
        assert detect_llm_provider("gpt-4") == "litellm"
        assert detect_llm_provider("claude-3") == "litellm"

    def test_detect_llamafile(self):
        assert detect_llm_provider("llamafile") == "llamafile"
        assert detect_llm_provider("local") == "llamafile"

    def test_factory_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMProviderFactory.create("nonexistent", api_key="", model="x")


class TestGeminiLLM(unittest.TestCase):
    @patch("podcastfy.llm.providers.gemini.ChatGoogleGenerativeAI")
    def test_generate_called(self, mock_llm_class):
        mock_instance = MagicMock(spec=Runnable)
        mock_instance.invoke.return_value = "generated text"
        mock_llm_class.return_value = mock_instance

        provider = LLMProviderFactory.create("gemini", api_key="fake-key", model="gemini-2.5-flash")
        result = provider.generate("test prompt")
        assert result == "generated text"
        mock_instance.invoke.assert_called_once()


class TestLiteLLM(unittest.TestCase):
    @patch("podcastfy.llm.providers.litellm.ChatLiteLLM")
    def test_generate_called(self, mock_llm_class):
        mock_instance = MagicMock(spec=Runnable)
        mock_instance.invoke.return_value = "litellm response"
        mock_llm_class.return_value = mock_instance

        provider = LLMProviderFactory.create("litellm", api_key="fake-key", model="gpt-4")
        result = provider.generate("test prompt")
        assert result == "litellm response"


class TestLlamafileLLM(unittest.TestCase):
    @patch("podcastfy.llm.providers.llamafile.Llamafile")
    def test_generate_called(self, mock_llm_class):
        mock_instance = MagicMock(spec=Runnable)
        mock_instance.invoke.return_value = "llamafile response"
        mock_llm_class.return_value = mock_instance

        provider = LLMProviderFactory.create("llamafile", api_key="", model="local")
        result = provider.generate("test prompt")
        assert result == "llamafile response"


if __name__ == "__main__":
    unittest.main()
