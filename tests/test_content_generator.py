import re
import unittest
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from podcastfy.content_generator import (
    ContentCleanerMixin,
    ContentGenerationStrategy,
    LLMBackend,
    PromptParams,
    StandardContentStrategy,
)
from podcastfy.utils.constants import COMMON_SSML_TAGS
from podcastfy.utils.enums import LLMProvider, SpeakerTag


class _FakeTTSProvider:
    multi_speaker = False

    def __new__(cls):
        return object.__new__(cls)

    def clean_tss_markup(self, input_text, additional_tags=None, supported_tags=None):
        if additional_tags is None:
            additional_tags = [SpeakerTag.PERSON1, SpeakerTag.PERSON2]
        if supported_tags is None:
            supported_tags = list(COMMON_SSML_TAGS)
        supported_tags.extend(additional_tags)
        pattern = r"</?(?:(?!" + "|".join(supported_tags) + r")\b)[^>]+>"
        cleaned_text = re.sub(pattern, "", input_text)
        cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
        additional_tag_values = [str(tag) for tag in additional_tags]
        for tag in additional_tag_values:
            cleaned_text = re.sub(
                f"<{tag}>(.*?)(?=<(?:{'|'.join(additional_tag_values)})>|$)",
                f"<{tag}>\\1</{tag}>",
                cleaned_text,
                flags=re.DOTALL,
            )
        return cleaned_text.strip()

    def generate_audio(self, text, voice, model, voice2=None):
        return b""


class TestPromptParams(unittest.TestCase):
    def test_default_construction(self):
        params = PromptParams()
        assert params.input_text == ""
        assert params.conversation_style == []
        assert params.roles_person1 == ""
        assert params.roles_person2 == ""
        assert params.dialogue_structure == []
        assert params.podcast_name == ""
        assert params.podcast_tagline == ""
        assert params.output_language == "English"
        assert params.user_instructions == ""
        assert params.engagement_techniques == []

    def test_conversation_style_defaults_to_empty_list(self):
        params = PromptParams()
        assert params.conversation_style == []

    def test_extra_forbid_rejects_unexpected_fields(self):
        with pytest.raises(ValidationError):
            PromptParams(unexpected_field="value")

    def test_extra_forbid_with_valid_fields(self):
        params = PromptParams(input_text="hello", output_language="French")
        assert params.input_text == "hello"
        assert params.output_language == "French"


class TestContentCleanerMixin(unittest.TestCase):
    @patch("podcastfy.content_generator.TTSProvider", _FakeTTSProvider)
    def test_supported_tags_preserved(self):
        text = "<speak>Hello world</speak>"
        result = ContentCleanerMixin._clean_tss_markup(text)  # noqa: SLF001
        assert "<speak>" in result
        assert "</speak>" in result
        assert "Hello world" in result

    @patch("podcastfy.content_generator.TTSProvider", _FakeTTSProvider)
    def test_lang_tag_preserved(self):
        text = "<lang>Bonjour</lang>"
        result = ContentCleanerMixin._clean_tss_markup(text)  # noqa: SLF001
        assert "<lang>" in result
        assert "Bonjour" in result

    @patch("podcastfy.content_generator.TTSProvider", _FakeTTSProvider)
    def test_unsupported_tags_removed(self):
        text = "<badtag>should be removed</badtag>"
        result = ContentCleanerMixin._clean_tss_markup(text)  # noqa: SLF001
        assert "<badtag>" not in result

    @patch("podcastfy.content_generator.TTSProvider", _FakeTTSProvider)
    def test_person1_person2_tags_preserved(self):
        text = "<Person1>Hello</Person1> <Person2>Hi</Person2>"
        result = ContentCleanerMixin._clean_tss_markup(text)  # noqa: SLF001
        assert "<Person1>" in result
        assert "<Person2>" in result
        assert "Hello" in result
        assert "Hi" in result

    @patch("podcastfy.content_generator.TTSProvider", _FakeTTSProvider)
    def test_closing_tags_added_when_missing(self):
        text = "<Person1>Hello"
        result = ContentCleanerMixin._clean_tss_markup(text)  # noqa: SLF001
        assert "<Person1>Hello</Person1>" in result

    @patch("podcastfy.content_generator.TTSProvider", _FakeTTSProvider)
    def test_empty_string_returns_empty(self):
        result = ContentCleanerMixin._clean_tss_markup("")  # noqa: SLF001
        assert result == ""


class TestLLMBackend(unittest.TestCase):
    @patch("podcastfy.content_generator.LLMProviderFactory.create")
    def test_local_creates_llamafile(self, mock_create):
        mock_provider = MagicMock()
        mock_provider.llm = "mock_llm"
        mock_create.return_value = mock_provider

        backend = LLMBackend(
            is_local=True,
            temperature=0.7,
            max_output_tokens=4096,
            model_name="local",
        )

        assert backend.is_local is True
        assert backend.is_multimodal is False
        mock_create.assert_called_once_with(
            LLMProvider.LLAMAFILE,
            api_key="",
            model="local",
            temperature=0.7,
            max_output_tokens=4096,
        )

    @patch("podcastfy.content_generator.detect_llm_provider")
    @patch("podcastfy.content_generator.LLMProviderFactory.create")
    def test_not_local_with_gemini(self, mock_create, mock_detect):
        mock_detect.return_value = LLMProvider.GEMINI
        mock_provider = MagicMock()
        mock_provider.llm = "mock_llm"
        mock_create.return_value = mock_provider

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            backend = LLMBackend(
                is_local=False,
                temperature=0.5,
                max_output_tokens=2048,
                model_name="gemini-2.5-flash",
            )

        assert backend.is_local is False
        assert backend.is_multimodal is True
        mock_create.assert_called_once_with(
            LLMProvider.GEMINI,
            api_key="test-key",
            model="gemini-2.5-flash",
            temperature=0.5,
            max_output_tokens=2048,
        )

    @patch("podcastfy.content_generator.LLMProviderFactory.create")
    def test_is_multimodal_true_when_not_local(self, mock_create):
        mock_provider = MagicMock()
        mock_provider.llm = "mock_llm"
        mock_create.return_value = mock_provider

        backend = LLMBackend(
            is_local=False,
            temperature=1.0,
            max_output_tokens=4096,
            model_name="gemini-2.5-flash",
        )
        assert backend.is_multimodal is True

    @patch("podcastfy.content_generator.LLMProviderFactory.create")
    def test_is_multimodal_false_when_local(self, mock_create):
        mock_provider = MagicMock()
        mock_provider.llm = "mock_llm"
        mock_create.return_value = mock_provider

        backend = LLMBackend(
            is_local=True,
            temperature=1.0,
            max_output_tokens=4096,
            model_name="local",
        )
        assert backend.is_multimodal is False


class TestContentGenerationStrategy(unittest.TestCase):
    def test_abstract_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ContentGenerationStrategy()

    def test_standard_strategy_can_be_instantiated(self):
        mock_llm = MagicMock()
        mock_cg_config = MagicMock()
        mock_config_conversation = MagicMock()
        strategy = StandardContentStrategy(mock_llm, mock_cg_config, mock_config_conversation)
        assert hasattr(strategy, "validate")
        assert hasattr(strategy, "generate")
        assert hasattr(strategy, "clean")
        assert hasattr(strategy, "compose_prompt_params")

    def test_standard_strategy_generate(self):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "generated"
        mock_llm = MagicMock()
        mock_cg_config = MagicMock()
        mock_config_conversation = MagicMock()
        strategy = StandardContentStrategy(mock_llm, mock_cg_config, mock_config_conversation)
        params = PromptParams(input_text="test")
        result = strategy.generate(mock_chain, "test", params)
        assert result == "generated"
        mock_chain.invoke.assert_called_once_with(params)
