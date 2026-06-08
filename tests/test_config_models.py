import unittest

import pytest

from podcastfy.utils.config_conversation import (
    ConversationConfigModel,
    OutputDirectories,
    TTSProviderConfig,
    load_conversation_config_model,
)


class TestPydanticConfigs(unittest.TestCase):
    def test_tts_provider_config_defaults(self):
        cfg = TTSProviderConfig()
        assert cfg.default_voices == {}
        assert cfg.model is None

    def test_tts_provider_config_with_values(self):
        cfg = TTSProviderConfig(
            default_voices={"question": "echo", "answer": "shimmer"},
            model="tts-1-hd",
        )
        assert cfg.default_voices["question"] == "echo"
        assert cfg.model == "tts-1-hd"

    def test_conversation_config_defaults(self):
        cfg = ConversationConfigModel()
        assert cfg.default_tts_model == "openai"
        assert cfg.creativity == 1.0
        assert cfg.ending_message == "See You Next Time!"

    def test_conversation_config_from_dict(self):
        cfg = ConversationConfigModel(
            default_tts_model="edge",
            creativity=0.5,
            ending_message="Goodbye!",
            text_to_speech={
                "edge": TTSProviderConfig(
                    default_voices={
                        "question": "en-US-JennyNeural",
                        "answer": "en-US-EricNeural",
                    }
                )
            },
        )
        assert cfg.default_tts_model == "edge"
        assert cfg.text_to_speech["edge"].default_voices["question"] == "en-US-JennyNeural"

    def test_load_conversation_config_model(self):
        """Test loading from YAML via the convenience function."""
        cfg = load_conversation_config_model()
        assert isinstance(cfg, ConversationConfigModel)
        assert cfg.default_tts_model is not None

    def test_load_conversation_config_model_with_dict(self):
        """Test loading from a dict."""
        cfg = load_conversation_config_model(
            {
                "default_tts_model": "elevenlabs",
                "text_to_speech": {
                    "elevenlabs": {
                        "default_voices": {"question": "Chris", "answer": "Jessica"},
                        "model": "eleven_multilingual_v2",
                    }
                },
            }
        )
        assert cfg.default_tts_model == "elevenlabs"
        assert cfg.text_to_speech["elevenlabs"].default_voices["question"] == "Chris"

    def test_creativity_range(self):
        with pytest.raises(ValueError):
            ConversationConfigModel(creativity=-1)
        with pytest.raises(ValueError):
            ConversationConfigModel(creativity=3)

    def test_max_num_chunks_range(self):
        with pytest.raises(ValueError):
            ConversationConfigModel(max_num_chunks=0)

    def test_conversation_style_empty_string(self):
        with pytest.raises(ValueError):
            ConversationConfigModel(conversation_style=["engaging", ""])

    def test_engagement_techniques_empty_string(self):
        with pytest.raises(ValueError):
            ConversationConfigModel(engagement_techniques=["humor", ""])

    def test_load_conversation_config_model_populates_all(self):
        cfg = load_conversation_config_model()
        assert cfg.podcast_name is not None
        assert cfg.roles_person1 is not None
        assert isinstance(cfg.conversation_style, list)
        assert isinstance(cfg.output_directories, OutputDirectories)
        assert len(cfg.conversation_style) > 0


if __name__ == "__main__":
    unittest.main()
