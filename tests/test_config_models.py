import unittest
from podcastfy.utils.config_conversation import (
    TTSProviderConfig,
    ConversationConfigModel,
    load_conversation_config_model,
)


class TestPydanticConfigs(unittest.TestCase):
    def test_tts_provider_config_defaults(self):
        cfg = TTSProviderConfig()
        self.assertEqual(cfg.default_voices, {})
        self.assertIsNone(cfg.model)

    def test_tts_provider_config_with_values(self):
        cfg = TTSProviderConfig(
            default_voices={"question": "echo", "answer": "shimmer"},
            model="tts-1-hd",
        )
        self.assertEqual(cfg.default_voices["question"], "echo")
        self.assertEqual(cfg.model, "tts-1-hd")

    def test_conversation_config_defaults(self):
        cfg = ConversationConfigModel()
        self.assertEqual(cfg.default_tts_model, "openai")
        self.assertEqual(cfg.creativity, 1.0)
        self.assertEqual(cfg.ending_message, "See You Next Time!")

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
        self.assertEqual(cfg.default_tts_model, "edge")
        self.assertEqual(cfg.text_to_speech["edge"].default_voices["question"], "en-US-JennyNeural")

    def test_load_conversation_config_model(self):
        """Test loading from YAML via the convenience function."""
        cfg = load_conversation_config_model()
        self.assertIsInstance(cfg, ConversationConfigModel)
        self.assertIsNotNone(cfg.default_tts_model)

    def test_load_conversation_config_model_with_dict(self):
        """Test loading from a dict."""
        cfg = load_conversation_config_model({
            "default_tts_model": "elevenlabs",
            "text_to_speech": {
                "elevenlabs": {
                    "default_voices": {"question": "Chris", "answer": "Jessica"},
                    "model": "eleven_multilingual_v2",
                }
            }
        })
        self.assertEqual(cfg.default_tts_model, "elevenlabs")
        self.assertEqual(
            cfg.text_to_speech["elevenlabs"].default_voices["question"], "Chris"
        )


if __name__ == "__main__":
    unittest.main()