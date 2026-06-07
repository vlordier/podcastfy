import unittest
from podcastfy.utils.config import AppConfigModel, ContentGeneratorConfigModel, ContentExtractorConfigModel, load_app_config_model


class TestAppConfig(unittest.TestCase):
    def test_content_generator_defaults(self):
        cfg = ContentGeneratorConfigModel()
        self.assertEqual(cfg.llm_model, "gemini-2.5-flash")
        self.assertEqual(cfg.max_output_tokens, 8192)

    def test_max_output_tokens_range(self):
        with self.assertRaises(ValueError):
            ContentGeneratorConfigModel(max_output_tokens=100)
        with self.assertRaises(ValueError):
            ContentGeneratorConfigModel(max_output_tokens=100000)

    def test_app_config_defaults(self):
        cfg = AppConfigModel()
        self.assertIsInstance(cfg.content_generator, ContentGeneratorConfigModel)
        self.assertIsInstance(cfg.content_extractor, ContentExtractorConfigModel)

    def test_load_app_config_model(self):
        cfg = load_app_config_model()
        self.assertIsInstance(cfg, AppConfigModel)
        self.assertIsNotNone(cfg.content_generator.llm_model)


if __name__ == "__main__":
    unittest.main()
