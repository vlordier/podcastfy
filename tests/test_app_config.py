import unittest

import pytest

from podcastfy.utils.config import (
    AppConfigModel,
    ContentExtractorConfigModel,
    ContentGeneratorConfigModel,
    load_app_config_model,
)


class TestAppConfig(unittest.TestCase):
    def test_content_generator_defaults(self):
        cfg = ContentGeneratorConfigModel()
        assert cfg.llm_model == "gemini-2.5-flash"
        assert cfg.max_output_tokens == 8192

    def test_max_output_tokens_range(self):
        with pytest.raises(ValueError, match="max_output_tokens"):
            ContentGeneratorConfigModel(max_output_tokens=100)
        with pytest.raises(ValueError, match="max_output_tokens"):
            ContentGeneratorConfigModel(max_output_tokens=100000)

    def test_app_config_defaults(self):
        cfg = AppConfigModel()
        assert isinstance(cfg.content_generator, ContentGeneratorConfigModel)
        assert isinstance(cfg.content_extractor, ContentExtractorConfigModel)

    def test_load_app_config_model(self):
        cfg = load_app_config_model()
        assert isinstance(cfg, AppConfigModel)
        assert cfg.content_generator.llm_model is not None


if __name__ == "__main__":
    unittest.main()
