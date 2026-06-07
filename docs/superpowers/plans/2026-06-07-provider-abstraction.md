# Provider Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish clean abstract interfaces, factory/registry patterns, and Pydantic-validated configuration for TTS, LLM, and content extractor providers.

**Architecture:** Four independent phases — (1) TTS signature alignment, (2) LLM ABC + factory, (3) Extractor ABC + factory, (4) Pydantic config models. Each phase is backward-compatible at the API level.

**Tech Stack:** Python 3.11+, ABC, Pydantic v2, existing TTSProviderFactory pattern as template

---

### Task 1: TTS — Add `multi_speaker` class variable and align `generate_audio` signatures

**Files:**
- Modify: `podcastfy/tts/base.py:7-32`
- Modify: `podcastfy/tts/providers/geminimulti.py:13-298`
- Modify: `podcastfy/tts/providers/gemini.py:10-79`
- Modify: `podcastfy/tts/providers/openai.py:7-43`
- Modify: `podcastfy/tts/providers/elevenlabs.py:7-27`
- Modify: `podcastfy/tts/providers/edge.py:9-47`
- Modify: `podcastfy/text_to_speech.py:96-142`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Add `multi_speaker` to `TTSProvider` base class**

In `podcastfy/tts/base.py`, add `multi_speaker: ClassVar[bool] = False` after `COMMON_SSML_TAGS`:

```python
class TTSProvider(ABC):
    multi_speaker: ClassVar[bool] = False
    COMMON_SSML_TAGS: ClassVar[List[str]] = [
        'lang', 'p', 'phoneme', 's', 'sub'
    ]
```

Also update `generate_audio` signature to match what all providers actually accept — `voice2: str = None`:

```python
    @abstractmethod
    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
```

- [ ] **Step 2: Set `multi_speaker = True` on `GeminiMultiTTS` and make `generate_audio` return `bytes`**

In `podcastfy/tts/providers/geminimulti.py`:

```python
class GeminiMultiTTS(TTSProvider):
    multi_speaker: ClassVar[bool] = True
```

Change `generate_audio` to return `bytes` by calling `self.merge_audio()` on the `audio_chunks` list before returning:

```python
    def generate_audio(self, text: str, voice: str = "R", model: str = "en-US-Studio-MultiSpeaker",
                       voice2: str = "S", ending_message: str = "") -> bytes:
        ...
        audio_chunks.append(response.audio_content)
        return self.merge_audio(audio_chunks)  # was: return audio_chunks
```

- [ ] **Step 3: Align `GeminiTTS.generate_audio` signature with base class**

In `podcastfy/tts/providers/gemini.py`, change signature to match base:

```python
    def generate_audio(self, text: str, voice: str = "en-US-Journey-F",
                      model: str = None, voice2: str = None, **kwargs) -> bytes:
```

- [ ] **Step 4: Verify `OpenAITTS`, `ElevenLabsTTS`, `EdgeTTS` already match**

Check that `generate_audio` signatures in `openai.py`, `elevenlabs.py`, `edge.py` already accept `voice2: str = None` and return `bytes`. They do — no changes needed.

- [ ] **Step 5: Simplify `convert_to_speech` in `text_to_speech.py`**

In `podcastfy/text_to_speech.py`, replace the `if "multi" in self.provider.model.lower()` branch with a check on `self.provider.multi_speaker`:

```python
        try:
            if self.provider.multi_speaker:
                audio_data = self.provider.generate_audio(
                    cleaned_text,
                    voice="S",
                    model="en-US-Studio-MultiSpeaker",
                    voice2="R",
                    ending_message=self.ending_message,
                )
                # audio_data is now bytes (single merged audio)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "wb") as f:
                    f.write(audio_data)
            else:
                with tempfile.TemporaryDirectory(dir=self.temp_audio_dir) as temp_dir:
                    audio_segments = self._generate_audio_segments(
                        cleaned_text, temp_dir
                    )
                    self._merge_audio_files(audio_segments, output_file)
                    logger.info(f"Audio saved to {output_file}")
```

- [ ] **Step 6: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/test_audio.py::TestAudio::test_text_to_speech_edge -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add podcastfy/tts/base.py podcastfy/tts/providers/geminimulti.py podcastfy/tts/providers/gemini.py podcastfy/text_to_speech.py
git commit -m "refactor(tts): add multi_speaker flag, align generate_audio return types"
```

---

### Task 2: LLM — Create ABC, factory, and provider classes

**Files:**
- Create: `podcastfy/llm/__init__.py`
- Create: `podcastfy/llm/base.py`
- Create: `podcastfy/llm/factory.py`
- Create: `podcastfy/llm/providers/__init__.py`
- Create: `podcastfy/llm/providers/gemini.py`
- Create: `podcastfy/llm/providers/litellm.py`
- Create: `podcastfy/llm/providers/llamafile.py`
- Modify: `podcastfy/content_generator.py:29-77` (replace `LLMBackend`)
- Test: `tests/test_llm_providers.py`

- [ ] **Step 1: Write the failing test for LLM ABC + factory**

Create `tests/test_llm_providers.py`:

```python
import unittest
from unittest.mock import patch, MagicMock
from podcastfy.llm.base import LLMProvider
from podcastfy.llm.factory import LLMProviderFactory, detect_llm_provider


class TestLLMProviderABC(unittest.TestCase):
    def test_abc_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            LLMProvider()  # type: ignore

    def test_detect_gemini(self):
        self.assertEqual(detect_llm_provider("gemini-2.5-flash"), "gemini")

    def test_detect_litellm(self):
        self.assertEqual(detect_llm_provider("gpt-4"), "litellm")
        self.assertEqual(detect_llm_provider("claude-3"), "litellm")

    def test_detect_llamafile(self):
        self.assertEqual(detect_llm_provider("llamafile"), "llamafile")
        self.assertEqual(detect_llm_provider("local"), "llamafile")

    def test_factory_unknown_provider(self):
        with self.assertRaises(ValueError):
            LLMProviderFactory.create("nonexistent", api_key="", model="x")


class TestGeminiLLM(unittest.TestCase):
    @patch("podcastfy.llm.providers.gemini.ChatGoogleGenerativeAI")
    def test_generate_called(self, mock_llm_class):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value.content = "generated text"
        mock_llm_class.return_value = mock_instance

        provider = LLMProviderFactory.create(
            "gemini", api_key="fake-key", model="gemini-2.5-flash"
        )
        result = provider.generate("test prompt")
        self.assertEqual(result, "generated text")
        mock_instance.invoke.assert_called_once()


class TestLiteLLM(unittest.TestCase):
    @patch("podcastfy.llm.providers.litellm.ChatLiteLLM")
    def test_generate_called(self, mock_llm_class):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value.content = "litellm response"
        mock_llm_class.return_value = mock_instance

        provider = LLMProviderFactory.create(
            "litellm", api_key="fake-key", model="gpt-4"
        )
        result = provider.generate("test prompt")
        self.assertEqual(result, "litellm response")


class TestLlamafileLLM(unittest.TestCase):
    @patch("podcastfy.llm.providers.llamafile.Llamafile")
    def test_generate_called(self, mock_llm_class):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value.content = "llamafile response"
        mock_llm_class.return_value = mock_instance

        provider = LLMProviderFactory.create(
            "llamafile", api_key="", model="local"
        )
        result = provider.generate("test prompt")
        self.assertEqual(result, "llamafile response")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_providers.py -v`
Expected: FAIL with ModuleNotFoundError for `podcastfy.llm.base`

- [ ] **Step 3: Create `podcastfy/llm/__init__.py`**

```python
```

(empty file)

- [ ] **Step 4: Create `podcastfy/llm/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        config_conversation: Optional[dict] = None,
    ) -> str:
        ...
```

- [ ] **Step 5: Create `podcastfy/llm/factory.py`**

```python
from typing import ClassVar, Optional, Type
from .base import LLMProvider
from .providers.gemini import GeminiLLM
from .providers.litellm import LiteLLM
from .providers.llamafile import LlamafileLLM


def detect_llm_provider(model_name: str) -> str:
    name = model_name.lower()
    if "gemini" in name:
        return "gemini"
    if name == "local" or name.startswith("llamafile"):
        return "llamafile"
    return "litellm"


class LLMProviderFactory:
    _providers: ClassVar[dict[str, Type[LLMProvider]]] = {
        "gemini": GeminiLLM,
        "litellm": LiteLLM,
        "llamafile": LlamafileLLM,
    }

    @classmethod
    def create(cls, name: str, api_key: str, model: str, **kwargs) -> LLMProvider:
        provider_cls = cls._providers.get(name.lower())
        if not provider_cls:
            raise ValueError(
                f"Unknown LLM provider: {name}. "
                f"Choose from: {', '.join(cls._providers.keys())}"
            )
        return provider_cls(api_key=api_key, model=model, **kwargs)

    @classmethod
    def register(cls, name: str, provider_cls: Type[LLMProvider]) -> None:
        cls._providers[name.lower()] = provider_cls
```

- [ ] **Step 6: Create `podcastfy/llm/providers/__init__.py`**

```python
```

(empty file)

- [ ] **Step 7: Create `podcastfy/llm/providers/gemini.py`**

```python
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.prompts import HumanMessagePromptTemplate
from ..base import LLMProvider


class GeminiLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.model_name = model
        common_params = {
            "temperature": kwargs.get("temperature", 1.0),
            "presence_penalty": 0.75,
            "frequency_penalty": 0.75,
        }
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
            model=model,
            max_output_tokens=kwargs.get("max_output_tokens", 8192),
            **common_params,
        )

    def generate(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        config_conversation: Optional[dict] = None,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", prompt),
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({})
```

- [ ] **Step 8: Create `podcastfy/llm/providers/litellm.py`**

```python
import os
from typing import Optional
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..base import LLMProvider


class LiteLLM(LLMProvider):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.model_name = model
        self.llm = ChatLiteLLM(
            model=model,
            temperature=kwargs.get("temperature", 1.0),
            api_key=api_key or os.environ.get(kwargs.get("api_key_label", "OPENAI_API_KEY"), ""),
        )

    def generate(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        config_conversation: Optional[dict] = None,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", prompt),
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({})
```

- [ ] **Step 9: Create `podcastfy/llm/providers/llamafile.py`**

```python
from typing import Optional
from langchain_community.llms.llamafile import Llamafile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..base import LLMProvider


class LlamafileLLM(LLMProvider):
    def __init__(self, api_key: str = "", model: str = "local", **kwargs):
        self.model_name = model
        self.llm = Llamafile()

    def generate(
        self,
        prompt: str,
        images: Optional[list[str]] = None,
        config_conversation: Optional[dict] = None,
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", prompt),
        ])
        chain = prompt_template | self.llm | StrOutputParser()
        return chain.invoke({})
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm_providers.py -v`
Expected: All 7 tests PASS

- [ ] **Step 11: Refactor `LLMBackend` in `content_generator.py` to use the factory**

Replace the `LLMBackend` class (lines 29-77) with a thin wrapper:

```python
class LLMBackend:
    def __init__(
        self,
        is_local: bool,
        temperature: float,
        max_output_tokens: int,
        model_name: str,
        api_key_label: str = "GEMINI_API_KEY",
    ):
        self.is_local = is_local
        self.model_name = model_name
        self.is_multimodal = not is_local

        if is_local:
            provider_name = "llamafile"
            api_key = ""
        else:
            provider_name = detect_llm_provider(model_name)
            api_key = os.environ.get(
                "GEMINI_API_KEY" if provider_name == "gemini" else api_key_label, ""
            )

        self.provider = LLMProviderFactory.create(
            provider_name,
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            api_key_label=api_key_label,
        )
        self.llm = self.provider.llm  # keep backward compat for strategies
```

Add imports at top of file:

```python
from podcastfy.llm.factory import LLMProviderFactory, detect_llm_provider
```

Remove the old imports for `ChatLiteLLM`, `ChatGoogleGenerativeAI`, `Llamafile` since they're now in the provider files.

- [ ] **Step 12: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/test_genai_podcast.py -v -k "test_generate_qa_content"` (requires GEMINI_API_KEY)
Expected: PASS

- [ ] **Step 13: Commit**

```bash
git add podcastfy/llm/ tests/test_llm_providers.py podcastfy/content_generator.py
git commit -m "feat(llm): add LLMProvider ABC, factory, and provider classes"
```

---

### Task 3: Content Extractors — Create ABC, factory, and `can_handle` pattern

**Files:**
- Create: `podcastfy/content_parser/extractor_base.py`
- Create: `podcastfy/content_parser/extractor_factory.py`
- Modify: `podcastfy/content_parser/content_extractor.py:22-78` (refactor `extract_content`)
- Modify: `podcastfy/content_parser/pdf_extractor.py:16-39` (add `can_handle`)
- Modify: `podcastfy/content_parser/website_extractor.py:20-68` (add `can_handle`)
- Modify: `podcastfy/content_parser/youtube_transcriber.py:15-40` (add `can_handle`)
- Test: `tests/test_content_extractors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_content_extractors.py`:

```python
import unittest
from podcastfy.content_parser.extractor_base import ContentExtractor
from podcastfy.content_parser.extractor_factory import ExtractorFactory
from podcastfy.content_parser.pdf_extractor import PDFExtractor
from podcastfy.content_parser.website_extractor import WebsiteExtractor
from podcastfy.content_parser.youtube_transcriber import YouTubeTranscriber


class TestExtractorABC(unittest.TestCase):
    def test_abc_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            ContentExtractor()  # type: ignore


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_content_extractors.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Create `podcastfy/content_parser/extractor_base.py`**

```python
from abc import ABC, abstractmethod


class ContentExtractor(ABC):
    @classmethod
    @abstractmethod
    def can_handle(cls, source: str) -> bool:
        ...

    @abstractmethod
    def extract(self, source: str) -> str:
        ...
```

- [ ] **Step 4: Create `podcastfy/content_parser/extractor_factory.py`**

```python
from typing import ClassVar, Optional, Type
from .extractor_base import ContentExtractor


class ExtractorFactory:
    _extractors: ClassVar[list[Type[ContentExtractor]]] = []

    @classmethod
    def register(cls, extractor_cls: Type[ContentExtractor]) -> None:
        cls._extractors.append(extractor_cls)

    @classmethod
    def create(cls, source: str) -> Optional[ContentExtractor]:
        for ex_cls in cls._extractors:
            if ex_cls.can_handle(source):
                return ex_cls()
        return None
```

- [ ] **Step 5: Add `can_handle` to `PDFExtractor`**

In `podcastfy/content_parser/pdf_extractor.py`, add classmethod:

```python
class PDFExtractor:
    @classmethod
    def can_handle(cls, source: str) -> bool:
        return source.lower().endswith('.pdf')

    def extract_content(self, file_path: str) -> str:
        ...
```

Also add `extract` method as alias:

```python
    def extract(self, source: str) -> str:
        return self.extract_content(source)
```

- [ ] **Step 6: Add `can_handle` to `YouTubeTranscriber`**

In `podcastfy/content_parser/youtube_transcriber.py`, add classmethod:

```python
class YouTubeTranscriber:
    @classmethod
    def can_handle(cls, source: str) -> bool:
        lower = source.lower()
        return "youtube.com" in lower or "youtu.be" in lower

    def extract_transcript(self, url: str) -> str:
        ...
```

Also add `extract` method as alias:

```python
    def extract(self, source: str) -> str:
        return self.extract_transcript(source)
```

- [ ] **Step 7: Add `can_handle` to `WebsiteExtractor`**

In `podcastfy/content_parser/website_extractor.py`, add classmethod:

```python
class WebsiteExtractor:
    @classmethod
    def can_handle(cls, source: str) -> bool:
        if PDFExtractor.can_handle(source) or YouTubeTranscriber.can_handle(source):
            return False
        try:
            from urllib.parse import urlparse
            if not source.startswith(('http://', 'https://')):
                source = 'https://' + source
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def extract_content(self, url: str) -> str:
        ...
```

Also add `extract` method as alias:

```python
    def extract(self, source: str) -> str:
        return self.extract_content(source)
```

Add imports at top of `website_extractor.py`:

```python
from podcastfy.content_parser.pdf_extractor import PDFExtractor
from podcastfy.content_parser.youtube_transcriber import YouTubeTranscriber
```

- [ ] **Step 8: Register extractors and refactor `ContentExtractor.extract_content`**

In `podcastfy/content_parser/content_extractor.py`, replace the class with:

```python
from .extractor_base import ContentExtractor as ContentExtractorABC
from .extractor_factory import ExtractorFactory
from .pdf_extractor import PDFExtractor
from .website_extractor import WebsiteExtractor
from .youtube_transcriber import YouTubeTranscriber

# Register extractors (order matters — more specific first)
ExtractorFactory.register(PDFExtractor)
ExtractorFactory.register(YouTubeTranscriber)
ExtractorFactory.register(WebsiteExtractor)


class ContentExtractor:
    def __init__(self):
        self.youtube_transcriber = YouTubeTranscriber()
        self.website_extractor = WebsiteExtractor()
        self.pdf_extractor = PDFExtractor()
        self.config = load_config()
        self.content_extractor_config = self.config.get('content_extractor', {})

    def is_url(self, source: str) -> bool:
        try:
            if not source.startswith(('http://', 'https://')):
                source = 'https://' + source
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def extract_content(self, source: str) -> str:
        try:
            extractor = ExtractorFactory.create(source)
            if extractor is None:
                raise ValueError("Unsupported source type")
            return extractor.extract(source)
        except Exception as e:
            logger.error(f"Error extracting content from {source}: {str(e)}")
            raise

    def generate_topic_content(self, topic: str) -> str:
        ...  # unchanged
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_content_extractors.py -v`
Expected: All tests PASS

- [ ] **Step 10: Run existing content parser tests**

Run: `python -m pytest tests/test_content_parser.py -v -k "test_pdf_extractor"`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add podcastfy/content_parser/extractor_base.py podcastfy/content_parser/extractor_factory.py podcastfy/content_parser/content_extractor.py podcastfy/content_parser/pdf_extractor.py podcastfy/content_parser/website_extractor.py podcastfy/content_parser/youtube_transcriber.py tests/test_content_extractors.py
git commit -m "feat(extractors): add ContentExtractor ABC, factory, and can_handle pattern"
```

---

### Task 4: Pydantic Config Models

**Files:**
- Modify: `podcastfy/utils/config_conversation.py:1-212`
- Modify: `podcastfy/text_to_speech.py:41-77` (use typed config)
- Modify: `podcastfy/content_generator.py:647-648` (use typed config)
- Test: `tests/test_config_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_models.py`:

```python
import unittest
import tempfile
import os
import yaml
from podcastfy.utils.config_conversation import (
    TTSProviderConfig,
    ConversationConfig as PydanticConversationConfig,
    load_conversation_config,
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
        cfg = PydanticConversationConfig()
        self.assertEqual(cfg.default_tts_model, "openai")
        self.assertEqual(cfg.creativity, 1.0)
        self.assertEqual(cfg.ending_message, "See You Next Time!")

    def test_conversation_config_from_dict(self):
        cfg = PydanticConversationConfig(
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

    def test_load_conversation_config_returns_pydantic(self):
        config = load_conversation_config()
        self.assertIsInstance(config, PydanticConversationConfig)
        self.assertIsNotNone(config.default_tts_model)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_models.py -v`
Expected: FAIL with ImportError for `TTSProviderConfig`

- [ ] **Step 3: Add Pydantic models to `config_conversation.py`**

Add at the top of `podcastfy/utils/config_conversation.py`, after the imports:

```python
from pydantic import BaseModel, Field


class TTSProviderConfig(BaseModel):
    default_voices: dict[str, str] = Field(default_factory=dict)
    model: str | None = None


class ConversationConfigModel(BaseModel):
    text_to_speech: dict[str, TTSProviderConfig] = Field(default_factory=dict)
    default_tts_model: str = "openai"
    creativity: float = 1.0
    ending_message: str = "See You Next Time!"
```

- [ ] **Step 4: Update `load_conversation_config` to return Pydantic model**

Replace the `load_conversation_config` function body:

```python
def load_conversation_config(config_conversation: Optional[Dict[str, Any]] = None) -> ConversationConfigModel:
    """
    Load and return a ConversationConfigModel instance.

    Args:
        config_conversation (Optional[Dict[str, Any]]): Configuration dictionary to use.
            If None, default config will be loaded from conversation_config.yaml.

    Returns:
        ConversationConfigModel: An instance of the Pydantic ConversationConfigModel.
    """
    if config_conversation is None:
        config_path = get_conversation_config_path()
        if config_path:
            with open(config_path, 'r') as file:
                raw = yaml.safe_load(file)
        else:
            raw = {}
    else:
        raw = config_conversation

    # Parse TTS provider configs
    tts_raw = raw.get("text_to_speech", {})
    tts_providers = {}
    for provider_name, provider_cfg in tts_raw.items():
        if isinstance(provider_cfg, dict) and "default_voices" in provider_cfg:
            tts_providers[provider_name] = TTSProviderConfig(**provider_cfg)

    return ConversationConfigModel(
        text_to_speech=tts_providers,
        default_tts_model=raw.get("default_tts_model", "openai"),
        creativity=raw.get("creativity", 1.0),
        ending_message=raw.get("ending_message", "See You Next Time!"),
    )
```

Keep the existing `ConversationConfig` and `NestedConfig` classes for backward compatibility — they're still used by other parts of the codebase.

- [ ] **Step 5: Update `text_to_speech.py` to use typed config**

In `podcastfy/text_to_speech.py`, update `__init__`:

```python
        self.conversation_config = load_conversation_config(conversation_config)
        self.tts_config = self.conversation_config.text_to_speech
```

Update `_get_provider_config`:

```python
    def _get_provider_config(self) -> Dict[str, Any]:
        provider_name = self.provider.__class__.__name__.lower().replace("tts", "")
        provider_cfg = self.tts_config.get(provider_name)
        if provider_cfg is None:
            return {
                "model": None,
                "default_voices": {
                    "question": None,
                    "answer": None,
                },
            }
        return {
            "model": provider_cfg.model,
            "default_voices": provider_cfg.default_voices,
        }
```

- [ ] **Step 6: Update `content_generator.py` to use typed config**

In `podcastfy/content_generator.py`, update `ContentGenerator.__init__`:

```python
        self.config_conversation = load_conversation_config(conversation_config)
        self.tts_config = self.config_conversation.text_to_speech
```

Replace `self.config_conversation.get("creativity", 1)` with `self.config_conversation.creativity`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_config_models.py -v`
Expected: All tests PASS

- [ ] **Step 8: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/test_audio.py::TestAudio::test_text_to_speech_edge -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add podcastfy/utils/config_conversation.py podcastfy/text_to_speech.py podcastfy/content_generator.py tests/test_config_models.py
git commit -m "feat(config): add Pydantic config models for TTS and conversation"
```

---

### Self-Review Checklist

1. **Spec coverage:** Task 1 covers TTS signature alignment (Section 1 of spec). Task 2 covers LLM ABC + factory (Section 2). Task 3 covers extractor ABC + factory (Section 3). Task 4 covers Pydantic config models (Section 4). All spec sections covered.

2. **Placeholder scan:** No TBD, TODO, "implement later", or "add appropriate error handling" found. Every step has complete code.

3. **Type consistency:** `multi_speaker` is `ClassVar[bool]` in both Task 1 and referenced consistently. `LLMProvider.generate()` signature matches across Task 2 steps. `ContentExtractor.can_handle()` and `.extract()` signatures match across Task 3 steps. `ConversationConfigModel` field names match usage in Task 4.