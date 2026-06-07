# Provider Abstraction Design

**Date:** 2026-06-07
**Status:** Approved
**Priority:** High

## Goals

Make it easy to add new TTS, LLM, and content extractor providers to podcastfy by establishing clean abstract interfaces, factory/registry patterns, and Pydantic-validated configuration.

## Non-goals

- Plugin auto-discovery (`importlib.metadata.entry_points`) — over-engineered for current needs
- Removing `llamafile` or `geminimulti` — keep existing functionality, fix interfaces
- Full config system rewrite — only add Pydantic validation where providers consume config

## 1. TTS Providers

### Current problems

| Problem | Impact |
|---------|--------|
| `GeminiMultiTTS.generate_audio()` returns `List[bytes]`, others return `bytes` | Forces `if "multi" in model.lower()` branch in `text_to_speech.py:96-98` |
| `GeminiTTS` and `GeminiMultiTTS` have different signatures from base | Can't rely on polymorphic calls |
| No standard way to declare multi-speaker capability | Caller must infer from model name string |

### Design

```python
class TTSProvider(ABC):
    multi_speaker: ClassVar[bool] = False

    @abstractmethod
    def generate_audio(
        self, text: str, voice: str = "default",
        voice2: str | None = None
    ) -> bytes:
        ...
```

- `generate_audio` always returns `bytes` — multi-speaker providers concatenate internally
- `multi_speaker` class variable replaces string-based inference
- Existing `TTSProviderFactory._providers` dict + `register_provider()` stays

### Changes

| File | Change |
|------|--------|
| `podcastfy/tts/base.py` | Add `multi_speaker: ClassVar[bool] = False` to `TTSProvider` |
| `podcastfy/tts/providers/geminimulti.py` | Set `multi_speaker = True`, concatenate `List[bytes]` → `bytes` in `generate_audio` |
| `podcastfy/tts/providers/gemini.py` | Align `generate_audio` signature with base class |
| `podcastfy/text_to_speech.py` | Remove `if "multi" in model.lower()` branch, use `provider.multi_speaker` if needed |

## 2. LLM Providers

### Current problems

| Problem | Impact |
|---------|--------|
| No abstract base class for LLMs | Adding a new LLM requires editing `LLMBackend.__init__()` hardcoded if/elif |
| `if "gemini" in model_name.lower()` special-cases Gemini | Bypasses LiteLLM which could handle it; duplicates logic |
| Llamafile hardcoded with no model selection | Assumes `http://localhost:8080`, no configurability |
| No factory/registry pattern | Unlike TTS, no standard way to register new LLM providers |

### Design

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: list[str] | None = None,
        config_conversation: dict | None = None,
    ) -> str:
        ...

class LLMProviderFactory:
    _providers: ClassVar[dict[str, type[LLMProvider]]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        cls._providers[name.lower()] = provider_cls

    @classmethod
    def create(cls, name: str, api_key: str,
               model: str, **kwargs) -> LLMProvider:
        provider_cls = cls._providers.get(name.lower())
        if not provider_cls:
            raise ValueError(f"Unknown LLM provider: {name}")
        return provider_cls(api_key=api_key, model=model, **kwargs)
```

### Initial providers

| Provider | Name | Backend |
|----------|------|---------|
| `GeminiLLM` | `"gemini"` | Wraps `ChatGoogleGenerativeAI` |
| `LiteLLM` | `"litellm"` | Wraps `ChatLiteLLM` (catches OpenAI, Anthropic, etc.) |
| `LlamafileLLM` | `"llamafile"` | Wraps hardcoded `Llamafile()` (local) |

### Detection logic

The LLM provider name is derived from the model string:
- `"gemini" in model_name.lower()` → `"gemini"`
- `model_name.lower().startswith("llamafile")` or `model_name == "local"` → `"llamafile"`
- Everything else → `"litellm"`

This detection lives in a single function `detect_llm_provider(model_name: str) -> str`. `LLMBackend` becomes thin orchestration that calls the factory.

### Changes

| File | Change |
|------|--------|
| `podcastfy/content_generator.py` | Add `LLMProvider` ABC + `LLMProviderFactory`; extract `LLMBackend` logic into provider classes; add `detect_llm_provider()` |
| New: `podcastfy/llm/providers/gemini.py` | `GeminiLLM` class |
| New: `podcastfy/llm/providers/litellm.py` | `LiteLLM` class |
| New: `podcastfy/llm/providers/llamafile.py` | `LlamafileLLM` class |
| New: `podcastfy/llm/__init__.py` | Package init, exports |

## 3. Content Extractors

### Current problems

| Problem | Impact |
|---------|--------|
| No abstract base class | No contract to implement; ad-hoc `extract_content()` methods |
| Selection via hardcoded if/elif chain | Adding new extractor = edit `ContentExtractor.extract_content()` + import |
| Pattern matching mixed with extraction logic | Single method does URL detection, extension check, and extraction |

### Design

```python
class ContentExtractor(ABC):
    @classmethod
    @abstractmethod
    def can_handle(cls, source: str) -> bool:
        """Return True if this extractor supports the source."""

    @abstractmethod
    def extract(self, source: str) -> str:
        ...

class ExtractorFactory:
    _extractors: ClassVar[list[type[ContentExtractor]]] = []

    @classmethod
    def register(cls, extractor_cls: type[ContentExtractor]) -> None:
        cls._extractors.append(extractor_cls)

    @classmethod
    def create(cls, source: str) -> ContentExtractor | None:
        for ex_cls in cls._extractors:
            if ex_cls.can_handle(source):
                return ex_cls()
        return None
```

Registration order matters — more specific matchers register first. `ContentExtractor.extract_content()` becomes three lines.

### Changes

| File | Change |
|------|--------|
| `podcastfy/content_parser/content_extractor.py` | Add `ContentExtractor` ABC + `ExtractorFactory`; refactor `extract_content()` to use factory; register existing extractors |
| `podcastfy/content_parser/pdf_extractor.py` | Implement `can_handle()` |
| `podcastfy/content_parser/website_extractor.py` | Implement `can_handle()` |
| `podcastfy/content_parser/youtube_transcriber.py` | Implement `can_handle()` |

## 4. Pydantic Config Models

### Current problems

| Problem | Impact |
|---------|--------|
| Config consumed as raw dicts | Every consumer does `dict.get("key", default)` — no validation, no IDE support |
| No schema for provider configs | Adding a new provider config key has no documentation or validation |
| API keys loaded with empty string defaults | Missing keys not caught until runtime error |

### Design

```python
class TTSProviderConfig(BaseModel):
    default_voices: dict[str, str] = Field(default_factory=dict)
    model: str | None = None

class ConversationConfig(BaseModel):
    text_to_speech: dict[str, TTSProviderConfig] = Field(default_factory=dict)
    default_tts_model: str = "openai"
    creativity: float = 1.0
    ending_message: str = "See You Next Time!"
```

`load_conversation_config()` returns a `ConversationConfig` instance instead of a raw dict. Consumers access typed fields.

### Changes

| File | Change |
|------|--------|
| `podcastfy/utils/config_conversation.py` | Add Pydantic models; parse YAML into models; return `ConversationConfig` instance |
| `podcastfy/text_to_speech.py` | Use `ConversationConfig.creativity`, `.ending_message`, `.text_to_speech[provider]` instead of `.get()` |
| `podcastfy/content_generator.py` | Use typed config fields where applicable |

## Migration Strategy

All changes are backward-compatible at the API level (CLI and FastAPI endpoints). Internal refactoring only.

| Phase | What | Risk |
|-------|------|------|
| 1 | TTS: fix signature, add multi_speaker, align return types | Low — interface-only, no behavior change |
| 2 | LLM: ABC + factory + provider classes | Medium — moves logic around, test thoroughly |
| 3 | Extractors: ABC + factory + can_handle | Low — predictable if/elif → loop |
| 4 | Pydantic configs: validate YAML through models | Low — additive, existing keys unchanged |

## Testing

- Each new ABC: test with a mock provider that confirms the contract
- TTS: verify all providers produce identical output before/after (bytes content match)
- LLM: verify provider selection logic maps correct model strings to provider classes
- Extractors: verify each `can_handle()` returns correct bool for known/unknown sources
- Config: verify `ConversationConfig` parses existing YAML without errors