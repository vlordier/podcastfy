from enum import StrEnum


class TTSProvider(StrEnum):
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    EDGE = "edge"
    GEMINI = "gemini"
    GEMINI_MULTI = "geminimulti"


class LLMProvider(StrEnum):
    GEMINI = "gemini"
    LITELLM = "litellm"
    LLAMAFILE = "llamafile"


class AudioFormat(StrEnum):
    MP3 = "mp3"
    WAV = "wav"


class SpeakerTag(StrEnum):
    PERSON1 = "Person1"
    PERSON2 = "Person2"


class ApiKeyLabel(StrEnum):
    GEMINI = "GEMINI_API_KEY"
    OPENAI = "OPENAI_API_KEY"
    ELEVENLABS = "ELEVENLABS_API_KEY"
    PODCASTFY = "PODCASTFY_API_KEY"
