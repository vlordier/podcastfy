from .enums import TTSProvider

# --- TTS chunking ---
DEFAULT_CHUNK_BYTES: int = 1300
DEFAULT_TURN_CHARS: int = 500
MAX_TEXT_BYTES_BEFORE_CHUNKING: int = 5000

# --- Audio encoding ---
DEFAULT_BITRATE: str = "320k"
DEFAULT_CODEC: str = "libmp3lame"

# --- TTS model defaults ---
DEFAULT_TTS_MODEL: TTSProvider = TTSProvider.OPENAI
OPENAI_TTS_MODEL: str = "tts-1-hd"
ELEVENLABS_TTS_MODEL: str = "eleven_multilingual_v2"
GEMINI_TTS_VOICE: str = "en-US-Journey-F"
GEMINI_MULTI_TTS_MODEL: str = "en-US-Studio-MultiSpeaker"
GEMINI_MULTI_VOICE1: str = "R"
GEMINI_MULTI_VOICE2: str = "S"
GEMINI_MULTI_LANGUAGE: str = "en-US"

# --- LLM defaults ---
DEFAULT_GEMINI_LLM: str = "gemini-2.5-flash"
DEFAULT_MAX_OUTPUT_TOKENS: int = 8192
MIN_OUTPUT_TOKENS: int = 256
MAX_OUTPUT_TOKENS: int = 65536
DEFAULT_PRESENCE_PENALTY: float = 0.75
DEFAULT_FREQUENCY_PENALTY: float = 0.75

# --- Conversation defaults ---
DEFAULT_MAX_NUM_CHUNKS: int = 8
DEFAULT_MIN_CHUNK_SIZE: int = 600
DEFAULT_CREATIVITY: float = 1.0

# --- Network / API ---
DEFAULT_TIMEOUT_SECONDS: int = 10
POST_NAVIGATION_WAIT_MS: int = 500
TEMP_FILE_MAX_AGE_SECONDS: int = 3600
DEFAULT_PORT: int = 8080
MAX_URLS: int = 50

# --- File paths ---
DEFAULT_TRANSCRIPTS_DIR: str = "./data/transcripts"
DEFAULT_AUDIO_DIR: str = "./data/audio"
DEFAULT_TEMP_AUDIO_DIR: str = "data/audio/tmp/"
TEMP_DIR_NAME: str = "temp_audio"

# --- SSML ---
COMMON_SSML_TAGS: list[str] = ["lang", "p", "phoneme", "s", "sub"]

# --- Other ---
PREVIEW_CHARS: int = 500
MIN_AUDIO_FILE_BYTES: int = 1024
MIN_LONGFORM_CHARS: int = 1000