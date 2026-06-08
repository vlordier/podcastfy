import os
import unittest

from podcastfy.text_to_speech import TextToSpeech
from podcastfy.utils.constants import MIN_AUDIO_FILE_BYTES


class TestAudio(unittest.TestCase):
    def setUp(self):
        self.test_text = "<Person1>Hello, how are you?</Person1><Person2>I'm doing great, thanks for asking!</Person2>"
        self.output_dir = "tests/data/audio"
        os.makedirs(self.output_dir, exist_ok=True)

    @unittest.skip("Testing edge only on Github Action as it's free")
    def test_text_to_speech_openai(self):
        tts = TextToSpeech(model="openai")
        output_file = os.path.join(self.output_dir, "test_openai.mp3")
        tts.convert_to_speech(self.test_text, output_file)

        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > MIN_AUDIO_FILE_BYTES

        # Clean up
        os.remove(output_file)

    @unittest.skip("Testing edge only on Github Action as it's free")
    def test_text_to_speech_elevenlabs(self):
        tts = TextToSpeech(model="elevenlabs")
        output_file = os.path.join(self.output_dir, "test_elevenlabs.mp3")
        tts.convert_to_speech(self.test_text, output_file)

        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > MIN_AUDIO_FILE_BYTES

        # Clean up
        os.remove(output_file)

    def test_text_to_speech_edge(self):
        tts = TextToSpeech(model="edge")
        output_file = os.path.join(self.output_dir, "test_edge.mp3")
        tts.convert_to_speech(self.test_text, output_file)

        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > MIN_AUDIO_FILE_BYTES

        # Clean up
        os.remove(output_file)

    @unittest.skip("Testing edge only on Github Action as it's free")
    def test_text_to_speech_google(self):
        tts = TextToSpeech(model="gemini")
        output_file = os.path.join(self.output_dir, "test_google.mp3")
        tts.convert_to_speech(self.test_text, output_file)

        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > MIN_AUDIO_FILE_BYTES

        # Clean up
        os.remove(output_file)

    @unittest.skip("Testing edge only on Github Action as it's free")
    def test_text_to_speech_google_multi(self):
        tts = TextToSpeech(model="gemini_multi")
        output_file = os.path.join(self.output_dir, "test_google_multi.mp3")
        tts.convert_to_speech(self.test_text, output_file)

        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > MIN_AUDIO_FILE_BYTES

        # Clean up
        os.remove(output_file)


if __name__ == "__main__":
    unittest.main()
