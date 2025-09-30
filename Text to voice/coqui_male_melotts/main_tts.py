import os
import sys

# Set environment variables for offline mode
os.environ["HF_HOME"] = "model"  # Point to local model directory
os.environ["MECABRC"] = "/mnt/ssd1/male_coqui_melotts/myenv/lib/python3.11/site-packages/unidic/dicdir"  # MeCab dictionary path
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Enforce offline mode for transformers

# Preload Japanese tokenizer to avoid online request
try:
    from transformers import AutoTokenizer
    japanese_tokenizer_path = "/mnt/ssd1/male_coqui_melotts/model/hub/models--tohoku-nlp--bert-base-japanese-v3/snapshots/65243d6e5629b969c77309f217bd7b1a79d43c7e"
    AutoTokenizer.from_pretrained(japanese_tokenizer_path)
except Exception as e:
    print(f"Error preloading Japanese tokenizer: {str(e)}")
    sys.exit(1)

# Import the TTS class
try:
    from melo.api import TTS
except ModuleNotFoundError as e:
    print("Error: 'melotts' package is not installed. Run 'pip install git+https://github.com/myshell-ai/MeloTTS.git' and 'python3 -m unidic download'.")
    sys.exit(1)
except RuntimeError as e:
    print(f"Error initializing MeCab: {str(e)}")
    print("Ensure MeCab and unidic are installed: 'sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8' and 'python3 -m unidic download'.")
    print("Check if '/mnt/ssd1/male_coqui_melotts/myenv/lib/python3.11/site-packages/unidic/dicdir' contains dictionary files.")
    sys.exit(1)

# Initialize TTS with the Chinese language
language = 'ZH'
try:
    tts = TTS(language=language, device='cpu')
except Exception as e:
    print(f"Error initializing TTS for language '{language}': {str(e)}")
    sys.exit(1)

# Define simpler Chinese text to test
text = "你好，这是一个测试句子。"

# Debug: Print available speakers and text processing
print(f"Available speakers: {tts.hps.data.spk2id}")
from melo.text import cleaned_text_to_sequence
try:
    cleaned_text, _ = tts.cleaner(text, language=language)
    print(f"Cleaned text: {cleaned_text}")
except Exception as e:
    print(f"Error cleaning text: {str(e)}")
    sys.exit(1)

# Choose an output file path
output_path = "melotts_output_zh.wav"

# Synthesize speech
try:
    tts.tts_to_file(text, language, output_path, 0.9)  # Use positional arguments
    print(f"MeloTTS: Speech saved as '{output_path}' with {language} voice, slightly slowed speed.")
except Exception as e:
    print(f"Error synthesizing speech: {str(e)}")
    sys.exit(1)