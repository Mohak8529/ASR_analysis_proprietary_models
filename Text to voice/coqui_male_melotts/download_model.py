import os
import sys

# Set the HF_HOME environment variable to store models in "model"
os.environ["HF_HOME"] = "model"

# Set MeCab dictionary path to unidic
os.environ["MECABRC"] = "/mnt/ssd1/male_coqui_melotts/myenv/lib/python3.11/site-packages/unidic/dicdir"

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

# Initialize TTS to trigger model download
# Options: 'EN' (English: EN-US, EN-BR, EN_INDIA, EN-AU, EN-Default), 'ZH' (Chinese), 'JP' (Japanese), 'KR' (Korean)
language = 'ZH'  # Change to 'EN', 'JP', or 'KR' as needed
try:
    tts = TTS(language=language, device='cpu')
except Exception as e:
    print(f"Error initializing TTS for language '{language}': {str(e)}")
    sys.exit(1)

# Verify model download location
model_path = os.path.join(os.environ["HF_HOME"], "hub", "models--myshell-ai--MeloTTS-Chinese")
print(f"Model downloaded to: {model_path}")