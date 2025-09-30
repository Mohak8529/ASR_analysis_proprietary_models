import os

# Set TTS_HOME to the desired storage location
tts_home = "/mnt/ssd1/coqui_female/model"

# Create the directory if it doesn't exist
os.makedirs(tts_home, exist_ok=True)

# Set the TTS_HOME environment variable
os.environ["TTS_HOME"] = tts_home

# Import the TTS class
from TTS.api import TTS

# Initialize TTS with the model name (this will download it if not already present)
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=False)

# Verify that the model has been downloaded
print(f"Model downloaded to: {os.path.join(tts_home, 'tts', 'tts_models', 'en', 'ljspeech', 'vits')}")