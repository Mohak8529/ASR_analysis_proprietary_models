import os

# Set the TTS_HOME environment variable to store models in "model/"
os.environ["TTS_HOME"] = "model"

# Import the TTS class
from TTS.api import TTS

# Initialize TTS with the desired model (this will download it if not already present)
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)

# Verify that the model has been downloaded
print("Model downloaded to:", os.path.join(os.environ["TTS_HOME"], "tts", "tts_models", "en", "vctk", "vits"))