import os

# Set the TTS_HOME environment variable to point to the local model directory
os.environ["TTS_HOME"] = "model"

# Import the TTS class
from TTS.api import TTS

# Initialize TTS with the same model_name (it will use the locally stored model)
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True, gpu=False)

# Define your text
text = """The central bank has decided to maintain current interest rates as inflation shows signs of easing. This decision reflects a cautious approach aimed at supporting economic recovery while keeping inflation under control. Financial markets responded positively, with gains led by banking and investment sectors. Moving forward, investors are closely watching for any signals about future policy shifts."""

# Choose an output file path
output_path = "coqui_output_male_slow_p251.wav"

# Synthesize speech with a male speaker (e.g., "p226") and slower speed
tts.tts_to_file(
    text=text,
    file_path=output_path,
    speaker="p233", # 
    speed=0.1  # < 1.0 is slower, > 1.0 is faster, 1.0 is normal
)

print(f"Coqui TTS: Speech saved as '{output_path}' with male voice ('p226'), slowed down.")
