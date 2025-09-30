import os

# Model Settings
MODEL_PATH = os.path.join("model", "large-v3.pt")  # Local model
DEVICE = "cpu"  # CPU only
LANGUAGE = "en"  # or None for autodetect

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 3  # seconds per chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Buffer max size (in seconds)
BUFFER_MAX_SIZE = 120

# Output paths
TRANSCRIPT_LOG = "transcription_log.txt"
RECORDED_AUDIO_FILE = "session_recording.wav"
SHOW_TIMESTAMPS = True  # <--- FIXED
