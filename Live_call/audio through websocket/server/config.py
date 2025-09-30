import os

# Model Settings
MODEL_PATH = os.path.join("model", "large-v3.pt")
DEVICE = "cpu"
LANGUAGE = "en"
SHOW_TIMESTAMPS = True

# Audio Settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Output
TRANSCRIPT_LOG = "transcription_log.txt"
RECORDED_AUDIO_FILE = "session_recording.wav"

# WebSocket
WS_HOST = "0.0.0.0"
WS_PORT = 8080
