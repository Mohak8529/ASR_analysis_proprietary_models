# server/config.py
import os

# Model settings
MODEL_PATH      = os.path.join("model", "large-v3.pt")
DEVICE          = "cpu"
LANGUAGE        = "en"
SHOW_TIMESTAMPS = True

# Audio settings
SAMPLE_RATE    = 16000
CHANNELS       = 1
CHUNK_DURATION = 1
CHUNK_SIZE     = SAMPLE_RATE * CHUNK_DURATION

# VAD settings
VAD_THRESHOLD = 0.2      # Threshold for speech detection (0.0-1.0)
MIN_SPEECH_DURATION = 0.2  # Minimum speech duration in seconds to trigger transcription
ENERGY_THRESHOLD = 50   # Minimum energy threshold for audio

# Output files
TRANSCRIPT_LOG   = "transcription_log.txt"
TRANSCRIPT_A_LOG = "transcriptionA.txt"
TRANSCRIPT_B_LOG = "transcriptionB.txt"
REC_AUDIO_FILE   = "session_recording.wav"
REC_AUDIO_A      = "session_recording_A.wav"
REC_AUDIO_B      = "session_recording_B.wav"

# WebSocket server
WS_HOST = "0.0.0.0"
WS_PORT = 8080

# Hallucination filtering
HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "don't forget to subscribe"
]