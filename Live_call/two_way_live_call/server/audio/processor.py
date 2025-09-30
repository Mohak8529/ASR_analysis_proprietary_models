import numpy as np
import wave
from ..config import SAMPLE_RATE, CHANNELS

def audio_bytes_to_float32(audio_bytes: bytes):
    return np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

def save_raw_pcm_to_wav(raw_bytes: bytes, filename: str):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
