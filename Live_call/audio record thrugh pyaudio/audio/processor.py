import numpy as np
import wave
from config import SAMPLE_RATE, CHANNELS

def audio_bytes_to_float32(audio_bytes):
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    return audio_np

def save_raw_pcm_to_wav(raw_bytes, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
