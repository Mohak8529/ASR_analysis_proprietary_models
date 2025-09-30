# server/audio/processor.py
import numpy as np
import wave
from ..config import SAMPLE_RATE, CHANNELS, ENERGY_THRESHOLD

def audio_bytes_to_float32(audio_bytes: bytes):
    """Convert audio bytes to float32 numpy array"""
    return np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0

def save_raw_pcm_to_wav(raw_bytes: bytes, filename: str):
    """Save raw PCM bytes to WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)

def calculate_audio_energy(audio_bytes: bytes):
    """Calculate RMS energy of audio chunk"""
    if len(audio_bytes) == 0:
        return 0.0
    
    # Convert to int16 array and calculate RMS
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
    return rms

def has_sufficient_energy(audio_bytes: bytes, threshold: float = ENERGY_THRESHOLD):
    """Check if audio chunk has sufficient energy to potentially contain speech"""
    energy = calculate_audio_energy(audio_bytes)
    return energy > threshold