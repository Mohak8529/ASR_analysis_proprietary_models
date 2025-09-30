# server/audio/vad_processor.py
import torch
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps
from ..config import SAMPLE_RATE, VAD_THRESHOLD, MIN_SPEECH_DURATION

class VADProcessor:
    """Voice Activity Detection processor using Silero VAD"""
    
    def __init__(self):
        print("[VAD] Loading Silero VAD model...")
        try:
            # Load Silero VAD model
            self.model = load_silero_vad()
            print("[VAD] Silero VAD loaded successfully")
        except Exception as e:
            print(f"[VAD] Error loading Silero VAD: {e}")
            print("[VAD] Falling back to energy-based detection")
            self.model = None
    
    def has_speech(self, audio_float32: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech using Silero VAD
        
        Args:
            audio_float32: Audio data as float32 numpy array
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        if self.model is None:
            # Fallback to simple energy-based detection
            return self._energy_based_detection(audio_float32)
        
        try:
            # Convert numpy array to torch tensor
            if len(audio_float32) == 0:
                return False
                
            # Ensure we have the right sample rate (16000 Hz)
            if len(audio_float32) < int(SAMPLE_RATE * 0.03):  # Minimum 30ms
                return False
                
            audio_tensor = torch.from_numpy(audio_float32)
            
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=100
            )
            
            # Check if any speech was detected
            if speech_timestamps:
                total_speech_duration = sum(
                    (segment['end'] - segment['start']) / SAMPLE_RATE 
                    for segment in speech_timestamps
                )
                return total_speech_duration >= MIN_SPEECH_DURATION
            
            return False
            
        except Exception as e:
            print(f"[VAD] Error during speech detection: {e}")
            # Fallback to energy-based detection
            return self._energy_based_detection(audio_float32)
    
    def _energy_based_detection(self, audio_float32: np.ndarray, threshold: float = 0.01) -> bool:
        """Fallback energy-based speech detection"""
        if len(audio_float32) == 0:
            return False
            
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_float32 ** 2))
        return rms > threshold
    
    def get_speech_segments(self, audio_float32: np.ndarray):
        """
        Get detailed speech segments from audio
        
        Args:
            audio_float32: Audio data as float32 numpy array
            
        Returns:
            list: List of speech segments with start and end timestamps
        """
        if self.model is None:
            return []
            
        try:
            if len(audio_float32) == 0:
                return []
                
            audio_tensor = torch.from_numpy(audio_float32)
            
            speech_timestamps = get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=100
            )
            
            return speech_timestamps
            
        except Exception as e:
            print(f"[VAD] Error getting speech segments: {e}")
            return []