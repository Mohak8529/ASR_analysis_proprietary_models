# server/transcription/whisper_handler.py
import whisper
import numpy as np
from ..config import MODEL_PATH, DEVICE, LANGUAGE, SHOW_TIMESTAMPS, HALLUCINATION_PHRASES

class WhisperHandler:
    def __init__(self):
        print(f"[Model] Loading Whisper from {MODEL_PATH} on {DEVICE}…")
        self.model = whisper.load_model(MODEL_PATH, device=DEVICE)
        print("[Model] Loaded.")

    def transcribe(self, audio_array):
        """Transcribe audio with improved hallucination filtering"""
        if len(audio_array) == 0:
            return "", None

        result = self.model.transcribe(
            audio_array,
            language=LANGUAGE,
            fp16=False,
            verbose=False,
            word_timestamps=False,
            condition_on_previous_text=False,  # Important for reducing hallucinations
            no_speech_threshold=0.6,           # Higher threshold for stricter silence detection
            logprob_threshold=-1.0,            # Filter low-confidence transcriptions
            suppress_tokens=[],                # Empty list to avoid default suppression issues
        )
        
        # Post-process to filter hallucinations
        text = result["text"].strip()
        filtered_text = self._filter_hallucinations(text)
        
        if SHOW_TIMESTAMPS:
            return filtered_text, result["segments"]
        return filtered_text, None
    
    def _filter_hallucinations(self, text: str) -> str:
        """Filter out common hallucination phrases"""
        if not text:
            return text
            
        text_lower = text.lower().strip()
        
        # Check for exact matches with common hallucination phrases
        for phrase in HALLUCINATION_PHRASES:
            if text_lower == phrase.lower():
                print(f"[Filter] Blocked hallucination: '{text}'")
                return ""
            
        return text
    
    