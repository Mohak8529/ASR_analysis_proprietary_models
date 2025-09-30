import whisper
from ..config import MODEL_PATH, DEVICE, LANGUAGE, SHOW_TIMESTAMPS

class WhisperHandler:
    def __init__(self):
        print(f"[Model] Loading Whisper from {MODEL_PATH} on {DEVICE}...")
        self.model = whisper.load_model(MODEL_PATH, device=DEVICE)
        print("[Model] Loaded.")

    def transcribe(self, audio_array):
        if len(audio_array) == 0:
            return None, None
        result = self.model.transcribe(audio_array, language=LANGUAGE, fp16=False)
        if SHOW_TIMESTAMPS:
            return result["text"], result["segments"]
        return result["text"], None
