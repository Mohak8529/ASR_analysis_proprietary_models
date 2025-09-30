import time
import threading
import queue
import signal
import sys
import numpy as np

from config import TRANSCRIPT_LOG, RECORDED_AUDIO_FILE
from audio.recorder import AudioRecorder
from audio.processor import audio_bytes_to_float32, save_raw_pcm_to_wav
from transcription.whisper_handler import WhisperHandler
from utils.helpers import save_transcript

# Global flags and objects
audio_queue = queue.Queue()  # to hold chunks for transcription
recorder = None
whisperer = None
running = True
transcription_thread = None

def transcriber_worker():
    """
    Worker thread to process audio chunks as they arrive in audio_queue.
    Runs Whisper transcription on each chunk sequentially.
    """
    while running or not audio_queue.empty():
        try:
            audio_chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_float32 = audio_bytes_to_float32(audio_chunk)
        text, _ = whisperer.transcribe(audio_float32)
        if text:
            print(f"[Transcript] {text}")
            save_transcript(text, TRANSCRIPT_LOG)
        audio_queue.task_done()

def signal_handler(sig, frame):
    global running, recorder
    print("\nStopping recording and transcription... Please wait.")
    running = False

    if recorder:
        recorder.stop()

    # Wait for transcription queue to empty
    audio_queue.join()

    # Save recorded audio fully
    raw_audio = recorder.get_all_recorded_frames()
    save_raw_pcm_to_wav(raw_audio, RECORDED_AUDIO_FILE)
    print(f"💾 Saved recorded audio to {RECORDED_AUDIO_FILE}")

    print(f"📄 Transcript saved to {TRANSCRIPT_LOG}")
    sys.exit(0)

def main():
    global recorder, whisperer, running, transcription_thread

    # Clear old transcript log before starting
    open(TRANSCRIPT_LOG, "w").close()

    # Initialize components
    recorder = AudioRecorder()
    whisperer = WhisperHandler()

    print("🎙 Starting real-time transcription...")
    recorder.start()

    # Start transcription thread
    transcription_thread = threading.Thread(target=transcriber_worker, daemon=True)
    transcription_thread.start()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running:
            frame = recorder.read()
            if frame:
                # Put audio data into queue for transcription worker
                audio_queue.put(frame)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        # Should trigger signal_handler
        pass

if __name__ == "__main__":
    main()
