import threading
import queue
import signal
import sys
import asyncio

from .config import TRANSCRIPT_LOG, RECORDED_AUDIO_FILE
from .audio.processor import audio_bytes_to_float32, save_raw_pcm_to_wav
from .transcription.whisper_handler import WhisperHandler
from .utils.helpers import save_transcript
from .ws_server import WebSocketAudioServer

audio_queue = queue.Queue()
all_raw_frames = []
running = True
whisperer = None

def transcriber_worker(whisper_model):
    """Worker thread that processes audio chunks with loaded Whisper model"""
    while running or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        
        audio_float32 = audio_bytes_to_float32(chunk)
        text, _ = whisper_model.transcribe(audio_float32)
        if text:
            print(f"[Transcript] {text}")
            save_transcript(text, TRANSCRIPT_LOG)
        audio_queue.task_done()

def signal_handler(sig, frame):
    global running
    print("\n[!] Stopping server...")
    running = False
    audio_queue.join()  # Wait for all audio chunks to be processed
    raw_audio = b''.join(all_raw_frames)
    save_raw_pcm_to_wav(raw_audio, RECORDED_AUDIO_FILE)
    print(f"💾 Audio saved to {RECORDED_AUDIO_FILE}")
    print(f"📄 Transcript saved to {TRANSCRIPT_LOG}")
    sys.exit(0)

async def main():
    global whisperer
    
    # Clear old transcript log
    open(TRANSCRIPT_LOG, "w").close()
    
    # Load Whisper model FIRST before starting server
    print("Loading Whisper model...")
    whisperer = WhisperHandler()  # This loads the model synchronously
    print("Whisper model loaded, starting server.")
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start transcription worker with loaded model
    t_thread = threading.Thread(target=transcriber_worker, args=(whisperer,), daemon=True)
    t_thread.start()
    
    # Start WebSocket server
    ws_server = WebSocketAudioServer(audio_queue, all_raw_frames)
    await ws_server.run()

if __name__ == "__main__":
    asyncio.run(main())
