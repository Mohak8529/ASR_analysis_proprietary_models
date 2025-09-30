import threading
import queue
import signal
import sys
import asyncio
import time

from .config import TRANSCRIPT_LOG, RECORDED_AUDIO_FILE
from .audio.processor import audio_bytes_to_float32, save_raw_pcm_to_wav
from .transcription.whisper_handler import WhisperHandler
from .utils.helpers import save_transcript
from .ws_server import WebSocketAudioServer

audio_queue = queue.Queue()
all_raw_frames = []
running = True

def transcriber_worker(whisper_model):
    chunk_count = 0
    while running or not audio_queue.empty():
        try:
            chunk = audio_queue.get(timeout=0.5)
            chunk_count += 1
        except queue.Empty:
            continue

        start_time = time.time()
        audio_float32 = audio_bytes_to_float32(chunk)
        text, _ = whisper_model.transcribe(audio_float32)
        elapsed = time.time() - start_time

        label = text.strip() if text.strip() else "(no speech)"
        print(f"[Transcript #{chunk_count}] ({elapsed:.1f}s) {label}")
        save_transcript(f"[{time.strftime('%H:%M:%S')}] #{chunk_count}: {label}", TRANSCRIPT_LOG)

        sys.stdout.flush()
        audio_queue.task_done()

def signal_handler(sig, frame):
    global running
    print("\n[!] Stopping server...")
    running = False
    print("[!] Waiting for audio queue...")
    audio_queue.join()
    raw_audio = b''.join(all_raw_frames)
    save_raw_pcm_to_wav(raw_audio, RECORDED_AUDIO_FILE)
    print(f"💾 Audio saved to {RECORDED_AUDIO_FILE}")
    print(f"📄 Transcript saved to {TRANSCRIPT_LOG}")
    sys.exit(0)

async def main():
    whisperer = None
    open(TRANSCRIPT_LOG, "w").write(f"=== Session {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    print("Loading Whisper model...")
    whisperer = WhisperHandler()
    print("Whisper model loaded, starting server.")

    signal.signal(signal.SIGINT, signal_handler)

    t = threading.Thread(target=transcriber_worker, args=(whisperer,), daemon=True)
    t.start()

    ws = WebSocketAudioServer(audio_queue, all_raw_frames)
    await ws.run()

if __name__ == "__main__":
    asyncio.run(main())
