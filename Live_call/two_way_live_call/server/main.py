import threading
import queue
import signal
import sys
import asyncio
import time

from .config import (
    TRANSCRIPT_LOG, TRANSCRIPT_A_LOG, TRANSCRIPT_B_LOG,
    REC_AUDIO_FILE, REC_AUDIO_A, REC_AUDIO_B
)
from .audio.processor import audio_bytes_to_float32, save_raw_pcm_to_wav
from .transcription.whisper_handler import WhisperHandler
from .utils.helpers import save_transcript
from .ws_server import WebSocketAudioServer

audio_queue     = queue.Queue()
all_raw_frames  = []      # List of (channel, bytes)
running         = True

def transcriber_worker(model):
    chunk_count = 0
    while running or not audio_queue.empty():
        try:
            ch, chunk = audio_queue.get(timeout=0.5)
            chunk_count += 1
        except queue.Empty:
            continue

        start = time.time()
        audio_f32 = audio_bytes_to_float32(chunk)
        text, _   = model.transcribe(audio_f32)
        latency   = time.time() - start

        label     = text.strip() or "(no speech)"
        timestamp = time.strftime("%H:%M:%S")
        log       = f"[{timestamp}] ch={ch} #{chunk_count} ({latency:.1f}s): {label}"

        print(log)
        save_transcript(log, TRANSCRIPT_LOG)
        if ch == "A":
            save_transcript(log, TRANSCRIPT_A_LOG)
        else:
            save_transcript(log, TRANSCRIPT_B_LOG)

        audio_queue.task_done()

def signal_handler(sig, frame):
    global running
    print("\n[!] Stopping server – completing remaining work…")
    running = False
    # Drain queue and finish transcription
    audio_queue.join()

    # Split raw frames and save WAVs
    raw_all = b"".join(b for _, b in all_raw_frames)
    raw_A   = b"".join(b for ch, b in all_raw_frames if ch == "A")
    raw_B   = b"".join(b for ch, b in all_raw_frames if ch == "B")

    save_raw_pcm_to_wav(raw_all, REC_AUDIO_FILE)
    save_raw_pcm_to_wav(raw_A,   REC_AUDIO_A)
    save_raw_pcm_to_wav(raw_B,   REC_AUDIO_B)

    print(f"💾 Saved {REC_AUDIO_FILE}, {REC_AUDIO_A}, {REC_AUDIO_B}")
    print(f"📄 Logs: {TRANSCRIPT_LOG}, {TRANSCRIPT_A_LOG}, {TRANSCRIPT_B_LOG}")
    sys.exit(0)

async def main():
    # Initialize logs
    header = f"=== Session {time.strftime('%Y-%m-%d %H:%M:%S')} ==="
    for path in (TRANSCRIPT_LOG, TRANSCRIPT_A_LOG, TRANSCRIPT_B_LOG):
        open(path, "w").write(header + "\n")

    print("Loading Whisper model…")
    whisperer = WhisperHandler()

    # Install signal handler that does full shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start transcription worker
    worker = threading.Thread(target=transcriber_worker, args=(whisperer,), daemon=True)
    worker.start()

    # Run WebSocket server (never returns until process exit)
    ws = WebSocketAudioServer(audio_queue, all_raw_frames)
    await ws.run()

if __name__ == "__main__":
    asyncio.run(main())
