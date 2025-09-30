# server/main.py
import threading
import queue
import signal
import sys
import asyncio
import time
import json
from .config import (
    TRANSCRIPT_LOG, TRANSCRIPT_A_LOG, TRANSCRIPT_B_LOG,
    REC_AUDIO_FILE, REC_AUDIO_A, REC_AUDIO_B,
    CHUNK_DURATION
)
from .audio.processor import audio_bytes_to_float32, save_raw_pcm_to_wav, has_sufficient_energy
from .audio.vad_processor import VADProcessor
from .transcription.whisper_handler import WhisperHandler
from .utils.helpers import save_transcript, is_meaningful_text
from .ws_server import WebSocketAudioServer

audio_queue     = queue.Queue()
all_raw_frames  = []      # List of (channel, bytes, ts)
running         = True

serialized_transcription = []
session_start_time = None  # Track when session begins

def group_consecutive_dialogues(entries):
    if not entries:
        return []
    grouped = []
    current = entries[0]
    for entry in entries[1:]:
        if entry['speaker'] == current['speaker']:
            # Merge dialogue and update end time
            current['dialogue'] += " " + entry['dialogue']
            current['endTime'] = entry['endTime']
        else:
            grouped.append(current)
            current = entry
    grouped.append(current)
    return grouped

def transcriber_worker(whisper_model, vad_processor):
    """Worker thread for processing audio chunks with VAD + Whisper"""
    global session_start_time
    chunk_count = 0
    vad_blocked_count = 0
    energy_blocked_count = 0
    hallucination_blocked_count = 0

    while running or not audio_queue.empty():
        try:
            ch, chunk, ts = audio_queue.get(timeout=0.5)
            chunk_count += 1

            # Set session start time from first chunk
            if session_start_time is None:
                session_start_time = ts

        except queue.Empty:
            continue
        start = time.time()

        # First check: Energy-based pre-filtering
        if not has_sufficient_energy(chunk):
            energy_blocked_count += 1
            timestamp = time.strftime("%H:%M:%S")
            log = f"[{timestamp}] ch={ch} #{chunk_count} (0.0s): [ENERGY] Low energy - skipped"
            print(log)
            save_transcript(log, TRANSCRIPT_LOG)
            if ch == "A":
                save_transcript(log, TRANSCRIPT_A_LOG)
            else:
                save_transcript(log, TRANSCRIPT_B_LOG)
            audio_queue.task_done()
            continue

        # Convert to float32 for VAD processing
        audio_f32 = audio_bytes_to_float32(chunk)

        # VAD check
        has_speech = vad_processor.has_speech(audio_f32)

        if not has_speech:
            vad_blocked_count += 1
            vad_latency = time.time() - start
            timestamp = time.strftime("%H:%M:%S")
            log = f"[{timestamp}] ch={ch} #{chunk_count} ({vad_latency:.1f}s): [VAD] No speech detected - skipped"
            print(log)
            save_transcript(log, TRANSCRIPT_LOG)
            if ch == "A":
                save_transcript(log, TRANSCRIPT_A_LOG)
            else:
                save_transcript(log, TRANSCRIPT_B_LOG)
            audio_queue.task_done()
            continue

        # Transcribe with Whisper
        text, segments = whisper_model.transcribe(audio_f32)
        latency = time.time() - start

        if not text or not is_meaningful_text(text):
            label = f"[FILTERED] {text}" if text else "[FILTERED] Empty"
        else:
            label = text.strip()
            # Add to serialized_transcription ONLY if not skipped/filtered
            # Calculate relative timestamps from session start
            relative_start_time = ts - session_start_time
            relative_end_time = relative_start_time + CHUNK_DURATION

            entry = {
                "dialogue": label,
                "speaker": f"ch{ch}",
                "startTime": round(relative_start_time, 1),
                "endTime": round(relative_end_time, 1)
            }
            serialized_transcription.append(entry)

        timestamp = time.strftime("%H:%M:%S")
        log = f"[{timestamp}] ch={ch} #{chunk_count} ({latency:.1f}s): {label}"
        print(log)
        save_transcript(log, TRANSCRIPT_LOG)
        if ch == "A":
            save_transcript(log, TRANSCRIPT_A_LOG)
        else:
            save_transcript(log, TRANSCRIPT_B_LOG)
        # Print statistics every 50 chunks
        if chunk_count % 50 == 0:
            print(f"[Stats] Processed: {chunk_count}, Energy blocked: {energy_blocked_count}, "
                  f"VAD blocked: {vad_blocked_count}, Hallucinations blocked: {hallucination_blocked_count}")
        audio_queue.task_done()

def signal_handler(sig, frame):
    global running
    print("\n[!] Stopping server – completing remaining work…")
    running = False
    audio_queue.join()

    # Split raw frames with timestamp and save WAVs (combine for compatibility)
    raw_all = b"".join(b for _, b, _ in all_raw_frames)
    raw_A   = b"".join(b for ch, b, _ in all_raw_frames if ch == "A")
    raw_B   = b"".join(b for ch, b, _ in all_raw_frames if ch == "B")

    save_raw_pcm_to_wav(raw_all, REC_AUDIO_FILE)
    save_raw_pcm_to_wav(raw_A,   REC_AUDIO_A)
    save_raw_pcm_to_wav(raw_B,   REC_AUDIO_B)

    print(f"💾 Saved {REC_AUDIO_FILE}, {REC_AUDIO_A}, {REC_AUDIO_B}")
    print(f"📄 Logs: {TRANSCRIPT_LOG}, {TRANSCRIPT_A_LOG}, {TRANSCRIPT_B_LOG}")

    # Group consecutive dialogues by channel to compact serialized output
    grouped_transcription = group_consecutive_dialogues(serialized_transcription)

    # Write grouped serialized transcription to file
    with open("serialized_transcription.json", "w", encoding="utf-8") as f:
        json.dump({"serializedTranscription": grouped_transcription}, f, indent=2)

    print(f"📄 Serialized transcription written to serialized_transcription.json")
    sys.exit(0)

async def main():
    # Initialize logs
    header = f"=== Session {time.strftime('%Y-%m-%d %H:%M:%S')} ==="
    for path in (TRANSCRIPT_LOG, TRANSCRIPT_A_LOG, TRANSCRIPT_B_LOG):
        open(path, "w").write(header + "\n")
    print("Loading VAD model…")
    vad_processor = VADProcessor()
    print("Loading Whisper model…")
    whisperer = WhisperHandler()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    worker = threading.Thread(
        target=transcriber_worker,
        args=(whisperer, vad_processor),
        daemon=True
    )
    worker.start()
    ws = WebSocketAudioServer(audio_queue, all_raw_frames)
    await ws.run()

if __name__ == "__main__":
    asyncio.run(main())
