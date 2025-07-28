import asyncio
import websockets
import json
import numpy as np
import whisper
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TTokenizer
import scipy.io.wavfile
import soundfile as sf
import os
import time
import signal
import sys
import gc
from typing import Dict, List
import base64
from concurrent.futures import ThreadPoolExecutor

# Environment settings
os.environ["NUMBA_CACHE_DIR"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
torch.cuda.is_available = lambda: False
device = "cpu"

# Configuration
SAMPLE_RATE = 48000  # Hz
OUTPUT_DIR = "live_output_api"
INTERMEDIATE_JSON = os.path.join(OUTPUT_DIR, "live_transcription.json")
MIC1_WAV = os.path.join(OUTPUT_DIR, "mic1.wav")
MIC2_WAV = os.path.join(OUTPUT_DIR, "mic2.wav")
FULL_CALL_WAV = os.path.join(OUTPUT_DIR, "full_call.wav")
WHISPER_MODEL_PATH = "model/large-v3.pt"
SEAMLESS_MODEL_PATH = "hf_cache/hub/models--facebook--seamless-m4t-v2-large/snapshots/5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
CHUNK_DURATION = 6.0  # Seconds per audio chunk
MIN_AUDIO_DURATION = 0.5  # Minimum seconds of audio to process
SERVER_HOST = "localhost"
SERVER_PORT = 8765

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
recording_mic1 = []
recording_mic2 = []
is_recording = False
segments_data = []
transcription_time = 0.0
overall_start_time = time.time()
mic1_overflows = 0
mic2_overflows = 0
connected_clients = set()
websocket_server = None
chunk_start_time = 0.0
last_process_time = None
all_mic1_chunks = []
all_mic2_chunks = []
mic1_processed_chunks = 0  # Track processed mic1 chunks
mic2_processed_chunks = 0  # Track processed mic2 chunks
executor = ThreadPoolExecutor(max_workers=1)

# Load models
print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device="cpu")
print("Loading SeamlessM4Tv2 model...")
tokenizer = SeamlessM4TTokenizer.from_pretrained(SEAMLESS_MODEL_PATH, use_fast=False, local_files_only=True)
processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL_PATH, tokenizer=tokenizer, local_files_only=True)
seamless_model = SeamlessM4Tv2Model.from_pretrained(SEAMLESS_MODEL_PATH, local_files_only=True).to(device)

def signal_handler(sig, frame):
    """Handle Ctrl+C to stop recording, save audio, and save results."""
    global is_recording, mic1_overflows, mic2_overflows, websocket_server
    if is_recording:
        print(f"\nStopping recording... (Mic1 overflows: {mic1_overflows}, Mic2 overflows: {mic2_overflows})")
        is_recording = False
        asyncio.run_coroutine_threadsafe(process_remaining_audio(), asyncio.get_event_loop())
        save_audio_files()
        save_transcription()
        if websocket_server:
            websocket_server.close()
        sys.exit(0)

def normalize_and_scale(audio):
    """Normalize and scale audio to int16 for Whisper and WAV saving."""
    if len(audio) == 0:
        return audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return (audio * 32767).astype(np.int16)

def save_audio_files():
    """Save recorded audio from mic1, mic2, and combined full call to WAV files."""
    global all_mic1_chunks, all_mic2_chunks
    try:
        if all_mic1_chunks:
            audio_mic1 = np.concatenate(all_mic1_chunks, axis=0).flatten()
            audio_mic1 = normalize_and_scale(audio_mic1)
            sf.write(MIC1_WAV, audio_mic1, SAMPLE_RATE)
            print(f"Saved mic1 audio to {MIC1_WAV} ({len(audio_mic1)/SAMPLE_RATE:.2f}s)")
        else:
            print("No audio recorded for mic1, skipping save")

        if all_mic2_chunks:
            audio_mic2 = np.concatenate(all_mic2_chunks, axis=0).flatten()
            audio_mic2 = normalize_and_scale(audio_mic2)
            sf.write(MIC2_WAV, audio_mic2, SAMPLE_RATE)
            print(f"Saved mic2 audio to {MIC2_WAV} ({len(audio_mic2)/SAMPLE_RATE:.2f}s)")
        else:
            print("No audio recorded for mic2, skipping save")

        if all_mic1_chunks and all_mic2_chunks:
            audio_mic1 = np.concatenate(all_mic1_chunks, axis=0).flatten()
            audio_mic2 = np.concatenate(all_mic2_chunks, axis=0).flatten()
            max_length = max(len(audio_mic1), len(audio_mic2))
            audio_mic1 = np.pad(audio_mic1, (0, max_length - len(audio_mic1)), mode='constant')
            audio_mic2 = np.pad(audio_mic2, (0, max_length - len(audio_mic2)), mode='constant')
            combined_audio = np.stack((audio_mic1, audio_mic2), axis=-1)
            combined_audio = normalize_and_scale(combined_audio)
            sf.write(FULL_CALL_WAV, combined_audio, SAMPLE_RATE)
            print(f"Saved combined audio to {FULL_CALL_WAV} ({max_length/SAMPLE_RATE:.2f}s)")
        else:
            print("Missing audio from one or both mics, skipping full call save")
    except Exception as e:
        print(f"Error saving audio files: {e}")

def transcribe_audio(temp_file, language="tl"):
    """Transcribe audio in a separate thread."""
    try:
        result = whisper_model.transcribe(temp_file, language=language, word_timestamps=True)
        return result
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

async def process_audio_chunk(audio_chunks, speaker, chunk_start_time):
    """Process an audio chunk: transcribe and translate."""
    global transcription_time
    if not audio_chunks:
        print(f"No audio data for {speaker}")
        return

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    audio_duration = len(audio) / SAMPLE_RATE
    if audio_duration < MIN_AUDIO_DURATION:
        print(f"Audio too short for {speaker}: {audio_duration:.2f}s, skipping")
        return
    
    print(f"Processing {speaker} audio: {audio_duration:.2f}s")
    audio = normalize_and_scale(audio)
    
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{speaker}.wav")
    scipy.io.wavfile.write(temp_file, SAMPLE_RATE, audio)
    
    transcribe_start = time.time()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, transcribe_audio, temp_file, "tl")
    transcribe_end = time.time()
    transcription_time += transcribe_end - transcribe_start
    
    if result is None:
        os.remove(temp_file)
        return
    
    for segment in result["segments"]:
        text = segment["text"].strip()
        if not text:
            print(f"Empty transcription for {speaker}, skipping")
            continue
        
        start = segment["start"] + chunk_start_time
        end = segment["end"] + chunk_start_time
        
        try:
            text_inputs = processor(text=text, src_lang="tgl", return_tensors="pt").to(device)
            output_tokens = seamless_model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
            translated_text = processor.decode(output_tokens[0].cpu().numpy().flatten().tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f"Translation error for {speaker}: {e}")
            continue
        
        segment_data = {
            "index": len(segments_data),
            "text": text,
            "start": start,
            "end": end,
            "speaker": speaker,
            "translated_text": translated_text
        }
        segments_data.append(segment_data)
        
        print(f"\n[{round(start, 2)} - {round(end, 2)}] ({speaker}): {text}")
        print(f"Translated: {translated_text}")
        
        await broadcast_transcription(segment_data)
    
    os.remove(temp_file)
    gc.collect()

async def process_remaining_audio():
    """Process all remaining audio chunks after stopping."""
    global all_mic1_chunks, all_mic2_chunks, chunk_start_time, mic1_processed_chunks, mic2_processed_chunks
    print(f"Processing remaining audio: mic1={len(all_mic1_chunks)} chunks, mic2={len(all_mic2_chunks)} chunks")
    
    # Process mic1 remaining chunks
    if all_mic1_chunks[mic1_processed_chunks:]:
        total_duration = len(np.concatenate(all_mic1_chunks[mic1_processed_chunks:], axis=0).flatten()) / SAMPLE_RATE
        current_time = chunk_start_time
        chunk_index = mic1_processed_chunks
        while chunk_index < len(all_mic1_chunks):
            chunk_audio = []
            chunk_duration = 0.0
            while chunk_index < len(all_mic1_chunks) and chunk_duration < CHUNK_DURATION:
                chunk_audio.append(all_mic1_chunks[chunk_index])
                chunk_duration += len(all_mic1_chunks[chunk_index]) / SAMPLE_RATE
                chunk_index += 1
            if chunk_audio:
                await process_audio_chunk(chunk_audio, "spk1", current_time)
            current_time += CHUNK_DURATION
        mic1_processed_chunks = len(all_mic1_chunks)  # Update processed count
    
    # Process mic2 remaining chunks
    if all_mic2_chunks[mic2_processed_chunks:]:
        total_duration = len(np.concatenate(all_mic2_chunks[mic2_processed_chunks:], axis=0).flatten()) / SAMPLE_RATE
        current_time = chunk_start_time
        chunk_index = mic2_processed_chunks
        while chunk_index < len(all_mic2_chunks):
            chunk_audio = []
            chunk_duration = 0.0
            while chunk_index < len(all_mic2_chunks) and chunk_duration < CHUNK_DURATION:
                chunk_audio.append(all_mic2_chunks[chunk_index])
                chunk_duration += len(all_mic2_chunks[chunk_index]) / SAMPLE_RATE
                chunk_index += 1
            if chunk_audio:
                await process_audio_chunk(chunk_audio, "spk2", current_time)
            current_time += CHUNK_DURATION
        mic2_processed_chunks = len(all_mic2_chunks)  # Update processed count

def save_transcription():
    """Save serialized transcription to JSON."""
    global mic1_overflows, mic2_overflows
    merged_segments = []
    current_segment = None
    
    for segment in sorted(segments_data, key=lambda x: x["start"]):
        current_speaker = segment["speaker"]
        dialogue = segment["text"]
        translated_dialogue = segment["translated_text"]
        
        if current_segment is None:
            current_segment = {
                "dialogue": dialogue,
                "translated_dialogue": translated_dialogue,
                "speaker": current_speaker,
                "startTime": round(segment["start"], 2),
                "endTime": round(segment["end"], 2)
            }
        else:
            if current_segment["speaker"] == current_speaker:
                current_segment["dialogue"] += " " + dialogue
                current_segment["translated_dialogue"] += " " + translated_dialogue
                current_segment["endTime"] = round(segment["end"], 2)
            else:
                merged_segments.append(current_segment)
                current_segment = {
                    "dialogue": dialogue,
                    "translated_dialogue": translated_dialogue,
                    "speaker": current_speaker,
                    "startTime": round(segment["start"], 2),
                    "endTime": round(segment["end"], 2)
                }
    
    if current_segment is not None:
        merged_segments.append(current_segment)
    
    output_data = {
        "timeTakenForTranscription": round(transcription_time, 2),
        "timeTakenForDiarization": 0.0,
        "serializedTranscription": merged_segments,
        "mic1_overflows": mic1_overflows,
        "mic2_overflows": mic2_overflows
    }
    
    try:
        with open(INTERMEDIATE_JSON, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSerialized transcription saved to {INTERMEDIATE_JSON}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
    
    print(f"Total transcription time: {transcription_time:.2f} seconds")
    print(f"Total process time: {time.time() - overall_start_time:.2f} seconds")
    print(f"Overflow counts: Mic1={mic1_overflows}, Mic2={mic2_overflows}")

async def broadcast_transcription(segment_data):
    """Broadcast transcription to all connected clients."""
    if not connected_clients:
        return
    
    message = {
        "type": "transcription",
        "data": {
            "speaker": segment_data["speaker"],
            "text": segment_data["text"],
            "translated_text": segment_data["translated_text"],
            "start": segment_data["start"],
            "end": segment_data["end"]
        }
    }
    
    disconnected_clients = []
    for client in connected_clients:
        try:
            await client.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            disconnected_clients.append(client)
    
    for client in disconnected_clients:
        connected_clients.discard(client)

async def handle_client(websocket):
    """Handle client connections and audio data."""
    global is_recording, recording_mic1, recording_mic2, chunk_start_time, last_process_time
    global mic1_overflows, mic2_overflows, all_mic1_chunks, all_mic2_chunks
    global mic1_processed_chunks, mic2_processed_chunks
    
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        await websocket.send(json.dumps({"type": "ready", "message": "Server ready for audio"}))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data["type"] == "start_recording":
                    if not is_recording:
                        print("Starting recording session...")
                        is_recording = True
                        chunk_start_time = 0.0
                        last_process_time = time.time()
                        overall_start_time = time.time()
                        all_mic1_chunks.clear()
                        all_mic2_chunks.clear()
                        recording_mic1.clear()
                        recording_mic2.clear()
                        mic1_processed_chunks = 0
                        mic2_processed_chunks = 0
                        await websocket.send(json.dumps({"type": "recording_started", "message": "Recording started. You can speak now."}))
                
                elif data["type"] == "audio_data":
                    if not is_recording:
                        continue
                        
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32).reshape(-1, 1)
                    chunk_id = data.get("chunk_id", 0)
                    
                    if data["speaker"] == "mic1":
                        recording_mic1.append(audio_data)
                        all_mic1_chunks.append(audio_data)
                        print(f"Received mic1 chunk {chunk_id} ({len(audio_data)/SAMPLE_RATE:.2f}s)")
                    elif data["speaker"] == "mic2":
                        recording_mic2.append(audio_data)
                        all_mic2_chunks.append(audio_data)
                        print(f"Received mic2 chunk {chunk_id} ({len(audio_data)/SAMPLE_RATE:.2f}s)")
                    
                    await websocket.send(json.dumps({"type": "ack", "speaker": data["speaker"], "chunk_id": chunk_id}))
                    
                    if "overflow" in data and data["overflow"]:
                        if data["speaker"] == "mic1":
                            mic1_overflows += 1
                        elif data["speaker"] == "mic2":
                            mic2_overflows += 1
                    
                    current_time = time.time()
                    if current_time - last_process_time >= CHUNK_DURATION:
                        chunks_mic1 = recording_mic1.copy()
                        chunks_mic2 = recording_mic2.copy()
                        recording_mic1.clear()
                        recording_mic2.clear()
                        
                        print(f"Processing chunk at {round(chunk_start_time, 2)}s: mic1={len(chunks_mic1)} chunks, mic2={len(chunks_mic2)} chunks")
                        if chunks_mic1:
                            await process_audio_chunk(chunks_mic1, "spk1", chunk_start_time)
                            mic1_processed_chunks += len(chunks_mic1)
                        if chunks_mic2:
                            await process_audio_chunk(chunks_mic2, "spk2", chunk_start_time)
                            mic2_processed_chunks += len(chunks_mic2)
                        
                        chunk_start_time += CHUNK_DURATION
                        last_process_time = current_time
                
                elif data["type"] == "stop_recording":
                    if is_recording:
                        print("Client requested to stop recording...")
                        is_recording = False
                        await process_remaining_audio()
                        save_audio_files()
                        save_transcription()
                        await websocket.send(json.dumps({"type": "recording_stopped", "message": "Recording completed and files saved."}))
                        
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Error processing message: {e}")
    
    except websockets.exceptions.ConnectionClosed as e:
        if e.code == 1000:
            print("Client disconnected normally")
        else:
            print(f"Client disconnected with error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"Client removed. Total clients: {len(connected_clients)}")
        
        if len(connected_clients) == 0 and is_recording:
            print("No clients connected, stopping recording...")
            is_recording = False
            await process_remaining_audio()
            save_audio_files()
            save_transcription()
            if websocket_server:
                websocket_server.close()

async def main():
    global websocket_server
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting WebSocket server on {SERVER_HOST}:{SERVER_PORT}")
    print("Start recording")
    print("Waiting for client connection...")
    
    websocket_server = await websockets.serve(
        handle_client, 
        SERVER_HOST, 
        SERVER_PORT,
        ping_interval=60,
        ping_timeout=30
    )
    
    try:
        await websocket_server.wait_closed()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        websocket_server.close()
        await websocket_server.wait_closed()
    finally:
        executor.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main())