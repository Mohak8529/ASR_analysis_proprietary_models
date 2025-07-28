import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TTokenizer
import scipy.io.wavfile
import os
import json
import time
import signal
import sys
import gc

# Environment settings
os.environ["NUMBA_CACHE_DIR"] = ""
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
torch.cuda.is_available = lambda: False
device = "cpu"

# Configuration
SAMPLE_RATE = 48000  # Hz
CHANNELS = 1  # Mono
MIC1_DEVICE_INDEX = 11  # AB13X USB Audio (you)
MIC2_DEVICE_INDEX = 12  # JBL TUNE 305C USB-C (friend)
OUTPUT_DIR = "live_output"
INTERMEDIATE_JSON = os.path.join(OUTPUT_DIR, "live_transcription.json")
MIC1_WAV = os.path.join(OUTPUT_DIR, "mic1.wav")
MIC2_WAV = os.path.join(OUTPUT_DIR, "mic2.wav")
WHISPER_MODEL_PATH = "model/large-v3.pt"
SEAMLESS_MODEL_PATH = "hf_cache/hub/models--facebook--seamless-m4t-v2-large/snapshots/5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
CHUNK_DURATION = 6.0  # Seconds per audio chunk
BLOCKSIZE = 4096  # Blocksize for buffering
MIN_AUDIO_DURATION = 0.5  # Minimum seconds of audio to process

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables
recording_mic1 = []
recording_mic2 = []
is_recording = False
segments_data = []
transcription_time = 0.0
overall_start_time = time.time()

# Load models
print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device="cpu")
print("Loading SeamlessM4Tv2 model...")
tokenizer = SeamlessM4TTokenizer.from_pretrained(SEAMLESS_MODEL_PATH, use_fast=False, local_files_only=True)
processor = AutoProcessor.from_pretrained(SEAMLESS_MODEL_PATH, tokenizer=tokenizer, local_files_only=True)
seamless_model = SeamlessM4Tv2Model.from_pretrained(SEAMLESS_MODEL_PATH, local_files_only=True).to(device)

def signal_handler(sig, frame):
    """Handle Ctrl+C to stop recording, save audio, and save results."""
    global is_recording
    if is_recording:
        print("\nStopping recording...")
        is_recording = False
        # Process any remaining audio for transcription
        process_remaining_audio()
        # Save audio files
        save_audio_files()
        # Save serialized transcription
        save_transcription()
        sys.exit(0)

def callback_mic1(indata, frames, time, status):
    """Callback for microphone 1."""
    global recording_mic1
    if status:
        print(f"Mic1 status: {status}")
    if is_recording:
        recording_mic1.append(indata.copy())

def callback_mic2(indata, frames, time, status):
    """Callback for microphone 2."""
    global recording_mic2
    if status:
        print(f"Mic2 status: {status}")
    if is_recording:
        recording_mic2.append(indata.copy())

def normalize_and_scale(audio):
    """Normalize and scale audio to int16 for Whisper and WAV saving."""
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return (audio * 32767).astype(np.int16)

def save_audio_files():
    """Save recorded audio from mic1 and mic2 to WAV files."""
    global recording_mic1, recording_mic2
    try:
        # Save mic1 audio
        if recording_mic1:
            audio_mic1 = np.concatenate(recording_mic1, axis=0).flatten()
            audio_mic1 = normalize_and_scale(audio_mic1)
            sf.write(MIC1_WAV, audio_mic1, SAMPLE_RATE)
            print(f"Saved mic1 audio to {MIC1_WAV} ({len(audio_mic1)/SAMPLE_RATE:.2f}s)")
        else:
            print("No audio recorded for mic1, skipping save")

        # Save mic2 audio
        if recording_mic2:
            audio_mic2 = np.concatenate(recording_mic2, axis=0).flatten()
            audio_mic2 = normalize_and_scale(audio_mic2)
            sf.write(MIC2_WAV, audio_mic2, SAMPLE_RATE)
            print(f"Saved mic2 audio to {MIC2_WAV} ({len(audio_mic2)/SAMPLE_RATE:.2f}s)")
        else:
            print("No audio recorded for mic2, skipping save")
    except Exception as e:
        print(f"Error saving audio files: {e}")

def process_audio_chunk(audio_chunks, speaker, chunk_start_time):
    """Process an audio chunk: transcribe and translate."""
    global transcription_time
    if not audio_chunks:
        print(f"No audio data for {speaker}")
        return

    # Concatenate chunks
    audio = np.concatenate(audio_chunks, axis=0).flatten()
    
    # Check audio duration
    audio_duration = len(audio) / SAMPLE_RATE
    if audio_duration < MIN_AUDIO_DURATION:
        print(f"Audio too short for {speaker}: {audio_duration:.2f}s, skipping")
        return
    
    print(f"Processing {speaker} audio: {audio_duration:.2f}s")
    audio = normalize_and_scale(audio)
    
    # Save to temporary WAV file
    temp_file = os.path.join(OUTPUT_DIR, f"temp_{speaker}.wav")
    scipy.io.wavfile.write(temp_file, SAMPLE_RATE, audio)
    
    # Transcribe
    transcribe_start = time.time()
    try:
        result = whisper_model.transcribe(temp_file, language="tl", word_timestamps=True)
    except Exception as e:
        print(f"Transcription error for {speaker}: {e}")
        os.remove(temp_file)
        return
    transcribe_end = time.time()
    transcription_time += transcribe_end - transcribe_start
    
    # Process transcription
    for segment in result["segments"]:
        text = segment["text"].strip()
        if not text:
            print(f"Empty transcription for {speaker}, skipping")
            continue
        
        # Adjust timestamps to global timeline
        start = segment["start"] + chunk_start_time
        end = segment["end"] + chunk_start_time
        
        # Translate
        try:
            text_inputs = processor(text=text, src_lang="tgl", return_tensors="pt").to(device)
            output_tokens = seamless_model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
            translated_text = processor.decode(output_tokens[0].cpu().numpy().flatten().tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f"Translation error for {speaker}: {e}")
            continue
        
        # Store segment
        segment_data = {
            "index": len(segments_data),
            "text": text,
            "start": start,
            "end": end,
            "speaker": speaker,
            "translated_text": translated_text
        }
        segments_data.append(segment_data)
        
        # Display immediately
        print(f"\n[{round(start, 2)} - {round(end, 2)}] ({speaker}): {text}")
        print(f"Translated: {translated_text}")
    
    # Clean up
    os.remove(temp_file)
    gc.collect()

def process_remaining_audio():
    """Process any remaining audio chunks after stopping."""
    global recording_mic1, recording_mic2
    print(f"Processing remaining audio: mic1={len(recording_mic1)} chunks, mic2={len(recording_mic2)} chunks")
    if recording_mic1:
        process_audio_chunk(recording_mic1, "spk1", 0.0)  # Use 0.0 as base for remaining audio
    if recording_mic2:
        process_audio_chunk(recording_mic2, "spk2", 0.0)  # Use 0.0 as base for remaining audio

def save_transcription():
    """Save serialized transcription to JSON."""
    # Merge consecutive segments from the same speaker
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
    
    # Prepare output
    output_data = {
        "timeTakenForTranscription": round(transcription_time, 2),
        "timeTakenForDiarization": 0.0,  # Channel-based diarization
        "serializedTranscription": merged_segments
    }
    
    # Save to JSON
    try:
        with open(INTERMEDIATE_JSON, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSerialized transcription saved to {INTERMEDIATE_JSON}")
    except Exception as e:
        print(f"Error saving JSON: {e}")
    
    print(f"Total transcription time: {transcription_time:.2f} seconds")
    print(f"Total process time: {time.time() - overall_start_time:.2f} seconds")

def main():
    global is_recording, recording_mic1, recording_mic2
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # List audio devices
    print("Available audio devices:")
    print(sd.query_devices())
    print(f"Using Mic1 device index: {MIC1_DEVICE_INDEX} (AB13X USB Audio)")
    print(f"Using Mic2 device index: {MIC2_DEVICE_INDEX} (JBL TUNE 305C USB-C)")
    
    try:
        # Create input streams with adjusted blocksize
        stream1 = sd.InputStream(
            device=MIC1_DEVICE_INDEX,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=callback_mic1,
            blocksize=BLOCKSIZE,
            latency="high"
        )
        stream2 = sd.InputStream(
            device=MIC2_DEVICE_INDEX,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=callback_mic2,
            blocksize=BLOCKSIZE,
            latency="high"
        )
        
        # Start recording
        print("Start recording")
        is_recording = True
        with stream1, stream2:
            chunk_start_time = 0.0
            last_process_time = time.time()
            
            while is_recording:
                current_time = time.time()
                if current_time - last_process_time >= CHUNK_DURATION:
                    # Process accumulated audio
                    chunks_mic1 = recording_mic1
                    chunks_mic2 = recording_mic2
                    recording_mic1 = []
                    recording_mic2 = []
                    
                    print(f"Processing chunk at {round(chunk_start_time, 2)}s: mic1={len(chunks_mic1)} chunks, mic2={len(chunks_mic2)} chunks")
                    if chunks_mic1:
                        process_audio_chunk(chunks_mic1, "spk1", chunk_start_time)
                    if chunks_mic2:
                        process_audio_chunk(chunks_mic2, "spk2", chunk_start_time)
                    
                    chunk_start_time += CHUNK_DURATION
                    last_process_time = current_time
                
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Error during recording: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()