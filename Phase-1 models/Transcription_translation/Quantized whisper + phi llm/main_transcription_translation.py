import whisper  # This import is no longer used; kept for reference if needed
from faster_whisper import WhisperModel  # Replacement for original whisper
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import json
import time
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio
import scipy.io.wavfile
import os
import copy
import gc
from llama_cpp import Llama

# Disable numba caching to avoid disk space issues
os.environ["NUMBA_CACHE_DIR"] = ""

# Ensure offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Disable CUDA explicitly to ensure CPU usage
torch.cuda.is_available = lambda: False
device = "cpu"

# Paths
audio_file = "filipino_5.wav"
output_dir = "output"
intermediate_json = os.path.join(output_dir, "filipino_5.json")
output_json = os.path.join(output_dir, "filipino_5TT.json")
whisper_model_path = "transcription_model"  # Directory path to the downloaded faster-whisper model
phi_model_path = os.path.join("translation_model", "Phi-3-mini-4k-instruct-q4.gguf")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Start overall timing
overall_start_time = time.time()

# Step 1: Check if audio is stereo or mono
waveform, sample_rate = torchaudio.load(audio_file)
num_channels = waveform.shape[0]  # 1 for mono, 2 for stereo
is_stereo = num_channels == 2

# Load the faster-whisper model on CPU with int8 quantization for speed with minimal accuracy loss
if not os.path.exists(whisper_model_path):
    raise FileNotFoundError(f"Faster-Whisper model directory not found at {whisper_model_path}. Ensure the model is downloaded to the transcription_model/ folder.")
model = WhisperModel(whisper_model_path, device="cpu", compute_type="int8", local_files_only=True)

# Time transcription
transcribe_start_time = time.time()

if is_stereo:
    # Split stereo audio into left and right channels
    left_channel = waveform[0].numpy()  # Left channel
    right_channel = waveform[1].numpy()  # Right channel
    
    # Normalize audio levels
    left_channel = left_channel / (np.max(np.abs(left_channel)) + 1e-8)  # Add small epsilon to avoid division by zero
    right_channel = right_channel / (np.max(np.abs(right_channel)) + 1e-8)
    
    # Scale to int16 range for WAV file (Whisper expects this format)
    left_channel = (left_channel * 32767).astype(np.int16)
    right_channel = (right_channel * 32767).astype(np.int16)
    
    # Save temporary mono files for Whisper (Whisper expects a file input)
    left_channel_file = os.path.join(output_dir, "temp_left_channel.wav")
    right_channel_file = os.path.join(output_dir, "temp_right_channel.wav")
    
    # Write the channels to temporary files
    scipy.io.wavfile.write(left_channel_file, sample_rate, left_channel)
    scipy.io.wavfile.write(right_channel_file, sample_rate, right_channel)
    
    # Transcribe each channel separately (with language="tl")
    segments_left, _ = model.transcribe(left_channel_file, language="tl", word_timestamps=True)
    result_left = {"segments": [{"text": segment.text.strip(), "start": segment.start, "end": segment.end} for segment in segments_left]}
    
    # Debug: Print left channel transcription immediately
    print("\nDebug: Left Channel Transcription:")
    if result_left["segments"]:
        for segment in result_left["segments"]:
            text = segment["text"].strip()
            if text:  # Only print non-empty segments
                print(f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}]: {text}")
    else:
        print("No segments found in left channel.")
    
    segments_right, _ = model.transcribe(right_channel_file, language="tl", word_timestamps=True)
    result_right = {"segments": [{"text": segment.text.strip(), "start": segment.start, "end": segment.end} for segment in segments_right]}
    
    # Debug: Print right channel transcription immediately
    print("\nDebug: Right Channel Transcription:")
    if result_right["segments"]:
        for segment in result_right["segments"]:
            text = segment["text"].strip()
            if text:  # Only print non-empty segments
                print(f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}]: {text}")
    else:
        print("No segments found in right channel.")
    
    # Clean up temporary files
    os.remove(left_channel_file)
    os.remove(right_channel_file)
    
    # Assign speakers based on channels
    segments_data = []
    
    # Left channel -> spk1
    for i, segment in enumerate(result_left["segments"]):
        text = segment["text"].strip()
        if text:  # Only include non-empty segments
            segments_data.append({
                "index": i,
                "text": text,
                "start": segment["start"],
                "end": segment["end"],
                "speaker": "spk1"
            })
    
    # Right channel -> spk2
    for i, segment in enumerate(result_right["segments"]):
        text = segment["text"].strip()
        if text:  # Only include non-empty segments
            segments_data.append({
                "index": i + len(result_left["segments"]),  # Offset index
                "text": text,
                "start": segment["start"],
                "end": segment["end"],
                "speaker": "spk2"
            })
    
    # Sort segments by start time
    segments_data.sort(key=lambda x: x["start"])
    
    # Print combined transcription
    print("\nTranscription (Stereo - Channel-Based):")
    if segments_data:
        for segment in segments_data:
            print(f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}] ({segment['speaker']}): {segment['text']}")
    else:
        print("No segments found after combining channels.")
    
    # End transcription timing
    transcribe_end_time = time.time()
    transcription_time = transcribe_end_time - transcribe_start_time
    
    # Diarization is already done (channel-based)
    diarization_time = 0.0  # No clustering needed
    
else:
    # Mono audio: Preprocess with normalization
    mono_audio = waveform[0].numpy()  # Mono channel
    
    # Normalize audio levels
    mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-8)
    
    # Scale to int16 range for WAV file
    mono_audio = (mono_audio * 32767).astype(np.int16)
    
    # Save temporary file for Whisper
    temp_mono_file = os.path.join(output_dir, "temp_mono_channel.wav")
    scipy.io.wavfile.write(temp_mono_file, sample_rate, mono_audio)
    
    # Transcribe with language constraint
    segments, _ = model.transcribe(temp_mono_file, language="tl", word_timestamps=True)
    result = {"segments": [{"text": segment.text.strip(), "start": segment.start, "end": segment.end} for segment in segments]}
    
    # Clean up temporary file
    os.remove(temp_mono_file)
    
    # End transcription timing
    transcribe_end_time = time.time()
    transcription_time = transcribe_end_time - transcribe_start_time
    
    # Print transcription
    print("\nTranscription (Mono):")
    for segment in result["segments"]:
        text = segment["text"].strip()
        if text:  # Only print non-empty segments
            print(f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}]: {text}")
    
    # Prepare segments_data directly from Whisper results
    segments_data = []
    for i, segment in enumerate(result["segments"]):
        text = segment["text"].strip()
        if not text:  # Skip empty segments
            continue
        segments_data.append({
            "index": i,
            "text": text,
            "start": segment["start"],
            "end": segment["end"]
        })
    
    # Free memory after transcription
    del model
    gc.collect()
    
    # Time diarization
    diarization_start_time = time.time()
    
    # Rule-based speaker assignment: Alternate speakers assuming turn-taking
    if segments_data:
        n_speakers = 2  # Hardcoded, as before
        for i, segment in enumerate(segments_data):
            speaker_id = (i % n_speakers) + 1  # Alternate: 1,2,1,2,...
            segment["speaker"] = f"spk{speaker_id}"
    
    # End diarization timing
    diarization_end_time = time.time()
    diarization_time = diarization_end_time - diarization_start_time

# Merge consecutive segments from the same speaker
merged_segments = []
current_segment = None

for segment in segments_data:
    current_speaker = segment["speaker"]
    
    if current_segment is None:
        current_segment = {
            "dialogue": segment["text"],
            "speaker": current_speaker,
            "startTime": round(segment["start"], 2),
            "endTime": round(segment["end"], 2)
        }
    else:
        if current_segment["speaker"] == current_speaker:
            current_segment["dialogue"] += " " + segment["text"]
            current_segment["endTime"] = round(segment["end"], 2)
        else:
            merged_segments.append(current_segment)
            current_segment = {
                "dialogue": segment["text"],
                "speaker": current_speaker,
                "startTime": round(segment["start"], 2),
                "endTime": round(segment["end"], 2)
            }

if current_segment is not None:
    merged_segments.append(current_segment)

# Prepare intermediate output
intermediate_output = {
    "timeTakenForTranscription": round(transcription_time, 2),
    "timeTakenForDiarization": round(diarization_time, 2),
    "serializedTranscription": merged_segments
}

# Save intermediate output to JSON file
with open(intermediate_json, "w") as f:
    json.dump(intermediate_output, f, indent=2)

# Print times
print(f"\nTime taken for transcription: {transcription_time:.2f} seconds")
print(f"Time taken for diarization: {diarization_time:.2f} seconds")

# Print serialized output
print("\nSerialized Transcription:")
print(json.dumps(intermediate_output, indent=2))

# Free memory before translation
del segments_data
gc.collect()

# Step 2: Translation using Phi-3 LLM
# Check if the Phi-3 model file exists
if not os.path.exists(phi_model_path):
    print(f"Error: Phi-3 model file not found at {phi_model_path}. Please ensure the model is in the models/ folder.")
    exit(1)

# Load the Phi-3 model
try:
    print("Loading Phi-3 model...")
    llm = Llama(
        model_path=phi_model_path,
        n_ctx=512,  # Context length for input/output
        n_threads=os.cpu_count() or 4,  # Use available CPU cores
        n_gpu_layers=0,  # CPU-only inference
        verbose=False  # Reduce logging
    )
    print("Phi-3 model loaded successfully.")
except Exception as e:
    print(f"Error loading Phi-3 model: {e}")
    exit(1)

# Define the translation prompt template (based on Phi-3 model card)
prompt_template = (
    "<|system|>You are a precise translator from the Philippines who understands natural, candid, and usual Filipino+English dialect and code-switching perfectly.<|end|>\n"
    "<|user|>Transalte the input into pure english strictly maintaining all context and information. Your output will ONLY be the translated text, that's it, with NO ADDITIONAL TEXT OR EXPLANATION. Following is the input:\n"
    "Input: {}\n"
    "<|end|>\n"
    "<|assistant|>"
)

# Read the intermediate JSON file
try:
    with open(intermediate_json, "r") as f:
        transcription_data = json.load(f)
except FileNotFoundError:
    print(f"{intermediate_json} not found. Exiting.")
    exit(1)

# Process transcription.json (Direct Translation using Phi-3)
print("Processing transcription.json (Direct Translation with Phi-3)\n")

transcription_data_seg1 = copy.deepcopy(transcription_data)

for entry in transcription_data_seg1["serializedTranscription"]:
    dialogue_text = entry["dialogue"]
    
    try:
        prompt = prompt_template.format(dialogue_text)
        output = llm(
            prompt,
            max_tokens=100,  # Limit output length
            temperature=0.1,  # Balance creativity and determinism
            stop=["<|end|>"],  # Stop at end token
            echo=False  # Exclude prompt from output
        )
        response = output["choices"][0]["text"].strip()
        # Extract the response after <|assistant|> tag
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[1].strip()
        translated_text = response
    except Exception as e:
        print(f"Error during translation of '{dialogue_text}': {e}")
        translated_text = dialogue_text  # Fallback to original text if translation fails
    
    entry["dialogue"] = translated_text
    print(f"Original: {dialogue_text}")
    print(f"Translated: {translated_text}\n")

# Clean up the LLM
llm.reset()

# Save translated output to JSON file
with open(output_json, "w") as f:
    json.dump(transcription_data_seg1, f, indent=2)
print(f"Translated transcription saved to {output_json}")

# Optionally, include S2ST (skipped as it was specific to SeamlessM4T)
print("Speech-to-Speech translation skipped (not supported by Phi-3 model).")

# End overall timing
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"Total process time: {overall_time:.2f} seconds")