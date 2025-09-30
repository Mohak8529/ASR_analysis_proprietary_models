import whisper
import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import json
import time
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TTokenizer
import scipy.io.wavfile
import os
import copy
import gc

# Disable numba caching to avoid disk space issues
os.environ["NUMBA_CACHE_DIR"] = ""

# Ensure offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Disable CUDA explicitly to ensure CPU usage
torch.cuda.is_available = lambda: False
device = "cpu"

# Paths
audio_file = "stereofilipino/Left Message_Erecah Lee Formanes_Andrie Capilitan Cadunggan_616046990126_2025-01-16_16.40.51.wav"
output_dir = "stereofilipino"
intermediate_json = os.path.join(output_dir, "1.json")
output_json = os.path.join(output_dir, "1t.json")
whisper_model_path = "model/large-v3.pt"
seamless_model_path = "hf_cache/hub/models--facebook--seamless-m4t-v2-large/snapshots/5f8cc790b19fc3f67a61c105133b20b34e3dcb76"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Start overall timing
overall_start_time = time.time()

# Step 1: Check if audio is stereo or mono
waveform, sample_rate = torchaudio.load(audio_file)
num_channels = waveform.shape[0]  # 1 for mono, 2 for stereo
is_stereo = num_channels == 2

# Load the Whisper model on CPU
if not os.path.exists(whisper_model_path):
    raise FileNotFoundError(f"Whisper model not found at {whisper_model_path}. Ensure the model is in the model/ folder.")
model = whisper.load_model(whisper_model_path, device="cpu")

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
    result_left = model.transcribe(left_channel_file, language="tl", word_timestamps=True)
    
    # Debug: Print left channel transcription immediately
    print("\nDebug: Left Channel Transcription:")
    if result_left["segments"]:
        for segment in result_left["segments"]:
            text = segment["text"].strip()
            if text:  # Only print non-empty segments
                print(f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}]: {text}")
    else:
        print("No segments found in left channel.")
    
    result_right = model.transcribe(right_channel_file, language="tl", word_timestamps=True)
    
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
    result = model.transcribe(temp_mono_file, language="tl", word_timestamps=True)
    
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

# Step 2: Translation
tokenizer = SeamlessM4TTokenizer.from_pretrained(seamless_model_path, use_fast=False, local_files_only=True)
processor = AutoProcessor.from_pretrained(seamless_model_path, tokenizer=tokenizer, local_files_only=True)
model = SeamlessM4Tv2Model.from_pretrained(seamless_model_path, local_files_only=True)
model = model.to(device)

# Read the intermediate JSON file
try:
    with open(intermediate_json, "r") as f:
        transcription_data = json.load(f)
except FileNotFoundError:
    print(f"{intermediate_json} not found. Exiting.")
    exit(1)

# Process transcription.json (Direct Translation, src_lang='tgl')
print("Processing transcription.json (Direct Translation, src_lang='tgl')\n")

transcription_data_seg1 = copy.deepcopy(transcription_data)

for entry in transcription_data_seg1["serializedTranscription"]:
    dialogue_text = entry["dialogue"]
    
    text_inputs = processor(text=dialogue_text, src_lang="tgl", return_tensors="pt").to(device)
    output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
    token_list = output_tokens[0].cpu().numpy().flatten().tolist()
    translated_text = processor.decode(token_list, skip_special_tokens=True)
    
    entry["dialogue"] = translated_text
    print(f"Original: {dialogue_text}")
    print(f"Translated: {translated_text}\n")

with open(output_json, "w") as f:
    json.dump(transcription_data_seg1, f, indent=2)
print(f"Translated transcription saved to {output_json}")

# Optionally, include S2ST if the audio file exists
audio_path = audio_file
if os.path.exists(audio_path):
    audio, orig_freq = torchaudio.load(audio_path)
    if orig_freq != 16000:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
    audio_inputs = processor(audios=audio, sampling_rate=16000, return_tensors="pt").to(device)
    audio_array = model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
    sample_rate = model.config.sampling_rate
    scipy.io.wavfile.write("output_s2st_eng.wav", rate=sample_rate, data=audio_array)
    print("Speech-to-Speech translation (to English) saved as output_s2st_eng.wav")
else:
    print(f"Audio file {audio_path} not found. Skipping S2ST.")

# End overall timing
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"Total process time: {overall_time:.2f} seconds")