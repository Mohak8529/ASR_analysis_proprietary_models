import whisper
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler  # Kept for compatibility, though not used now
import json
import time
import os
import gc
import scipy.io.wavfile
import torchaudio

# Paths
audio_file = "calls for diarization testing/calls/RPC5 - PROMISED TO PAY_AIRISH LAMPA_Aiza Barot_648691514786_2232025.wav"
output_dir = "stereofilipino"
intermediate_json = os.path.join(output_dir, "diarized_transcription.json")
whisper_model_path = "model/large-v3.pt"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Start overall timing
overall_start_time = time.time()

# Check if audio is mono
waveform, sample_rate = torchaudio.load(audio_file)
num_channels = waveform.shape[0]
if num_channels != 1:
    raise ValueError(f"Audio file {audio_file} is not mono (has {num_channels} channels). This script only handles mono audio.")

# Load Whisper model on CPU
if not os.path.exists(whisper_model_path):
    raise FileNotFoundError(f"Whisper model not found at {whisper_model_path}. Ensure the model is in the model/ folder.")
model = whisper.load_model(whisper_model_path, device="cpu")

# Time transcription
transcribe_start_time = time.time()

# Preprocess audio: Normalize and save as temporary WAV file for Whisper
mono_audio = waveform[0].numpy()
mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-8)  # Normalize with epsilon to avoid division by zero
mono_audio = (mono_audio * 32767).astype(np.int16)  # Scale to int16 for WAV

temp_mono_file = os.path.join(output_dir, "temp_mono_channel.wav")
scipy.io.wavfile.write(temp_mono_file, sample_rate, mono_audio)

# Transcribe with language constraint (Tagalog)
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

# No need to load audio for features (skipping clustering)

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
        "end": segment["end"],
        # "features": not needed anymore
    })

# Free memory after transcription (model still deleted)
del model
gc.collect()

# Time diarization (now just rule-based assignment)
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

# Prepare output
output_data = {
    "timeTakenForTranscription": round(transcription_time, 2),
    "timeTakenForDiarization": round(diarization_time, 2),
    "serializedTranscription": merged_segments
}

# Save output to JSON file
with open(intermediate_json, "w") as f:
    json.dump(output_data, f, indent=2)

# Print times and results
print(f"\nTime taken for transcription: {transcription_time:.2f} seconds")
print(f"Time taken for diarization: {diarization_time:.2f} seconds")

print("\nDiarized Transcription:")
for segment in merged_segments:
    print(f"[{segment['startTime']} - {segment['endTime']}] ({segment['speaker']}): {segment['dialogue']}")

print(f"\nDiarized transcription saved to {intermediate_json}")

# Free memory
del segments_data
gc.collect()

# End overall timing
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"Total process time: {overall_time:.2f} seconds")
