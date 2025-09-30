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
audio_file = "filipinofortesting/Left Message_Erecah Lee Formanes_Andrie Capilitan Cadunggan_616046990126_2025-01-16_16.40.51.wav"
output_dir = "filipinofortesting"
intermediate_json = os.path.join(output_dir, "1.json")
output_json = os.path.join(output_dir, "1t.json")
whisper_model_path = "model/large-v3.pt"
seamless_model_path = "hf_cache/hub/models--facebook--seamless-m4t-v2-large/snapshots/5f8cc790b19fc3f67a61c105133b20b34e3dcb76"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Start overall timing
overall_start_time = time.time()

# Step 1: Transcription and Diarization
# Load the Whisper model on CPU
if not os.path.exists(whisper_model_path):
    raise FileNotFoundError(f"Whisper model not found at {whisper_model_path}. Ensure the model is in the model/ folder.")
model = whisper.load_model(whisper_model_path, device="cpu")

# Time transcription
transcribe_start_time = time.time()

# Transcribe with timestamps, specify language as Tagalog (Filipino)
result = model.transcribe(audio_file, language="tl", word_timestamps=True)

# End transcription timing
transcribe_end_time = time.time()
transcription_time = transcribe_end_time - transcribe_start_time

# Print transcription
print("\nTranscription:")
for segment in result["segments"]:
    print(f"[{round(segment['start'], 2)} - {round(segment['end'], 2)}]: {segment['text'].strip()}")

# Load audio for feature extraction
y, sr = librosa.load(audio_file)

# Create more robust feature extraction
def extract_audio_features(y, sr, start_time, end_time):
    # Convert times to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Safety checks
    if start_sample >= len(y) or start_sample >= end_sample:
        return None
        
    # Extract segment
    segment = y[start_sample:min(end_sample, len(y))]
    
    if len(segment) < sr * 0.5:  # Require at least 0.5 seconds of audio
        return None
        
    # Extract multiple features
    features = []
    
    # MFCCs (vocal tract characteristics)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
    features.append(np.mean(mfcc, axis=1))
    features.append(np.std(mfcc, axis=1))
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    features.append(np.mean(spectral_centroid))
    features.append(np.std(spectral_centroid))
    
    # Flatten and combine all features
    return np.concatenate([f.flatten() for f in features])

# Extract features for each segment
segments_data = []

for i, segment in enumerate(result["segments"]):
    features = extract_audio_features(y, sr, segment["start"], segment["end"])
    
    if features is not None:
        segments_data.append({
            "index": i,
            "text": segment["text"].strip(),
            "start": segment["start"],
            "end": segment["end"],
            "features": features
        })

# Free memory after feature extraction
del y
gc.collect()

# Time diarization
diarization_start_time = time.time()

# Apply clustering with improved approach
if segments_data:
    # Extract features for clustering
    features = np.array([s["features"] for s in segments_data])
    
    # Standardize features for better clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Clustering - with basic parameters that work in all scikit-learn versions
    n_speakers = 2  # Adjust based on your audio
    
    # Apply clustering with basic parameters
    clustering = AgglomerativeClustering(n_clusters=n_speakers).fit(features_scaled)
    
    speaker_labels = clustering.labels_
    
    # Find the first segment of each cluster to determine order
    first_segments = {}
    for i, label in enumerate(speaker_labels):
        if label not in first_segments or segments_data[i]["start"] < first_segments[label]["start"]:
            first_segments[label] = {
                "start": segments_data[i]["start"],
                "cluster_id": label
            }

    # Sort clusters by their first occurrence (earliest start time)
    sorted_clusters = sorted(first_segments.values(), key=lambda x: x["start"])

    # Map the earliest cluster to spk1, next to spk2, etc.
    speaker_mapping = {cluster["cluster_id"]: i + 1 for i, cluster in enumerate(sorted_clusters)}

    # Merge consecutive segments from the same speaker
    merged_segments = []
    current_segment = None

    for i in range(len(segments_data)):
        segment = segments_data[i]
        cluster_id = speaker_labels[i]
        speaker_id = speaker_mapping[cluster_id]
        current_speaker = f"spk{speaker_id}"

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

    # End diarization timing
    diarization_end_time = time.time()
    diarization_time = diarization_end_time - diarization_start_time

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

    # Also print serialized output
    print("\nSerialized Transcription:")
    print(json.dumps(intermediate_output, indent=2))
else:
    print("No valid segments found for speaker diarization")
    exit(1)

# Free memory before translation
del model
del segments_data
del features
del features_scaled
gc.collect()

# Step 2: Translation
# Load processor and model with slow tokenizer
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

# Create a deep copy to avoid modifying the original data
transcription_data_seg1 = copy.deepcopy(transcription_data)

# Process each dialogue in serializedTranscription
for entry in transcription_data_seg1["serializedTranscription"]:
    dialogue_text = entry["dialogue"]
    
    # Translate the dialogue directly (no proper noun replacement)
    text_inputs = processor(text=dialogue_text, src_lang="tgl", return_tensors="pt").to(device)
    output_tokens = model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
    token_list = output_tokens[0].cpu().numpy().flatten().tolist()
    translated_text = processor.decode(token_list, skip_special_tokens=True)
    
    # Update the dialogue field with the translated text
    entry["dialogue"] = translated_text
    print(f"Original: {dialogue_text}")
    print(f"Translated: {translated_text}\n")

# Write the updated JSON to a file
with open(output_json, "w") as f:
    json.dump(transcription_data_seg1, f, indent=2)
print(f"Translated transcription saved to {output_json}")


# End overall timing
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"Total process time: {overall_time:.2f} seconds")