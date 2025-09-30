import whisper
import librosa
import numpy as np
import json
import time
import os
import gc
import scipy.io.wavfile
import torchaudio
from pyannote.audio import Pipeline
from huggingface_hub import login

# Paths
audio_file = "audio/RPC5_JAMES CARDONA_Kenneth Barredo Mirasol_615322953808_232025.wav"
output_dir = "output"
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

# Free Whisper model memory
del model
gc.collect()

# Time diarization
diarization_start_time = time.time()

# Authenticate Hugging Face & load Pyannote
hf_token = "hf_sqyfxfpXXXXXXXXXXXXXXXXXXX"
if not hf_token:
    raise RuntimeError("Set HUGGINGFACE_TOKEN or run `huggingface-cli login`")
login(token=hf_token)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

def normalize_speakers(diar_segs):
    """Map Pyannote speakers SPEAKER_00, SPEAKER_01 → spk1, spk2 consistently."""
    speaker_map = {}
    next_id = 1
    norm_segs = []
    for d0, d1, spk in diar_segs:
        if spk not in speaker_map:
            speaker_map[spk] = f"spk{next_id}"
            next_id += 1
        norm_segs.append((d0, d1, speaker_map[spk]))
    return norm_segs

def align_diarization_with_transcription(audio_path, transcription):
    """Align transcription segments with Pyannote diarization."""
    diarization = pipeline(audio_path)

    # Extract diarization turns
    diar_segs = [(turn.start, turn.end, spk) for turn, _, spk in diarization.itertracks(yield_label=True)]
    diar_segs.sort(key=lambda x: x[0])

    # Normalize speaker labels
    diar_segs = normalize_speakers(diar_segs)

    segments_data = []
    for segment in transcription:
        text = segment["text"].strip()
        if not text:  # Skip empty segments
            continue
        segments_data.append({
            "text": text,
            "start": segment["start"],
            "end": segment["end"]
        })

    new_segments = []
    for tseg in segments_data:
        t0, t1, text = tseg["start"], tseg["end"], tseg["text"]

        # Find diarization parts that overlap transcription segment
        overlaps = [(d0, d1, spk) for d0, d1, spk in diar_segs if not (d1 <= t0 or d0 >= t1)]

        if overlaps:
            # Split transcription text among overlapping diarization segments
            for d0, d1, spk in overlaps:
                sub0, sub1 = max(t0, d0), min(t1, d1)
                # Estimate the proportion of the transcription segment covered by this diarization segment
                overlap_duration = sub1 - sub0
                trans_duration = t1 - t0
                if trans_duration > 0:
                    text_proportion = overlap_duration / trans_duration
                    # Assign the full text if the overlap is significant (e.g., >50% of the transcription segment)
                    assigned_text = text if text_proportion > 0.5 else ""
                    new_segments.append({
                        "dialogue": assigned_text,
                        "speaker": spk,
                        "startTime": round(sub0, 2),
                        "endTime": round(sub1, 2)
                    })
        else:
            # If no diarization found, assign default speaker
            new_segments.append({
                "dialogue": text,
                "speaker": "spk1",
                "startTime": round(t0, 2),
                "endTime": round(t1, 2)
            })

    # Sort segments by startTime
    new_segments = sorted(new_segments, key=lambda x: x["startTime"])

    # === NEW: Step 1 - Ensure all transcription dialogues are included ===
    # Create a mapping of transcription segments for lookup
    trans_segments = [(seg["start"], seg["end"], seg["text"]) for seg in segments_data]
    for t_start, t_end, t_text in trans_segments:
        # Check if this transcription segment's text is missing in new_segments
        found = False
        for seg in new_segments:
            if seg["dialogue"] == t_text and abs(seg["startTime"] - t_start) < 0.5 and abs(seg["endTime"] - t_end) < 0.5:
                found = True
                break
        if not found:
            # Find the diarization segment that best overlaps with this transcription segment
            best_overlap = None
            max_overlap_duration = 0
            for d0, d1, spk in diar_segs:
                overlap_start = max(t_start, d0)
                overlap_end = min(t_end, d1)
                overlap_duration = max(0, overlap_end - overlap_start)
                if overlap_duration > max_overlap_duration:
                    max_overlap_duration = overlap_duration
                    best_overlap = (d0, d1, spk)
            if best_overlap:
                d0, d1, spk = best_overlap
                # Append the missing transcription text to the segment with the same speaker
                for seg in new_segments:
                    if seg["speaker"] == spk and abs(seg["startTime"] - d0) < 0.5 and abs(seg["endTime"] - d1) < 0.5:
                        if seg["dialogue"]:
                            seg["dialogue"] += " " + t_text
                        else:
                            seg["dialogue"] = t_text
                        break
                else:
                    # If no matching segment found, add a new one
                    new_segments.append({
                        "dialogue": t_text,
                        "speaker": spk,
                        "startTime": round(max(t_start, d0), 2),
                        "endTime": round(min(t_end, d1), 2)
                    })

    # Sort segments again after adding missing dialogues
    new_segments = sorted(new_segments, key=lambda x: x["startTime"])

    # Print diarized output after Step 1
    print("\nDiarized Transcription (After Step 1 - Include All Transcription Dialogues):")
    for segment in new_segments:
        print(f"[{segment['startTime']} - {segment['endTime']}] ({segment['speaker']}): {segment['dialogue']}")

    # === NEW: Step 2 - Exclude empty dialogue segments ===
    new_segments = [seg for seg in new_segments if seg["dialogue"]]

    # Print diarized output after Step 2
    print("\nDiarized Transcription (After Step 2 - Exclude Empty Dialogues):")
    for segment in new_segments:
        print(f"[{segment['startTime']} - {segment['endTime']}] ({segment['speaker']}): {segment['dialogue']}")

    # === NEW: Step 3 - Merge consecutive same-speaker dialogues ===
    merged_segments = []
    current_segment = None

    for segment in new_segments:
        current_speaker = segment["speaker"]

        if current_segment is None:
            current_segment = {
                "dialogue": segment["dialogue"],
                "speaker": current_speaker,
                "startTime": segment["startTime"],
                "endTime": segment["endTime"]
            }
        else:
            if current_segment["speaker"] == current_speaker and segment["startTime"] <= current_segment["endTime"] + 0.5:
                # Merge if same speaker and segments are close in time
                current_segment["dialogue"] = current_segment["dialogue"] + " " + segment["dialogue"]
                current_segment["endTime"] = max(current_segment["endTime"], segment["endTime"])
            else:
                # Save current segment and start a new one
                merged_segments.append(current_segment)
                current_segment = {
                    "dialogue": segment["dialogue"],
                    "speaker": current_speaker,
                    "startTime": segment["startTime"],
                    "endTime": segment["endTime"]
                }

    # Append the last segment
    if current_segment is not None:
        merged_segments.append(current_segment)

    # Print diarized output after Step 3
    print("\nDiarized Transcription (After Step 3 - Merge Consecutive Same-Speaker Dialogues):")
    for segment in merged_segments:
        print(f"[{segment['startTime']} - {segment['endTime']}] ({segment['speaker']}): {segment['dialogue']}")

    return merged_segments

# Align diarization with transcription
merged_segments = align_diarization_with_transcription(audio_file, result["segments"])

# End diarization timing
diarization_end_time = time.time()
diarization_time = diarization_end_time - diarization_start_time

# Prepare output
output_data = {
    "timeTakenForTranscription": round(transcription_time, 2),
    "timeTakenForDiarization": round(diarization_time, 2),
    "serializedTranscription": merged_segments
}

# Save output to JSON file
with open(intermediate_json, "w") as f:
    json.dump(output_data, f, indent=2)

# Print times
print(f"\nTime taken for transcription: {transcription_time:.2f} seconds")
print(f"Time taken for diarization: {diarization_time:.2f} seconds")

print(f"\nDiarized transcription saved to {intermediate_json}")

# Free memory
del pipeline
gc.collect()

# End overall timing
overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time
print(f"Total process time: {overall_time:.2f} seconds")