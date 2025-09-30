import os
import json
import torch
import re
from transformers import pipeline
from typing import Dict, List, Optional
from pathlib import Path

# Directories
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "Speaker_Role_Classification")
os.makedirs(output_dir, exist_ok=True)

# Local model directory
model_dir = os.path.join(base_dir, "bart-large-mnli")

# Device config
device = 0 if torch.cuda.is_available() else -1

# Load zero-shot classifier pipeline from local directory
try:
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=model_dir,
        device=device
    )
except Exception as e:
    print(f"Error loading model from {model_dir}: {e}")
    print("Ensure the 'bart-large-mnli' directory contains the model files. Run 'bart_mnli_download.py' if missing.")
    exit(1)

candidate_labels = ["Customer"]

# Regex patterns for Bank Agent identification
agent_patterns = [
    r"welcome\s+to\s+.*bank",
    r"this\s+is\s+.*\s+from\s+.*bank",
    r"i\s+am\s+.*\s+from\s+.*bank",
    r"calling\s+.*\s+from\s+.*bank"
]

def check_agent_patterns(text: str) -> Optional[str]:
    """Check if the dialogue matches any Bank Agent patterns. Return matched pattern or None."""
    text_lower = text.lower()
    for pattern in agent_patterns:
        if re.search(pattern, text_lower):
            return pattern
    return None

def classify_speaker_role(text: str) -> float:
    """Classify the speaker's dialogue and return the 'Customer' score."""
    result = zero_shot_classifier(text, candidate_labels, multi_label=False)
    print(f"Raw pipeline result: {result}")
    score = result['scores'][0]  # Score for "Customer"
    return score

def analyze_speaker_roles(json_file: str) -> Dict[str, dict]:
    """Analyze a JSON file and classify speaker roles, using regex first, then model if needed."""
    json_path = Path(json_file)
    if not json_path.is_file():
        print(f"File not found: {json_file}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON from {json_file}: {e}")
            return None

    if not isinstance(data, list) or not all("dialogue" in d and "speaker" in d for d in data):
        print(f"Malformed transcription in {json_file}")
        return None

    # Aggregate dialogue by speaker, using only the dialogue field
    speaker_dialogues = {}
    filtered_data = []
    for entry in data:
        speaker = entry["speaker"]
        dialogue = entry["dialogue"]
        if speaker not in speaker_dialogues:
            speaker_dialogues[speaker] = []
        speaker_dialogues[speaker].append(dialogue)
        # Store only dialogue and speaker for output
        filtered_data.append({"dialogue": dialogue, "speaker": speaker})

    # Check for agent patterns first
    results = {}
    agent_speaker = None
    matched_pattern = None
    for speaker, dialogues in speaker_dialogues.items():
        full_dialogue = " ".join(dialogues)
        pattern = check_agent_patterns(full_dialogue)
        if pattern:
            agent_speaker = speaker
            matched_pattern = pattern
            break

    if agent_speaker and len(speaker_dialogues) == 2:
        # Assign roles based on regex match
        for speaker in speaker_dialogues:
            if speaker == agent_speaker:
                results[speaker] = {"role": "Bank Agent", "customer_score": None, "method": f"Regex match: {matched_pattern}"}
                print(f"[✔] {speaker} → Role: Bank Agent, Method: Regex match ('{matched_pattern}')")
            else:
                results[speaker] = {"role": "Customer", "customer_score": None, "method": "Assigned as non-agent"}
                print(f"[✔] {speaker} → Role: Customer, Method: Assigned as non-agent")
    else:
        # No regex match, use model-based classification
        speaker_scores = {}
        for speaker, dialogues in speaker_dialogues.items():
            full_dialogue = " ".join(dialogues)
            customer_score = classify_speaker_role(full_dialogue)
            speaker_scores[speaker] = customer_score

        # Assign roles: highest "Customer" score gets "Customer," other gets "Bank Agent"
        if len(speaker_scores) == 2:
            speaker1, speaker2 = speaker_scores.keys()
            score1, score2 = speaker_scores[speaker1], speaker_scores[speaker2]
            if score1 > score2:
                results[speaker1] = {"role": "Customer", "customer_score": score1, "method": "Model"}
                results[speaker2] = {"role": "Bank Agent", "customer_score": score2, "method": "Model"}
            else:
                results[speaker1] = {"role": "Bank Agent", "customer_score": score1, "method": "Model"}
                results[speaker2] = {"role": "Customer", "customer_score": score2, "method": "Model"}
        else:
            # Fallback for unexpected number of speakers
            for speaker, score in speaker_scores.items():
                role = "Customer" if score > 0.5 else "Bank Agent"
                results[speaker] = {"role": role, "customer_score": score, "method": "Model"}

        for speaker, result in results.items():
            print(f"[✔] {speaker} → Role: {result['role']}, Customer Score: {result['customer_score']}, Method: {result['method']}")

    # Save output
    base_filename = os.path.splitext(json_path.name)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_roles.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for speaker, result in results.items():
            f.write(f"{speaker}: {result['role']}\n")
            if result['customer_score'] is not None:
                f.write(f"Customer Score: {result['customer_score']}\n")
            f.write(f"Method: {result['method']}\n")
        f.write("\nInput JSON (dialogue and speaker only):\n")
        json.dump(filtered_data, f, indent=2)

    return results

def process_all_json_files(directory: str):
    """Process all JSON files in the specified directory."""
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    for json_file in json_files:
        analyze_speaker_roles(os.path.join(directory, json_file))

if __name__ == "__main__":
    # Process a single JSON file
    json_file = "transcription.json"
    analyze_speaker_roles(json_file)