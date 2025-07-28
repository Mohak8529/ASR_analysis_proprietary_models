import os
import json
import torch
from transformers import pipeline
from typing import List
from pathlib import Path

# Directories
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "Promise to Pay")
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

candidate_labels = [
    "Complete Settlement",
    "Partial Settlement",
    "Broken Promise",
    "Denial of Payment",
]

def classify_intent(text: str) -> tuple:
    result = zero_shot_classifier(text, candidate_labels, multi_label=False)
    # Debug: Print raw pipeline output
    print(f"Raw pipeline result: {result}")
    # Map scores to result['labels'] to match pipeline order
    scores = result['scores']
    labels = result['labels']
    label_score_pairs = list(zip(labels, scores))
    # Sort by score in descending order
    sorted_pairs = sorted(label_score_pairs, key=lambda x: x[1], reverse=True)
    top_label = sorted_pairs[0][0]
    # Create scores dictionary using result['labels'] order
    scores_dict = dict(zip(labels, scores))
    return top_label, scores_dict

def analyze_promises(json_file: str) -> List[dict]:
    """Analyze a single JSON file and classify its intent."""
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

    transcription = data if isinstance(data, list) else data.get("serializedTranscription", [])
    if not isinstance(transcription, list) or not all("dialogue" in d for d in transcription):
        print(f"Malformed transcription in {json_file}")
        return None

    full_conversation = " ".join(d["dialogue"] for d in transcription)

    # Classify intent
    top_intent, scores = classify_intent(full_conversation)

    # Save output
    base_filename = os.path.splitext(json_path.name)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_intent.txt")
    with open(output_file, "w", encoding="utf-8") as f_intent:
        f_intent.write(f"{top_intent}\nScores: {scores}")

    print(f"[✔] {base_filename} → Intent: {top_intent}, Scores: {scores}")
    return transcription

def process_all_json_files(directory: str):
    """Process all JSON files in the specified directory."""
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    for json_file in json_files:
        analyze_promises(os.path.join(directory, json_file))

if __name__ == "__main__":
    # Process all JSON files in the current directory
    # print(f"Processing JSON files in {base_dir}")
    # process_all_json_files(base_dir)

    # Alternatively, to process a single file, uncomment and specify the file:
    json_file = "transcription.json"
    analyze_promises(json_file)