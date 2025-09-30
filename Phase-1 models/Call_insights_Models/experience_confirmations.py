import os
import json
import torch
from transformers import pipeline
from typing import Dict, List
from pathlib import Path

# Directories
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "Confirmations")
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
        device=device,
        hypothesis_template="In this conversation, {}."
    )
except Exception as e:
    print(f"Error loading model from {model_dir}: {e}")
    print("Ensure the 'bart-large-mnli' directory contains the model files. Run 'bart_mnli_download.py' if missing.")
    exit(1)

def classify_confirmation(conversation: str, confirmed_label: str, not_confirmed_label: str) -> bool:
    """Classify whether a specific aspect was confirmed in the conversation."""
    try:
        candidate_labels = [confirmed_label, not_confirmed_label]
        result = zero_shot_classifier(conversation, candidate_labels, multi_label=False)
        print(f"Raw pipeline result for labels '{confirmed_label}' vs '{not_confirmed_label}': {result}")
        scores_dict = dict(zip(result['labels'], result['scores']))
        confirmed_score = scores_dict[confirmed_label]
        return confirmed_score > 0.5
    except Exception as e:
        print(f"Error classifying confirmation: {e}")
        return False

def analyze_confirmations(json_file: str) -> Dict[str, bool]:
    """Analyze a single JSON file and validate confirmations."""
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

    # Define the confirmation checks
    confirmations = {
        "Whether debtor name was confirmed": classify_confirmation(
            full_conversation,
            "the debtor name was confirmed",
            "the debtor name was not confirmed"
        ),
        "Was the amount indebted confirmed": classify_confirmation(
            full_conversation,
            "the amount indebted was confirmed",
            "the amount indebted was not confirmed"
        ),
        "Was the terms of debt payment confirmed": classify_confirmation(
            full_conversation,
            "the terms of debt payment were confirmed",
            "the terms of debt payment were not confirmed"
        ),
        "Was payment date confirmed": classify_confirmation(
            full_conversation,
            "the payment date was confirmed",
            "the payment date was not confirmed"
        )
    }

    # Save output
    base_filename = os.path.splitext(json_path.name)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_confirmations.txt")
    with open(output_file, "w", encoding="utf-8") as f_output:
        for key, value in confirmations.items():
            f_output.write(f"{key}: {str(value)}\n")

    print(f"[✔] {base_filename} → Confirmations:")
    for key, value in confirmations.items():
        print(f"{key}: {value}")

    return confirmations

def process_all_json_files(directory: str):
    """Process all JSON files in the specified directory."""
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    for json_file in json_files:
        analyze_confirmations(os.path.join(directory, json_file))

if __name__ == "__main__":
    # Process a single file
    json_file = "transcription.json"
    analyze_confirmations(json_file)

    # Optionally process all JSON files in the directory
    # print(f"Processing JSON files in {base_dir}")
    # process_all_json_files(base_dir)