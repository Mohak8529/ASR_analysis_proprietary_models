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
    "rpc",
    "Not an rpc",
]

# New — Rule-based pre-check for obvious RPC/Non-RPC indicators
def pre_check_rpc_presence(transcription: List[dict]) -> str:
    rpc_phrases = ["yes, speaking", "this is", "i am", "i'm", "that's me"]
    non_rpc_phrases = ["wrong number", "i don't know", "not me", "not here"]

    text = " ".join(d["dialogue"].lower() for d in transcription if d["speaker"] == "spk2")

    if any(phrase in text for phrase in rpc_phrases):
        return "rpc"
    elif any(phrase in text for phrase in non_rpc_phrases):
        return "Not an rpc"
    else:
        return "uncertain"

# Existing — Classify customer dialogues only
def classify_intent(customer_text: str) -> tuple:
    result = zero_shot_classifier(customer_text, candidate_labels, multi_label=False)
    print(f"Raw pipeline result: {result}")
    scores = result['scores']
    labels = result['labels']
    label_score_pairs = list(zip(labels, scores))
    sorted_pairs = sorted(label_score_pairs, key=lambda x: x[1], reverse=True)
    top_label = sorted_pairs[0][0]
    scores_dict = dict(zip(labels, scores))
    return top_label, scores_dict

# Existing — Analyze one JSON transcription
def analyze_promises(json_file: str) -> List[dict]:
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

    # Pre-check for obvious RPC patterns
    pre_check_result = pre_check_rpc_presence(transcription)
    if pre_check_result != "uncertain":
        top_intent = pre_check_result
        scores = {"rpc": 1.0 if top_intent == "rpc" else 0.0, "Not an rpc": 1.0 if top_intent == "Not an rpc" else 0.0}
    else:
        # If uncertain, classify using customer dialogues only
        customer_text = " ".join(d["dialogue"] for d in transcription if d["speaker"] == "spk2")
        if not customer_text.strip():
            top_intent = "Not an rpc"
            scores = {"rpc": 0.0, "Not an rpc": 1.0}
        else:
            top_intent, scores = classify_intent(customer_text)

    # Save output
    base_filename = os.path.splitext(json_path.name)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_intent.txt")
    with open(output_file, "w", encoding="utf-8") as f_intent:
        f_intent.write(f"{top_intent}\nScores: {scores}")

    # print(f"[✔] {base_filename} → Intent: {top_intent}, Scores: {scores}")
    print("WHETHER ITS RPC OR NOT:", top_intent)
    return transcription

# Existing — Process all JSON files in a directory
def process_all_json_files(directory: str):
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    for json_file in json_files:
        analyze_promises(os.path.join(directory, json_file))

if __name__ == "__main__":
    # Single file test
    json_file = "transcription.json"
    analyze_promises(json_file)
    # Or batch processing: process_all_json_files(base_dir)
