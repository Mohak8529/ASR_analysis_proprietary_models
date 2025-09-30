import os
import json
import torch
from transformers import pipeline
from typing import List
from pathlib import Path

# Directories
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "Analysis_Output")
os.makedirs(output_dir, exist_ok=True)

# Local model directory (or use Hugging Face model hub)
model_dir = os.path.join(base_dir, "bart-large-mnli")  # Update path if needed

# Device config
device = 0 if torch.cuda.is_available() else -1

# Load zero-shot classifier pipeline
try:
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli" if not os.path.exists(model_dir) else model_dir,
        device=device
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the 'bart-large-mnli' directory exists or internet connection for model download.")
    exit(1)

# Define candidate labels for each classification task
intent_labels = ["Positive", "Negative"]
product_labels = ["Credit Card"]
sentiment_labels = ["Positive", "Negative", "Neutral"]

# Classify text for a given set of labels
def classify_text(text: str, candidate_labels: List[str]) -> tuple:
    if not text.strip():
        # Return default values if text is empty
        default_label = candidate_labels[-1] if candidate_labels[-1] in ["Unknown", "Neutral"] else candidate_labels[0]
        scores = {label: 1.0 if label == default_label else 0.0 for label in candidate_labels}
        return default_label, scores
    
    result = zero_shot_classifier(text, candidate_labels, multi_label=False)
    scores = result['scores']
    labels = result['labels']
    label_score_pairs = list(zip(labels, scores))
    sorted_pairs = sorted(label_score_pairs, key=lambda x: x[1], reverse=True)
    top_label = sorted_pairs[0][0]
    scores_dict = dict(zip(labels, scores))
    return top_label, scores_dict

# Analyze one JSON transcription for intent, product, and sentiment
def analyze_transcription(json_file: str) -> List[dict]:
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

    # Extract customer dialogues (assuming spk2 is the customer)
    customer_text = " ".join(d["dialogue"] for d in transcription if d["speaker"] == "spk2")
    
    # Default results if no customer text
    if not customer_text.strip():
        intent = "Negative"
        intent_scores = {"Positive": 0.0, "Negative": 1.0}
        product = "Unknown"
        product_scores = {"Phone": 0.0, "Service": 0.0, "Subscription": 0.0, "Unknown": 1.0}
        sentiment = "Neutral"
        sentiment_scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": 1.0}
    else:
        # Classify intent
        intent, intent_scores = classify_text(customer_text, intent_labels)
        # Classify product
        product, product_scores = classify_text(customer_text, product_labels)
        # Classify sentiment
        sentiment, sentiment_scores = classify_text(customer_text, sentiment_labels)

    # Save output
    base_filename = os.path.splitext(json_path.name)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_analysis.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Intent: {intent}\nIntent Scores: {intent_scores}\n")
        f.write(f"Product: {product}\nProduct Scores: {product_scores}\n")
        f.write(f"Sentiment: {sentiment}\nSentiment Scores: {sentiment_scores}\n")

    print(f"Analysis for {base_filename}:")
    print(f"Intent: {intent}, Scores: {intent_scores}")
    print(f"Product: {product}, Scores: {product_scores}")
    print(f"Sentiment: {sentiment}, Scores: {sentiment_scores}")
    
    return transcription

# Process all JSON files in a directory
def process_all_json_files(directory: str):
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    for json_file in json_files:
        analyze_transcription(os.path.join(directory, json_file))

if __name__ == "__main__":
    # Single file test
    json_file = "transcription.json"
    analyze_transcription(json_file)
    # Or batch processing: process_all_json_files(base_dir)