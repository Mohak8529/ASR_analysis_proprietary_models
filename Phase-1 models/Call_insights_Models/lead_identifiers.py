import os
import json
import torch
from transformers import pipeline
from typing import List
from pathlib import Path

# Directories
base_dir = os.getcwd()
output_dir = os.path.join(base_dir, "LeadKeywords")
os.makedirs(output_dir, exist_ok=True)

# Local model directory
model_dir = os.path.join(base_dir, "bart-large-mnli")

# Device config (safe GPU/CPU handling)
try:
    if torch.cuda.is_available():
        torch.zeros(1).to("cuda")
        device = 0
    else:
        device = -1
except Exception:
    print("⚠ GPU not available or busy — falling back to CPU.")
    device = -1

# Load zero-shot classifier pipeline from local directory
try:
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=model_dir,
        device=device
    )
except Exception as e:
    print(f"Error loading model from {model_dir}: {e}")
    print("Ensure the 'bart-large-mnli' directory contains the model files.")
    exit(1)

# Candidate labels for lead detection — descriptive for the model
candidate_labels = [
    "Lead Identifier in a bank call such as name, phone number, date of birth, customer ID, account number,product name, lead ID, email address, or other identifying details",
    "Not a Lead Identifier in a bank call"
]

# Minimal stop words to remove only noise words
stop_words = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'from', 'of',
    'with', 'to', 'by', 'about', 'is', 'are', 'was', 'were', 'this', 'that', 'these',
    'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
    'them', 'your', 'please', 'okay', 'hi', 'what', 'whats', 'if', 'can', 'have', 'would', 'could', 'should', 'do', 'does', 'did', 'doing', 'also', 'be',
}

def classify_lead_word(word: str) -> tuple:
    """Classify a single word as a lead identifier or not."""
    try:
        result = zero_shot_classifier(word, candidate_labels, multi_label=False)
        print(f"Raw pipeline result for word '{word}': {result}")
        top_label = result['labels'][0]
        score = result['scores'][0]
        scores_dict = dict(zip(result['labels'], result['scores']))
        return top_label, score, scores_dict
    except Exception as e:
        print(f"Error classifying word '{word}': {e}")
        return None, None, None

def analyze_leads(json_file: str) -> List[dict]:
    """Analyze a single JSON file and extract lead identifier keywords."""
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

    # Break into words and filter stop words
    words = [word.lower() for word in full_conversation.split() if word.lower() not in stop_words]

    # Classify each word and collect lead keywords
    lead_keywords_dict = {}  # word -> score
    threshold = 0.75

    for word in words:
        clean_word = word.strip(",.?!")  # remove trailing punctuation
        top_label, score, _ = classify_lead_word(clean_word)
        if top_label is None:
            continue
        if top_label.startswith("Lead Identifier") and score >= threshold:
            # keep only the highest score if word repeats
            if clean_word not in lead_keywords_dict or score > lead_keywords_dict[clean_word]:
                lead_keywords_dict[clean_word] = score

    # Sort by score
    lead_keywords = sorted(lead_keywords_dict.items(), key=lambda x: x[1], reverse=True)

    # Extract just the words for console output
    lead_keyword_list = [kw[0] for kw in lead_keywords]

    # Save output
    base_filename = os.path.splitext(json_path.name)[0]
    output_file = os.path.join(output_dir, f"{base_filename}_lead_keywords.txt")
    with open(output_file, "w", encoding="utf-8") as f_output:
        f_output.write("Lead Identifier Keywords:\n")
        f_output.write("\n".join(f"{kw} (score: {score:.3f})" for kw, score in lead_keywords))

    print(f"[✔] {base_filename} → Lead Identifier Keywords: {lead_keyword_list}")
    return transcription

def process_all_json_files(directory: str):
    """Process all JSON files in the specified directory."""
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {directory}")
        return

    for json_file in json_files:
        analyze_leads(os.path.join(directory, json_file))

if __name__ == "__main__":
    # Process a single file
    json_file = "transcription.json"
    analyze_leads(json_file)

    # Optionally process all JSON files in the directory
    # process_all_json_files(base_dir)
