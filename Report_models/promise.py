import json
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

# Download NLTK data
nltk.download('vader_lexicon')

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Define revised keyword dictionary for banking context
keywords = {
    "Settlement": {
        "pay now": 0.95,
        "pay today": 0.95,
        "pay immediately": 0.95,
        "pay in full": 0.9,
        "clear the balance": 0.9,
        "settle the account": 0.9,
        "pay this off": 0.9,
        "pay online": 0.85,
        "send payment": 0.85,
        "wire the payment": 0.85,
        "pay by tonight": 0.9,
        "complete the payment": 0.85,
        "settle now": 0.9,
        "make the payment": 0.85,
        "resolve it today": 0.9,
        "pay right away": 0.95,
        "set up auto-pay": 0.85,
        "pay via check": 0.85,
        "clear it today": 0.9,
        "finalize payment": 0.85,
        "pay the full balance": 0.9,
        "process payment now": 0.9,
        "pay via bank transfer": 0.85,
        "settle the debt": 0.9,
        "pay the entire amount": 0.9
    },
    "Partial Settlement": {
        "pay some": 0.7,
        "partial payment": 0.8,
        "pay next week": 0.65,
        "installments": 0.75,
        "pay a portion": 0.7,
        "pay half": 0.75,
        "set up a payment plan": 0.8,
        "pay in parts": 0.7,
        "pay over time": 0.65,
        "make a small payment": 0.7,
        "pay next month": 0.65,
        "split the payment": 0.75,
        "pay in installments": 0.75,
        "pay a little": 0.65,
        "pay something": 0.65,
        "pay gradually": 0.7,
        "make a down payment": 0.75,
        "pay a small amount": 0.7,
        "pay bit by bit": 0.65,
        "pay in monthly installments": 0.75,
        "cover a portion": 0.7,
        "pay a percentage": 0.7,
        "pay later": 0.6,
        "pay in stages": 0.7,
        "partial settlement": 0.8
    },
    "Promise Broken": {
        "meant to pay": 0.8,
        "forgot to pay": 0.75,
        "couldn’t pay": 0.7,
        "promised to pay": 0.9,
        "missed the payment": 0.8,
        "was supposed to pay": 0.85,
        "didn’t pay on time": 0.8,
        "intended to pay": 0.8,
        "tried to pay": 0.75,
        "missed the due date": 0.8,
        "failed to settle": 0.75,
        "promised earlier": 0.9,
        "didn’t follow through": 0.7,
        "planned to pay": 0.8,
        "meant to settle": 0.8,
        "forgot to clear": 0.75,
        "missed the deadline": 0.75,
        "couldn’t make payment": 0.7,
        "defaulted on payment": 0.8,
        "broke my promise": 0.85,
        "didn’t manage to pay": 0.75,
        "skipped the payment": 0.7,
        "intended to clear": 0.8,
        "payment fell through": 0.75,
        "overdue on promise": 0.8
    },
    "Denial": {
        "don’t owe": 0.95,
        "not my debt": 0.9,
        "refuse to pay": 0.95,
        "not responsible": 0.9,
        "billing error": 0.85,
        "incorrect charge": 0.85,
        "dispute the amount": 0.85,
        "not my account": 0.9,
        "wrongly charged": 0.85,
        "never owed": 0.9,
        "won’t pay": 0.95,
        "disagree with this": 0.85,
        "fraudulent charge": 0.9,
        "unauthorized transaction": 0.9,
        "already paid": 0.95,
        "invalid charge": 0.85,
        "mistake on bill": 0.85,
        "not liable": 0.9,
        "dispute this charge": 0.85,
        "incorrect billing": 0.85,
        "not my responsibility": 0.9,
        "error in billing": 0.85,
        "dispute the debt": 0.85,
        "account not mine": 0.9,
        "charge not recognized": 0.85
    }
}

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Read serialized transcription from transcriptions.json
try:
    with open('transcription.json', 'r') as file:
        serialized_transcription = json.load(file)
except FileNotFoundError:
    print("Error: transcriptions.json not found. Please provide the file or its content.")
    exit(1)

def extract_entities(dialogue):
    """Extract entities using spaCy NER and custom regex."""
    doc = nlp(dialogue)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "MONEY", "DATE", "CARDINAL"]:
            entities.append((ent.text, ent.label_))
    # Custom regex for phone numbers or partial card numbers
    if re.match(r'\d{10}', dialogue):
        entities.append((re.search(r'\d{10}', dialogue).group(), "PHONE"))
    if re.search(r'ending with \d{4}', dialogue):
        entities.append((re.search(r'\d{4}', dialogue).group(), "CARD"))
    return entities

def analyze_dialogue(serialized_transcription):
    """Analyze serialized transcription and return output structure with confidence."""
    critical_keywords = []
    lead_identifiers = []
    type_scores = {"Settlement": 0, "Partial Settlement": 0, "Promise Broken": 0, "Denial": 0}

    for entry in serialized_transcription:
        dialogue = entry["dialogue"].lower()
        speaker = entry["speaker"]

        # Extract entities for lead identifiers
        entities = extract_entities(entry["dialogue"])
        for entity, label in entities:
            if label in ["PERSON", "MONEY", "DATE", "CARDINAL", "PHONE", "CARD"]:
                lead_identifiers.append(entity)

        # Sentiment analysis
        sentiment = sia.polarity_scores(dialogue)["compound"]

        # Check for keywords
        for conv_type, kw_dict in keywords.items():
            for kw, weight in kw_dict.items():
                if kw in dialogue:
                    critical_keywords.append(kw)
                    # Adjust score based on speaker and sentiment
                    if speaker == "spk2":  # Customer
                        type_scores[conv_type] += weight * (1 if sentiment > 0 else 0.5)
                    else:  # Agent
                        type_scores[conv_type] += weight * 0.3  # Lower weight for agent

    # Fuzzy logic: Normalize scores
    total_score = sum(type_scores.values())
    normalized_scores = {}
    if total_score > 0:
        for conv_type in type_scores:
            normalized_scores[conv_type] = type_scores[conv_type] / total_score
    else:
        for conv_type in type_scores:
            normalized_scores[conv_type] = 0.0

    # Determine conversation type and confidence
    conv_type = max(normalized_scores, key=normalized_scores.get)
    confidence = normalized_scores[conv_type]
    if normalized_scores[conv_type] < 0.3:  # Threshold for low confidence
        conv_type = "Uncategorized"
        confidence = 0.0  # No confidence for uncategorized

    return {
        "type": conv_type,
        "critical_keywords": list(set(critical_keywords)),
        "lead_identifiers": list(set(lead_identifiers)),
        "confidence": round(confidence, 3)
    }

# Run analysis
result = analyze_dialogue(serialized_transcription)
print(json.dumps(result, indent=2))