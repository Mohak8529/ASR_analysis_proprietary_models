import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from fuzzywuzzy import fuzz
import re
import math

# Download required NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')

# Emotion parameters from the document
emotion_params = {
    "Anger": ["Loudness", "Pitch variation", "Speaking rate", "Spectral density"],
    "Frustration": ["Speaking rate", "Articulation rate", "Syllable rate", "Emotional intensity"],
    "Confusion": ["Semantic similarity", "Language identification", "Pronunciation accuracy"],
    "Stress": ["Speaking rate", "Emotional valence", "Spectral flux"],
    "Anxiety": ["Tremor", "Jitter", "Shimmer", "Spectral roll-off"],
    "Resignation": ["Speaking rate", "Emotional valence", "Language identification"],
    "Hopefulness": ["Pitch variation", "Emotional valence", "Articulation rate"],
    "Distrust": ["Irony detection", "Pronunciation accuracy", "Spectral tilt"],
    "Regret": ["Emotional valence", "Pitch variation", "Speaking rate"],
    "Empathy": ["Semantic similarity", "Emotional valence", "Language biases"],
    "Defensiveness": ["Pitch variation", "Loudness", "Zero-cross rate"],
    "Negotiation": ["Language identification", "Syllable rate", "Emotional valence"],
    "Impatience": ["Speaking rate", "Articulation rate", "Emotional intensity"],
    "Contentment": ["Emotional valence", "Tone", "Pitch"],
    "Optimism": ["Pitch variation", "Emotional valence", "Semantic similarity"],
    "Desperation": ["Speaking rate", "Emotional intensity", "Tremor"],
    "Regret2": ["Emotional valence", "Pronunciation accuracy", "Spectral centroid"],
    "Indifference": ["Emotional valence", "Pitch variation", "Spectral flatness"],
    "Pessimism": ["Emotional valence", "Pitch variation", "Formant frequencies"],
    "Curiosity": ["Semantic similarity", "Language identification", "Spectral flux"],
    "Satisfaction": ["Emotional valence", "Tone", "Pronunciation accuracy"],
    "Disappointment": ["Emotional valence", "Pitch variation", "Irony detection"],
    "Shame": ["Emotional valence", "Syllable rate", "Spectral kurtosis"],
    "Reassurance": ["Emotional valence", "Tone", "Articulation rate"],
    "Relief": ["Emotional valence", "Pitch variation", "Resonance"],
    "Hopelessness": ["Emotional valence", "Speaking rate", "Spectral skewness"],
    "Gratitude": ["Emotional valence", "Tone", "Pronunciation accuracy"],
    "Hostility": ["Loudness", "Pitch variation", "Zero-cross rate"],
    "Acceptance": ["Emotional valence", "Pitch variation", "Formant frequencies"],
    "Apathy": ["Emotional valence", "Spectral tilt", "Zero-cross rate"]
}

def load_transcription(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_speaking_rate(dialogue, start_time, end_time):
    # Proxy: words per second, normalized to 0-1 with logarithmic scaling
    words = len(word_tokenize(dialogue))
    duration = end_time - start_time
    rate = words / duration if duration > 0 else 0
    max_rate = 6.0  # Typical max speaking rate
    # Logarithmic scaling to smooth high rates
    return min(math.log1p(rate) / math.log1p(max_rate), 1.0)

def calculate_emotional_valence(dialogue):
    # Use VADER for sentiment analysis
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(dialogue)
    # Normalize compound score to 0-1
    return (scores['compound'] + 1) / 2

def calculate_semantic_similarity(dialogue, context="debt collection"):
    # Fuzzy match with context
    return fuzz.partial_ratio(dialogue.lower(), context) / 100

def calculate_emotional_intensity(dialogue):
    # Proxy: presence of strong words or punctuation
    strong_words = ["urgent", "please", "immediately", "sorry", "frustrated", "thanks", "good", "great"]
    score = sum(1 for word in word_tokenize(dialogue.lower()) if word in strong_words)
    if "!" in dialogue or "?" in dialogue:
        score += 0.2
    return min(score / 5, 1.0)

def calculate_pitch_variation(dialogue):
    # Proxy: sentence length variation
    sentences = sent_tokenize(dialogue)
    if len(sentences) < 2:
        return 0.5
    lengths = [len(word_tokenize(s)) for s in sentences]
    return min((max(lengths) - min(lengths)) / (sum(lengths) / len(lengths)), 1.0)

def calculate_loudness(dialogue):
    # Proxy: use of capital letters or exclamations
    if dialogue.isupper() or "!" in dialogue:
        return 0.8
    return 0.5

def calculate_tone(dialogue):
    # Proxy: Combine TextBlob subjectivity with positive word boost
    blob = TextBlob(dialogue)
    subjectivity = blob.sentiment.subjectivity
    positive_words = ["thanks", "good", "great", "happy"]
    if any(word in dialogue.lower() for word in positive_words):
        subjectivity += 0.2
    return min(subjectivity, 1.0)

def calculate_syllable_rate(dialogue, start_time, end_time):
    # Proxy: syllables per second, normalized to 0-1 with logarithmic scaling
    words = word_tokenize(dialogue)
    syllables = sum(len(re.findall(r'[aeiouy]+', w.lower())) for w in words)
    duration = end_time - start_time
    rate = syllables / duration if duration > 0 else 0
    max_rate = 10.0  # Typical max syllable rate
    return min(math.log1p(rate) / math.log1p(max_rate), 1.0)

def calculate_irony_detection(dialogue):
    # Proxy: mismatch between sentiment and context
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(dialogue)
    if "good" in dialogue.lower() and scores['compound'] < 0:
        return 0.7
    return 0.3

# Placeholder for audio-based parameters (assign neutral values)
def calculate_audio_param():
    return 0.5

def analyze_dialogue(transcription):
    results = {}
    for emotion, params in emotion_params.items():
        emotion_score = 0
        param_values = {}
        for param in params:
            # Calculate parameter value based on type
            if param == "Speaking rate":
                rates = [calculate_speaking_rate(d['dialogue'], d['startTime'], d['endTime']) 
                         for d in transcription if d['speaker'] == 'spk2']
                value = sum(rates) / len(rates) if rates else 0.5
            elif param == "Emotional valence":
                valences = [calculate_emotional_valence(d['dialogue']) 
                           for d in transcription if d['speaker'] == 'spk2']
                value = sum(valences) / len(valences) if valences else 0.5
            elif param == "Semantic similarity":
                similarities = [calculate_semantic_similarity(d['dialogue']) 
                              for d in transcription if d['speaker'] == 'spk2']
                value = sum(similarities) / len(similarities) if similarities else 0.5
            elif param == "Emotional intensity":
                intensities = [calculate_emotional_intensity(d['dialogue']) 
                              for d in transcription if d['speaker'] == 'spk2']
                value = sum(intensities) / len(intensities) if intensities else 0.5
            elif param == "Pitch variation":
                variations = [calculate_pitch_variation(d['dialogue']) 
                             for d in transcription if d['speaker'] == 'spk2']
                value = sum(variations) / len(variations) if variations else 0.5
            elif param == "Loudness":
                loudness = [calculate_loudness(d['dialogue']) 
                           for d in transcription if d['speaker'] == 'spk2']
                value = sum(loudness) / len(loudness) if loudness else 0.5
            elif param == "Tone":
                tones = [calculate_tone(d['dialogue']) 
                        for d in transcription if d['speaker'] == 'spk2']
                value = sum(tones) / len(tones) if tones else 0.5
            elif param == "Syllable rate":
                rates = [calculate_syllable_rate(d['dialogue'], d['startTime'], d['endTime']) 
                         for d in transcription if d['speaker'] == 'spk2']
                value = sum(rates) / len(rates) if rates else 0.5
            elif param == "Irony detection":
                ironies = [calculate_irony_detection(d['dialogue']) 
                          for d in transcription if d['speaker'] == 'spk2']
                value = sum(ironies) / len(ironies) if ironies else 0.5
            else:
                value = calculate_audio_param()  # Neutral for audio params
            param_values[param] = round(value, 2)
            emotion_score += value
        # Normalize emotion score
        emotion_score = round(emotion_score / len(params), 2)
        results[emotion] = {
            "Value": emotion_score,
            "Parameters": param_values
        }
    return results

def main():
    # Assuming transcription.json is in the same directory
    transcription = load_transcription('transcription.json')
    results = analyze_dialogue(transcription)
    
    # Output results
    print(json.dumps(results, indent=2))
    
    # Save results to file
    with open('emotion_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()