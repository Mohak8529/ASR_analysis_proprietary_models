#!/usr/bin/env python3
import json
import uuid
from datetime import datetime
import re
from typing import Dict, List, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import math
import language_tool_python
import librosa
import numpy as np
from scipy import stats
import langid
import textstat
from transformers import BartForConditionalGeneration, BartTokenizer
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
import re
from transformers import pipeline, AutoTokenizer, logging as hf_logging
import re
import textwrap
import torch
import torchaudio
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForAudioClassification, AutoFeatureExtractor

# Configure NLTK to use local data directory
nltk_data_dir = "nltk_data"
nltk.data.path.append(nltk_data_dir)

# Model paths for locally downloaded models
TEXT_MODEL_PATH = "/mnt/ssd1/temp_call_insights/Call_insights_Models/emotion_models/emotion-english-distilroberta-base"
AUDIO_MODEL_PATH = "/mnt/ssd1/temp_call_insights/Call_insights_Models/emotion_models/hubert-large-superb-er"

# Load text model and tokenizer
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
text_pipeline = pipeline("text-classification", model=text_model, tokenizer=text_tokenizer, top_k=None, device=0 if torch.cuda.is_available() else -1)

# Load audio model and feature extractor
audio_model = AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL_PATH)
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_PATH)
audio_pipeline = pipeline("audio-classification", model=audio_model, feature_extractor=audio_feature_extractor, top_k=None, device=0 if torch.cuda.is_available() else -1)

# Emotion parameters
emotion_params = {
    "Anger": ["Loudness", "Pitch variation", "Speaking rate", "Spectral density"],
    "Frustration": ["Speaking rate", "Articulation rate", "Syllable rate", "Emotional intensity"],
    "Confusion": ["Semantic similarity", "Language identification", "Pronunciation accuracy"],
    "Stress": ["Speaking rate", "Emotional valence", "Spectral flux"],
    "Anxiety": ["Tremor", "Jitter", "Shimmer", "Spectral roll-off"],
    "Resignation": ["Speaking rate", "Emotional valence", "Language identification"],
    "Hopefulness": ["Pitch variation", "Emotional valence", "Articulation rate"],
    "Distrust": ["Irony detection", "Pronunciation accuracy", "Spectral tilt"],
    "Regret": ["Emotional valence", "Pitch variation", "Speaking rate", "Pronunciation accuracy", "Spectral centroid"],
    "Empathy": ["Semantic similarity", "Emotional valence", "Language biases"],
    "Defensiveness": ["Pitch variation", "Loudness", "Zero-cross rate"],
    "Negotiation": ["Language identification", "Syllable rate", "Emotional valence"],
    "Impatience": ["Speaking rate", "Articulation rate", "Emotional intensity"],
    "Contentment": ["Emotional valence", "Tone", "Pitch variation"],
    "Optimism": ["Pitch variation", "Emotional valence", "Semantic similarity"],
    "Desperation": ["Speaking rate", "Emotional intensity", "Tremor"],
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

# Parameter calculation functions

def calculate_energy(audio_segment, sr, dialogue):
    """Calculate energy from audio signal using RMS."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    rms = librosa.feature.rms(y=audio_segment).mean()
    max_rms = 0.1  # Typical max RMS for speech
    return min(float(rms / max_rms * 2), 2.0)  # Convert to float

def calculate_entropy(audio_segment, sr, dialogue):
    """Calculate spectral entropy from audio signal."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    S_power = S ** 2
    S_power = S_power / (np.sum(S_power, axis=0) + 1e-10)  # Normalize to probability
    entropy = -np.sum(S_power * np.log2(S_power + 1e-10), axis=0).mean()
    max_entropy = 10.0  # Typical max spectral entropy
    return min(float(entropy / max_entropy * 3), 3.0)  # Convert to float

def calculate_loudness(audio_segment, sr, dialogue):
    """Calculate loudness from audio amplitude (decibels)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    # Normalize audio to [-1, 1]
    audio_segment = audio_segment / (np.max(np.abs(audio_segment)) + 1e-10)
    # Compute RMS with proper framing
    rms = librosa.feature.rms(y=audio_segment, frame_length=2048, hop_length=512).mean()
    # Convert to dB with a fixed reference (e.g., 1.0 for normalized audio)
    db = librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0]
    # Normalize to [0, 1] using realistic dB range (-80 dB to 0 dB)
    min_db, max_db = -80, 0
    loudness = (db - min_db) / (max_db - min_db)
    return min(max(float(loudness), 0.0), 1.0)  # Clamp to [0, 1]

def calculate_pitch_variation(audio_segment, sr, dialogue):
    """Calculate pitch variation from audio (standard deviation of F0)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
    pitches = pitches[magnitudes > 0]
    if len(pitches) == 0:
        return 0.5
    return min(float(np.std(pitches) / 200), 1.0)  # Convert to float

def calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time):
    """Calculate speaking rate (words per second) from transcription and time."""
    words = len(word_tokenize(dialogue))
    duration = end_time - start_time
    rate = words / duration if duration > 0 else 0
    max_rate = 6.0
    return min(float(rate / max_rate), 1.0)  # Convert to float

def calculate_spectral_density(audio_segment, sr, dialogue):
    """Calculate spectral density (power spectrum energy)."""
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))**2
    density = np.mean(S)
    max_density = max(density, 1e-10) * 1000  # Scale for speech
    return min(float(density / max_density), 1.0)

def calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time):
    """Calculate articulation rate (syllables per second, excluding pauses)."""
    syllables = textstat.syllable_count(dialogue)
    duration = end_time - start_time
    rate = syllables / duration if duration > 0 else 0
    max_rate = 8.0
    return min(float(rate / max_rate), 1.0)  # Convert to float

def calculate_syllable_rate(audio_segment, sr, dialogue, start_time, end_time):
    """Calculate syllable rate (similar to articulation rate but audio-based)."""
    syllables = textstat.syllable_count(dialogue)
    duration = librosa.get_duration(y=audio_segment, sr=sr)
    rate = syllables / duration if duration > 0 else 0
    max_rate = 8.0
    return min(float(rate / max_rate), 1.0)  # Convert to float

def calculate_emotional_intensity(audio_segment, sr, dialogue):
    """Calculate emotional intensity from audio energy and text cues."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    rms = librosa.feature.rms(y=audio_segment).mean()
    strong_words = ["urgent", "please", "immediately", "sorry", "thanks"]
    text_score = sum(1 for word in word_tokenize(dialogue.lower()) if word in strong_words) / 5
    audio_score = min(rms / 0.1, 1.0)
    return float(0.5 * text_score + 0.5 * audio_score)  # Convert to float

def calculate_semantic_similarity(audio_segment, sr, dialogue):
    """Calculate semantic similarity to expected debt-related terms."""
    debt_terms = ["loan", "payment", "due", "balance", "debt"]
    tokens = word_tokenize(dialogue.lower())
    similarity = sum(1 for token in tokens if token in debt_terms) / len(debt_terms)
    return min(float(similarity), 1.0)  # Convert to float

def calculate_language_identification(audio_segment, sr, dialogue):
    """Calculate confidence that the language is English."""
    lang, confidence = langid.classify(dialogue)
    return float(min(max(confidence, 0.0), 1.0) if lang == "en" else 0.0)

def calculate_pronunciation_accuracy(audio_segment, sr, dialogue):
    """Calculate pronunciation accuracy (placeholder)."""
    if len(dialogue.strip()) == 0 or len(audio_segment) < 64:
        return 0.5
    # Placeholder: Actual implementation requires phoneme alignment
    return 0.5  # Return neutral value until proper logic is implemented

def calculate_emotional_valence(audio_segment, sr, dialogue):
    """Calculate emotional valence from text sentiment."""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(dialogue)
    compound = scores.get('compound', 0.0)  # Safe access with default
    sentiment = (compound + 1) / 2  # Normalize to [0, 1]
    # Ensure neutral for short or low-sentiment phrases
    if len(dialogue.split()) <= 3 or abs(compound) < 0.1:  # Include near-neutral
        sentiment = 0.5
    return float(max(min(sentiment, 1.0), 0.0))  # Clamp to [0, 1]

def calculate_spectral_flux(audio_segment, sr, dialogue):
    """Calculate spectral flux (rate of spectral change)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    flux = librosa.onset.onset_strength(y=audio_segment, sr=sr, S=S)
    return min(float(np.mean(flux) / 10), 1.0)  # Convert to float

def calculate_tremor(audio_segment, sr, dialogue):
    """Calculate vocal tremor (pitch fluctuation)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    pitches, _ = librosa.piptrack(y=audio_segment, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        return 0.5
    tremor = np.std(np.diff(pitches)) / 50
    return min(float(tremor), 1.0)  # Convert to float

def calculate_jitter(audio_segment, sr, dialogue):
    """Calculate jitter (pitch perturbation)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    pitches, _ = librosa.piptrack(y=audio_segment, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) < 2:
        return 0.5
    jitter = np.mean(np.abs(np.diff(pitches)) / pitches[:-1])
    return min(float(jitter / 0.05), 1.0)  # Convert to float

def calculate_shimmer(audio_segment, sr, dialogue):
    """Calculate shimmer (amplitude perturbation)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    rms = librosa.feature.rms(y=audio_segment)
    if len(rms) < 2:
        return 0.5
    shimmer = np.mean(np.abs(np.diff(rms)) / rms[:-1])
    return min(float(shimmer / 0.1), 1.0)  # Convert to float

def calculate_spectral_rolloff(audio_segment, sr, dialogue):
    """Calculate spectral roll-off (frequency below which 85% of energy lies)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr).mean()
    max_rolloff = sr / 2
    return min(float(rolloff / max_rolloff), 1.0)  # Convert to float

def calculate_irony_detection(audio_segment, sr, dialogue):
    """Estimate irony via sentiment-tone mismatch (simplified)."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(dialogue)['compound']
    blob = TextBlob(dialogue)
    subjectivity = blob.sentiment.subjectivity
    mismatch = abs(sentiment - subjectivity)
    return min(float(mismatch), 1.0)  # Convert to float

def calculate_spectral_tilt(audio_segment, sr, dialogue):
    """Calculate spectral tilt (slope of spectrum)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_spectrum = np.mean(S, axis=1)
    if len(freqs) != len(mean_spectrum):  # Ensure dimensions match
        min_len = min(len(freqs), len(mean_spectrum))
        freqs = freqs[:min_len]
        mean_spectrum = mean_spectrum[:min_len]
    if len(mean_spectrum) < 2:  # Safeguard for insufficient data
        return 0.5
    tilt = np.polyfit(freqs, 10 * np.log10(mean_spectrum + 1e-10), 1)[0]
    return min(float(abs(tilt) / 0.01), 1.0)  # Convert to float

def calculate_language_biases(audio_segment, sr, dialogue):
    """Estimate language biases via sentiment polarity deviation."""
    blob = TextBlob(dialogue)
    polarity = blob.sentiment.polarity
    neutral_score = 0.0
    bias = abs(polarity - neutral_score)
    return min(float(bias), 1.0)  # Convert to float

def calculate_zero_cross_rate(audio_segment, sr, dialogue):
    """Calculate zero-cross rate (signal sign changes)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    zcr = librosa.feature.zero_crossing_rate(y=audio_segment).mean()
    max_zcr = 0.5
    return min(float(zcr / max_zcr), 1.0)  # Convert to float

def calculate_spectral_centroid(audio_segment, sr, dialogue):
    """Calculate spectral centroid (center of mass of spectrum)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr, n_fft=n_fft).mean()
    max_centroid = sr / 2
    return min(float(centroid / max_centroid), 1.0)  # Convert to float

def calculate_spectral_flatness(audio_segment, sr, dialogue):
    """Calculate spectral flatness of the audio segment."""
    if len(audio_segment) < 64:
        return 0.5
    flatness = librosa.feature.spectral_flatness(y=audio_segment).mean()
    return min(max(float(flatness), 0.0), 1.0)

def calculate_formant_frequencies(audio_segment, sr, dialogue):
    """Calculate average formant frequency (simplified)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peaks = freqs[np.argmax(S, axis=0)]
    avg_formant = np.mean(peaks) if len(peaks) > 0 else 500
    max_formant = 5000
    return min(float(avg_formant / max_formant), 1.0)  # Convert to float

def calculate_spectral_kurtosis(audio_segment, sr, dialogue):
    """Calculate spectral kurtosis (peakedness of spectrum)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    if S.shape[1] < 2:  # Safeguard for insufficient frames
        return 0.5
    kurtosis = np.mean([stats.kurtosis(S[:, i]) for i in range(S.shape[1])])
    return min(float((kurtosis + 3) / 10), 1.0)  # Convert to float

def calculate_tone(audio_segment, sr, dialogue):
    """Calculate tone (subjectivity) from text and audio prosody."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    blob = TextBlob(dialogue)
    subjectivity = blob.sentiment.subjectivity
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr, n_fft=n_fft).mean()
    return min(float((subjectivity + chroma) / 2), 1.0)  # Convert to float

def calculate_resonance(audio_segment, sr, dialogue):
    """Calculate resonance based on formant frequencies."""
    if len(audio_segment) < 64:
        return 0.5
    formants = librosa.effects.preemphasis(audio_segment)
    resonance = np.mean(np.abs(formants)) / (np.max(np.abs(formants)) + 1e-10)
    return min(max(float(resonance), 0.0), 1.0)

def calculate_spectral_skewness(audio_segment, sr, dialogue):
    """Calculate spectral skewness (asymmetry of spectrum)."""
    if len(audio_segment) < 64:  # Safeguard for very short segments
        return 0.5
    n_fft = min(2048, len(audio_segment))  # Adjust n_fft dynamically
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    if S.shape[1] < 2:  # Safeguard for insufficient frames
        return 0.5
    skewness = np.mean([stats.skew(S[:, i]) for i in range(S.shape[1])])
    return min(float((skewness + 3) / 6), 1.0)  # Convert to float

# Emotion scoring functions
def calculate_anger_score(audio_segment, sr, dialogue, start_time, end_time):
    loudness = calculate_loudness(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    spectral_density = calculate_spectral_density(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.2, 0.1]
    return np.average([loudness, pitch_var, speaking_rate, spectral_density], weights=weights)

def calculate_frustration_score(audio_segment, sr, dialogue, start_time, end_time):
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    articulation_rate = calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time)
    syllable_rate = calculate_syllable_rate(audio_segment, sr, dialogue, start_time, end_time)
    emotional_intensity = calculate_emotional_intensity(audio_segment, sr, dialogue)
    weights = [0.3, 0.3, 0.2, 0.2]
    return np.average([speaking_rate, articulation_rate, syllable_rate, emotional_intensity], weights=weights)

def calculate_confusion_score(audio_segment, sr, dialogue, start_time, end_time):
    semantic_similarity = calculate_semantic_similarity(audio_segment, sr, dialogue)
    language_id = calculate_language_identification(audio_segment, sr, dialogue)
    pronunciation_accuracy = calculate_pronunciation_accuracy(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - semantic_similarity, language_id, 1 - pronunciation_accuracy], weights=weights)

def calculate_stress_score(audio_segment, sr, dialogue, start_time, end_time):
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    spectral_flux = calculate_spectral_flux(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([speaking_rate, 1 - emotional_valence, spectral_flux], weights=weights)

def calculate_anxiety_score(audio_segment, sr, dialogue, start_time, end_time):
    tremor = calculate_tremor(audio_segment, sr, dialogue)
    jitter = calculate_jitter(audio_segment, sr, dialogue)
    shimmer = calculate_shimmer(audio_segment, sr, dialogue)
    spectral_rolloff = calculate_spectral_rolloff(audio_segment, sr, dialogue)
    weights = [0.3, 0.3, 0.2, 0.2]
    return np.average([tremor, jitter, shimmer, spectral_rolloff], weights=weights)

def calculate_resignation_score(audio_segment, sr, dialogue, start_time, end_time):
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    language_id = calculate_language_identification(audio_segment, sr, dialogue)
    weights = [0.3, 0.4, 0.3]
    return np.average([1 - speaking_rate, 1 - emotional_valence, language_id], weights=weights)

def calculate_hopefulness_score(audio_segment, sr, dialogue, start_time, end_time):
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    articulation_rate = calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time)
    weights = [0.3, 0.4, 0.3]
    return np.average([pitch_var, emotional_valence, articulation_rate], weights=weights)

def calculate_distrust_score(audio_segment, sr, dialogue, start_time, end_time):
    irony_detection = calculate_irony_detection(audio_segment, sr, dialogue)
    pronunciation_accuracy = calculate_pronunciation_accuracy(audio_segment, sr, dialogue)
    spectral_tilt = calculate_spectral_tilt(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([irony_detection, 1 - pronunciation_accuracy, spectral_tilt], weights=weights)

def calculate_regret_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    pronunciation_accuracy = calculate_pronunciation_accuracy(audio_segment, sr, dialogue)
    spectral_centroid = calculate_spectral_centroid(audio_segment, sr, dialogue)
    weights = [0.3, 0.2, 0.2, 0.2, 0.1]
    return np.average([1 - emotional_valence, pitch_var, 1 - speaking_rate, pronunciation_accuracy, spectral_centroid], weights=weights)

def calculate_empathy_score(audio_segment, sr, dialogue, start_time, end_time):
    semantic_similarity = calculate_semantic_similarity(audio_segment, sr, dialogue)
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    language_biases = calculate_language_biases(audio_segment, sr, dialogue)
    weights = [0.3, 0.4, 0.3]
    return np.average([semantic_similarity, emotional_valence, 1 - language_biases], weights=weights)

def calculate_defensiveness_score(audio_segment, sr, dialogue, start_time, end_time):
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    loudness = calculate_loudness(audio_segment, sr, dialogue)
    zero_cross_rate = calculate_zero_cross_rate(audio_segment, sr, dialogue)
    weights = [0.3, 0.4, 0.3]
    return np.average([pitch_var, loudness, zero_cross_rate], weights=weights)

def calculate_negotiation_score(audio_segment, sr, dialogue, start_time, end_time):
    language_id = calculate_language_identification(audio_segment, sr, dialogue)
    syllable_rate = calculate_syllable_rate(audio_segment, sr, dialogue, start_time, end_time)
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    weights = [0.3, 0.3, 0.4]
    return np.average([language_id, syllable_rate, emotional_valence], weights=weights)

def calculate_impatience_score(audio_segment, sr, dialogue, start_time, end_time):
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    articulation_rate = calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time)
    emotional_intensity = calculate_emotional_intensity(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([speaking_rate, articulation_rate, emotional_intensity], weights=weights)

def calculate_contentment_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    tone = calculate_tone(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([emotional_valence, tone, pitch_var], weights=weights)

def calculate_optimism_score(audio_segment, sr, dialogue, start_time, end_time):
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    semantic_similarity = calculate_semantic_similarity(audio_segment, sr, dialogue)
    weights = [0.3, 0.4, 0.3]
    return np.average([pitch_var, emotional_valence, semantic_similarity], weights=weights)

def calculate_desperation_score(audio_segment, sr, dialogue, start_time, end_time):
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    emotional_intensity = calculate_emotional_intensity(audio_segment, sr, dialogue)
    tremor = calculate_tremor(audio_segment, sr, dialogue)
    weights = [0.3, 0.4, 0.3]
    return np.average([speaking_rate, emotional_intensity, tremor], weights=weights)

def calculate_indifference_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    spectral_flatness = calculate_spectral_flatness(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - emotional_valence, 1 - pitch_var, spectral_flatness], weights=weights)

def calculate_pessimism_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    formant_freqs = calculate_formant_frequencies(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - emotional_valence, 1 - pitch_var, formant_freqs], weights=weights)

def calculate_curiosity_score(audio_segment, sr, dialogue, start_time, end_time):
    semantic_similarity = calculate_semantic_similarity(audio_segment, sr, dialogue)
    language_id = calculate_language_identification(audio_segment, sr, dialogue)
    spectral_flux = calculate_spectral_flux(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([semantic_similarity, language_id, spectral_flux], weights=weights)

def calculate_satisfaction_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    tone = calculate_tone(audio_segment, sr, dialogue)
    pronunciation_accuracy = calculate_pronunciation_accuracy(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([emotional_valence, tone, pronunciation_accuracy], weights=weights)

def calculate_disappointment_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    irony_detection = calculate_irony_detection(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - emotional_valence, 1 - pitch_var, irony_detection], weights=weights)

def calculate_shame_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    syllable_rate = calculate_syllable_rate(audio_segment, sr, dialogue, start_time, end_time)
    spectral_kurtosis = calculate_spectral_kurtosis(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - emotional_valence, 1 - syllable_rate, spectral_kurtosis], weights=weights)

def calculate_reassurance_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    tone = calculate_tone(audio_segment, sr, dialogue)
    articulation_rate = calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time)
    weights = [0.4, 0.3, 0.3]
    return np.average([emotional_valence, tone, articulation_rate], weights=weights)

def calculate_relief_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    resonance = calculate_resonance(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([emotional_valence, pitch_var, resonance], weights=weights)

def calculate_hopelessness_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    speaking_rate = calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time)
    spectral_skewness = calculate_spectral_skewness(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - emotional_valence, 1 - speaking_rate, spectral_skewness], weights=weights)

def calculate_gratitude_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    tone = calculate_tone(audio_segment, sr, dialogue)
    pronunciation_accuracy = calculate_pronunciation_accuracy(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([emotional_valence, tone, pronunciation_accuracy], weights=weights)

def calculate_hostility_score(audio_segment, sr, dialogue, start_time, end_time):
    loudness = calculate_loudness(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    zero_cross_rate = calculate_zero_cross_rate(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([loudness, pitch_var, zero_cross_rate], weights=weights)

def calculate_acceptance_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    pitch_var = calculate_pitch_variation(audio_segment, sr, dialogue)
    formant_freqs = calculate_formant_frequencies(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([emotional_valence, pitch_var, formant_freqs], weights=weights)

def calculate_apathy_score(audio_segment, sr, dialogue, start_time, end_time):
    emotional_valence = calculate_emotional_valence(audio_segment, sr, dialogue)
    spectral_tilt = calculate_spectral_tilt(audio_segment, sr, dialogue)
    zero_cross_rate = calculate_zero_cross_rate(audio_segment, sr, dialogue)
    weights = [0.4, 0.3, 0.3]
    return np.average([1 - emotional_valence, spectral_tilt, zero_cross_rate], weights=weights)

def analyze_emotions(transcription, audio_file):
    """Analyze emotions for each dialogue using pre-trained text and audio models."""
    # Load audio file
    audio, sr = librosa.load(audio_file)
    
    # Emotion label mappings
    TEXT_LABEL_MAPPING = {
        "anger": "anger",
        "joy": "curiosity",
        "optimism": "optimism",
        "sadness": "contentment",
        "fear": "anxiety",
        "love": "empathy",
        "surprise": "curiosity",
        "trust": "reassurance",
        "disgust": "hostility",
        "anticipation": "negotiation"
    }

    AUDIO_LABEL_MAPPING = {
        "angry": "anger",
        "happy": "contentment",
        "excited": "optimism",
        "sad": "contentment",
        "neutral": "indifference",
        "fearful": "anxiety",
        "disgust": "hostility",
        "surprised": "curiosity"
    }

    ALLOWED_EMOTIONS = [
        "anger", "frustration", "confusion", "stress", "anxiety", "resignation",
        "hopefulness", "distrust", "regret", "empathy", "defensiveness", "negotiation",
        "impatience", "contentment", "optimism", "desperation", "indifference", "pessimism",
        "curiosity", "satisfaction", "disappointment", "shame", "reassurance", "relief",
        "hopelessness", "gratitude", "hostility", "acceptance", "apathy"
    ]

    def predict_text_emotion(text):
        """Predict emotion from text using the text model."""
        try:
            preds = text_pipeline(text)[0]
            mapped = []
            for p in preds:
                lbl = p['label'].lower()
                if lbl in TEXT_LABEL_MAPPING:
                    mapped_lbl = TEXT_LABEL_MAPPING[lbl]
                    mapped.append({'label': mapped_lbl, 'score': p['score']})
            if not mapped:
                return "Unknown", 0.0
            best = max(mapped, key=lambda x: x['score'])
            return best['label'], round(best['score'], 3)
        except:
            return "Unknown", 0.0

    def predict_audio_emotion(audio_segment, sr):
        """Predict emotion from audio segment using the audio model."""
        try:
            # Save temporary audio segment
            tmp_path = os.path.join("/tmp", f"temp_segment_{uuid.uuid4()}.wav")
            torchaudio.save(tmp_path, torch.tensor(audio_segment).unsqueeze(0), sr)
            preds = audio_pipeline({"array": audio_segment, "sampling_rate": sr})
            mapped = []
            for p in preds:
                lbl = p['label'].lower()
                if lbl in AUDIO_LABEL_MAPPING:
                    mapped_lbl = AUDIO_LABEL_MAPPING[lbl]
                    mapped.append({'label': mapped_lbl, 'score': p['score']})
            os.remove(tmp_path)  # Clean up
            if not mapped:
                return "Unknown", 0.0
            best = max(mapped, key=lambda x: x['score'])
            return best['label'], round(best['score'], 3)
        except:
            return "Unknown", 0.0

    def merge_emotions(text_emo, text_conf, audio_emo, audio_conf):
        """Merge text and audio emotion predictions."""
        if text_emo == "Unknown" and audio_emo == "Unknown":
            return "Unknown", 0.0
        if text_emo == audio_emo and text_emo != "Unknown":
            return text_emo, round((text_conf + audio_conf) / 2, 3)
        return (text_emo, text_conf) if text_conf >= audio_conf else (audio_emo, audio_conf)

    emotion_results = []
    
    for entry in transcription:
        dialogue = entry['dialogue']
        speaker = entry['speaker']
        start_time = entry['startTime']
        end_time = entry['endTime']
        
        # Extract audio segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]
        
        # Calculate existing audio metrics
        energy = calculate_energy(audio_segment, sr, dialogue)
        entropy = calculate_entropy(audio_segment, sr, dialogue)
        loudness = calculate_loudness(audio_segment, sr, dialogue)
        sentiment = calculate_emotional_valence(audio_segment, sr, dialogue)
        
        # Predict emotions using models
        t_emo, t_conf = predict_text_emotion(dialogue)
        a_emo, a_conf = predict_audio_emotion(audio_segment, sr)
        m_emo, _ = merge_emotions(t_emo, t_conf, a_emo, a_conf)
        
        # Ensure the predicted emotion is in ALLOWED_EMOTIONS
        if m_emo not in ALLOWED_EMOTIONS:
            m_emo = "Unknown"
        
        # Append to results
        emotion_results.append({
            "dialogue": dialogue,
            "speaker": speaker,
            "startTime": start_time,
            "endTime": end_time,
            "early_dialogue": dialogue,
            "emotion": m_emo,
            "energy": round(energy, 2),
            "entropy": round(entropy, 2),
            "loudness": round(loudness, 2),
            "sentiment": round(sentiment, 2)
        })
    
    return emotion_results

# Placeholder audio metrics calculation
def calculate_audio_metrics(transcription, audio_file):
    """Calculate all 27 audio metrics for each speaker using audio-based methods."""
    # Load audio file
    try:
        audio, sr = librosa.load(audio_file)
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        return {
            param.lower().replace(" ", "_"): [
                {"speaker": "spk1", param.lower().replace(" ", "_"): 0.0},
                {"speaker": "spk2", param.lower().replace(" ", "_"): 0.0}
            ] for param in [
                "Energy", "Entropy", "Loudness", "Pitch variation", "Speaking rate", "Spectral density",
                "Articulation rate", "Syllable rate", "Emotional intensity", "Semantic similarity",
                "Language identification", "Pronunciation accuracy", "Emotional valence", "Spectral flux",
                "Tremor", "Jitter", "Shimmer", "Spectral roll-off", "Irony detection", "Spectral tilt",
                "Language biases", "Zero-cross rate", "Spectral centroid", "Spectral flatness",
                "Formant frequencies", "Spectral kurtosis", "Tone", "Resonance", "Spectral skewness"
            ]
        }
    
    # Separate dialogues by speaker
    agent_dialogues = [e for e in transcription if e['speaker'] == 'spk1']
    customer_dialogues = [e for e in transcription if e['speaker'] == 'spk2']
    
    # Initialize results
    result = {
        param.lower().replace(" ", "_"): [
            {"speaker": "spk1", param.lower().replace(" ", "_"): 0.0},
            {"speaker": "spk2", param.lower().replace(" ", "_"): 0.0}
        ] for param in [
            "Energy", "Entropy", "Loudness", "Pitch variation", "Speaking rate", "Spectral density",
            "Articulation rate", "Syllable rate", "Emotional intensity", "Semantic similarity",
            "Language identification", "Pronunciation accuracy", "Emotional valence", "Spectral flux",
            "Tremor", "Jitter", "Shimmer", "Spectral roll-off", "Irony detection", "Spectral tilt",
            "Language biases", "Zero-cross rate", "Spectral centroid", "Spectral flatness",
            "Formant frequencies", "Spectral kurtosis", "Tone", "Resonance", "Spectral skewness"
        ]
    }
    
    # Parameter calculation functions mapping
    param_functions = {
        "Energy": calculate_energy,
        "Entropy": calculate_entropy,
        "Loudness": calculate_loudness,
        "Pitch variation": calculate_pitch_variation,
        "Speaking rate": calculate_speaking_rate,
        "Spectral density": calculate_spectral_density,
        "Articulation rate": calculate_articulation_rate,
        "Syllable rate": calculate_syllable_rate,
        "Emotional intensity": calculate_emotional_intensity,
        "Semantic similarity": calculate_semantic_similarity,
        "Language identification": calculate_language_identification,
        "Pronunciation accuracy": calculate_pronunciation_accuracy,
        "Emotional valence": calculate_emotional_valence,
        "Spectral flux": calculate_spectral_flux,
        "Tremor": calculate_tremor,
        "Jitter": calculate_jitter,
        "Shimmer": calculate_shimmer,
        "Spectral roll-off": calculate_spectral_rolloff,
        "Irony detection": calculate_irony_detection,
        "Spectral tilt": calculate_spectral_tilt,
        "Language biases": calculate_language_biases,
        "Zero-cross rate": calculate_zero_cross_rate,
        "Spectral centroid": calculate_spectral_centroid,
        "Spectral flatness": calculate_spectral_flatness,
        "Formant frequencies": calculate_formant_frequencies,
        "Spectral kurtosis": calculate_spectral_kurtosis,
        "Tone": calculate_tone,
        "Resonance": calculate_resonance,
        "Spectral skewness": calculate_spectral_skewness
    }
    
    # Calculate metrics for agent (spk1)
    if agent_dialogues:
        agent_metrics = {param: [] for param in param_functions.keys()}
        for entry in agent_dialogues:
            start_sample = int(entry['startTime'] * sr)
            end_sample = int(entry['endTime'] * sr)
            audio_segment = audio[start_sample:end_sample]
            dialogue = entry['dialogue']
            start_time = entry['startTime']
            end_time = entry['endTime']
            
            # Calculate all parameters
            for param, func in param_functions.items():
                if param in ["Speaking rate", "Articulation rate", "Syllable rate"]:
                    agent_metrics[param].append(func(audio_segment, sr, dialogue, start_time, end_time))
                else:
                    agent_metrics[param].append(func(audio_segment, sr, dialogue))
        
        # Aggregate parameters
        for param in param_functions.keys():
            result[param.lower().replace(" ", "_")][0][param.lower().replace(" ", "_")] = round(sum(agent_metrics[param]) / len(agent_metrics[param]), 2)
    
    # Calculate metrics for customer (spk2)
    if customer_dialogues:
        customer_metrics = {param: [] for param in param_functions.keys()}
        for entry in customer_dialogues:
            start_sample = int(entry['startTime'] * sr)
            end_sample = int(entry['endTime'] * sr)
            audio_segment = audio[start_sample:end_sample]
            dialogue = entry['dialogue']
            start_time = entry['startTime']
            end_time = entry['endTime']
            
            # Calculate all parameters
            for param, func in param_functions.items():
                if param in ["Speaking rate", "Articulation rate", "Syllable rate"]:
                    customer_metrics[param].append(func(audio_segment, sr, dialogue, start_time, end_time))
                else:
                    customer_metrics[param].append(func(audio_segment, sr, dialogue))
        
        # Aggregate parameters
        for param in param_functions.keys():
            result[param.lower().replace(" ", "_")][1][param.lower().replace(" ", "_")] = round(sum(customer_metrics[param]) / len(customer_metrics[param]), 2)
    
    return result

from transformers import pipeline, AutoTokenizer, logging as hf_logging
import re
import textwrap

# Suppress transformers logging to keep console tidy
hf_logging.set_verbosity_error()

def generate_call_summary(transcription_data):
    """Generate a collections-focused call summary using hierarchical summarization with automatic speaker labeling."""
    # Model paths (point to the correct snapshots subdirectory)
    SAMSUM_MODEL = "/mnt/ssd1/temp_call_insights/Call_insights_Models/models--/models--philschmid--bart-large-cnn-samsum/snapshots/e49b3d60d923f12db22bdd363356f1a4c68532ad"
    LED_MODEL = "/mnt/ssd1/temp_call_insights/Call_insights_Models/models--/models--allenai--led-base-16384/snapshots/38335783885b338d93791936c54bb4be46bebed9"

    # Tunables
    SAMSUM_TOK_LIMIT = 900  # Stay safely below 1024
    SAMSUM_MAX_SUM = 120   # ≈ 100 words
    LED_MAX_SUM = 300      # Max length for LED meta-summary
    CHUNK_STRIDE = 100     # Token overlap between chunks

    # Regex for Promise-to-Pay
    PTP_PAT_AMOUNT = re.compile(r"\b(?:PHP|₱)?\s?([0-9]{1,3}(?:[,0-9]{3})*(?:\.\d{1,2})?)")
    PTP_PAT_DATE = re.compile(r"\b(?:by|on|before|until)\s+(?:\d{1,2}\s*\w+|\w+\s*\d{1,2})(?:\s*\d{4})?", re.I)
    PTP_PAT_INTENT = re.compile(r"\b(promise|settle|pay|bayaran|payment)\b", re.I)

    # Initialize tokenizer and pipelines
    tok_samsum = AutoTokenizer.from_pretrained(SAMSUM_MODEL)
    summ_samsum = pipeline(
        "summarization",
        model=SAMSUM_MODEL,
        tokenizer=SAMSUM_MODEL,
    )
    summ_led = pipeline(
        "summarization",
        model=LED_MODEL,
        tokenizer=LED_MODEL,
    )

    def identify_speaker_roles(turns):
        """Identify which speaker is the agent vs customer based on dialogue patterns."""
        # Comprehensive agent identification patterns
        AGENT_PATTERNS = [
            # Bank/Company identification
            r"\b(sally from dbs|from\s+(?:dbs|bank|company)|representative|advisor)\b",
            # Professional greetings
            r"\b(good\s+(?:morning|afternoon|evening),?\s+(?:miss|mr|mrs|sir))\b",
            r"\b(how\s+(?:can|may)\s+i\s+(?:help|assist)\s+you)\b",
            r"\b(what\s+can\s+i\s+do\s+for\s+you)\b",
            # Service-oriented language
            r"\b(i\s+(?:check|checked|will\s+check)\s+for\s+you)\b",
            r"\b(let\s+me\s+(?:check|verify|confirm))\b",
            r"\b(the\s+rate\s+is|contract\s+reference\s+number)\b",
            r"\b(i\s+put\s+the\s+order|order\s+(?:ready|processed))\b",
            r"\b(?:anything\s+else\s+i\s+can|is\s+there\s+anything\s+else)\b",
            # Professional closings
            r"\b(no\s+problem|you\s+can\s+call\s+me\s+again)\b",
            # Transaction language
            r"\b(so\s+\d+\s+(?:STD|SGD|USD)|done\.|correct\.?)\b"
        ]
        
        # Customer identification patterns
        CUSTOMER_PATTERNS = [
            # Request language
            r"\b(i\s+want\s+to\s+buy|what\s+is\s+the\s+rate|can\s+you\s+buy\s+for\s+me)\b",
            r"\b(buy\s+for\s+me|my\s+account\s+number\s+is)\b",
            # Clarification requests
            r"\b(what\'?s\s+the\s+contract\s+reference\s+number)\b",
            # Personal pronouns in request context
            r"\b(i\s+think\s+ok|that\'?s\s+it)\b"
        ]

        # Compile patterns
        agent_regex = re.compile("|".join(AGENT_PATTERNS), re.I)
        customer_regex = re.compile("|".join(CUSTOMER_PATTERNS), re.I)
        
        # Collect all dialogue by speaker
        speaker_dialogues = {}
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            dialogue = turn.get("dialogue", "") or turn.get("early_dialogue", "")
            if speaker not in speaker_dialogues:
                speaker_dialogues[speaker] = []
            speaker_dialogues[speaker].append(dialogue.lower().strip())
        
        # Score each speaker
        speaker_roles = {}
        for speaker, dialogues in speaker_dialogues.items():
            combined_text = " ".join(dialogues)
            agent_score = len(agent_regex.findall(combined_text))
            customer_score = len(customer_regex.findall(combined_text))
            
            # Assign role based on score
            if agent_score > customer_score:
                speaker_roles[speaker] = "Agent"
            elif customer_score > agent_score:
                speaker_roles[speaker] = "Customer"
            else:
                # Fallback: typically spk1 is agent in banking contexts
                speaker_roles[speaker] = "Agent" if speaker == "spk1" else "Customer"
                
        return speaker_roles

    def extract_dialogue(turns):
        """Join all dialogue with appropriate speaker labels (Agent/Customer)."""
        speaker_roles = identify_speaker_roles(turns)
        
        lines = []
        for turn in turns:
            txt = turn.get("dialogue") or turn.get("early_dialogue") or ""
            if txt.strip():
                speaker = turn.get("speaker", "Spk")
                role = speaker_roles.get(speaker, "Customer")
                lines.append(f"{role}: {txt.strip()}")
        return "\n".join(lines)

    def chunk_tokens(text, max_len, stride):
        """Return text chunks ≤ max_len tokens with overlap."""
        ids = tok_samsum(text)["input_ids"]
        if len(ids) <= max_len:
            return [text]
        chunks = []
        step = max_len - stride
        for i in range(0, len(ids), step):
            chunk_ids = ids[i:i + max_len]
            chunks.append(tok_samsum.decode(chunk_ids, skip_special_tokens=True))
            if i + max_len >= len(ids):
                break
        return chunks

    def summarise_chunk(txt):
        """Summarize a single chunk using SAMSUM model."""
        return summ_samsum(
            txt, min_length=40, max_length=SAMSUM_MAX_SUM, do_sample=False
        )[0]["summary_text"]

    def hierarchical_summary(full_txt):
        """Samsum on chunks → LED meta-summary if needed."""
        chunks = chunk_tokens(full_txt, SAMSUM_TOK_LIMIT, CHUNK_STRIDE)
        if len(chunks) == 1:
            return summarise_chunk(chunks[0])

        partials = [summarise_chunk(c) for c in chunks]
        combined = " ".join(partials)
        meta = summ_led(
            combined,
            min_length=80,
            max_length=LED_MAX_SUM,
            no_repeat_ngram_size=3,
            num_beams=4,
        )[0]["summary_text"]
        return meta

    def extract_ptp(txt):
        """Return a Promise-to-Pay sentence if amount+date+intent appear."""
        for line in txt.splitlines():
            if (PTP_PAT_INTENT.search(line) and
                PTP_PAT_AMOUNT.search(line) and
                PTP_PAT_DATE.search(line)):
                amt = PTP_PAT_AMOUNT.search(line).group(1)
                date = PTP_PAT_DATE.search(line).group(0)
                return f"Customer promises to pay PHP {amt} {date.strip()}."
        return None

    # Extract dialogue from transcription with role labels
    dialogue_text = extract_dialogue(transcription_data)
    if not dialogue_text:
        dialogue_text = "No dialogue available for summarization."

    # Generate summary
    summary = hierarchical_summary(dialogue_text)
    ptp = extract_ptp(dialogue_text)
    if ptp:
        summary = textwrap.dedent(f"""
            {summary}

            ---
            **TL;DR / Promise-to-Pay**
            • {ptp}
        """).strip()

    return summary

# Fuzzy logic evaluation function
def evaluate_fuzzy_match(dialogue: str, keywords: List[str], threshold: float = 0.5) -> bool:
    dialogue_lower = dialogue.lower()
    keyword_count = sum(1 for keyword in keywords if keyword.lower() in dialogue_lower)
    return keyword_count / len(keywords) >= threshold

# Function to calculate days overdue
def calculate_days_overdue(call_date_str: str, due_date_str: str) -> int:
    due_date_str = due_date_str.replace("th.", "").replace(".", "").strip()
    due_date_str = re.match(r"([A-Za-z]+ \d+)", due_date_str).group(1) if re.match(r"([A-Za-z]+ \d+)", due_date_str) else due_date_str
    due_date_str = re.sub(r'\s+Have\s+you\s+received\s+your\s+SOA\?', '', due_date_str).strip()
    call_date = datetime.strptime(call_date_str, "%d/%m/%Y")
    due_date = datetime.strptime(due_date_str, "%B %d")
    due_date = due_date.replace(year=call_date.year)
    if due_date < call_date.replace(month=1, day=1):
        due_date = due_date.replace(year=call_date.year + 1)
    days_overdue = (call_date - due_date).days
    return days_overdue

def calculate_intent(qa_results: Dict[str, Any], speech: List[Dict[str, Any]], audio_metrics: Dict[str, Any]) -> float:
    # QA-based intent score (existing logic)
    qa_intent_score = 0.0
    if qa_results["did_the_agent_state_the_product_name_current_balance_and_due_date"]:
        qa_intent_score += 0.4  # Mentioning loan details indicates collections intent
    if qa_results["did_the_agent_use_effective_probing_questions"]:
        qa_intent_score += 0.3  # Probing for payment assistance
    if qa_results["call_recap"]:
        qa_intent_score += 0.2  # Recapping payment details
    if qa_results["type_of_collection"] == "Predues Collection":
        qa_intent_score += 0.1  # Predues calls are proactive reminders
    
    # Voice-based intent score
    voice_intent_score = 0.0
    # Get customer energy and pitch variance
    customer_energy = next((item["energy"] for item in audio_metrics["energy"] if item["speaker"] == "spk2"), 0.0)
    customer_pitch = next((item["pitch_variation"] for item in audio_metrics["pitch_variation"] if item["speaker"] == "spk2"), 0.0)
    # Normalize energy and pitch to [0, 1]
    energy_score = customer_energy / 2.0 if customer_energy <= 2.0 else 1.0
    pitch_score = customer_pitch / 1.0 if customer_pitch <= 1.0 else 1.0  # Updated to match [0, 1] scale
    
    # Analyze customer emotions from speech
    customer_emotions = [entry["emotion"] for entry in speech if entry["speaker"] == "spk2"]
    positive_emotions = [
        "Hopefulness", "Contentment", "Optimism", "Satisfaction", "Reassurance", 
        "Relief", "Gratitude", "Empathy", "Acceptance", "Curiosity"
    ]
    negative_emotions = [
        "Anger", "Hostility", "Desperation", "Anxiety", "Stress", "Frustration", 
        "Defensiveness", "Impatience", "Distrust", "Shame", "Hopelessness", 
        "Pessimism", "Disappointment", "Regret", "Confusion", "Resignation", 
        "Apathy", "Indifference"
    ]
    emotion_score = 0.0
    if customer_emotions:
        positive_count = sum(1 for emotion in customer_emotions if emotion in positive_emotions)
        negative_count = sum(1 for emotion in customer_emotions if emotion in negative_emotions)
        total_emotions = len(customer_emotions)
        # Higher intent for positive emotions (indicating willingness to engage)
        emotion_score = (positive_count - negative_count) / total_emotions if total_emotions > 0 else 0.0
        emotion_score = (emotion_score + 1) / 2  # Normalize to [0, 1]
    
    # Combine vocal features (equal weights for simplicity)
    voice_intent_score = (energy_score + pitch_score + emotion_score) / 3.0
    
    # Combine QA and voice scores (70% QA, 30% voice to prioritize existing logic)
    final_intent_score = (0.7 * qa_intent_score) + (0.3 * voice_intent_score)
    return min(final_intent_score, 1.0)

# Criteria evaluation function
def evaluate_criteria(transcription: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {
        "id": "2194ae10-33b4-4e8b-9442-e7d77c493ceb",
        "call_date": "06/04/2025",
        "audit_date": "06/04/2025",
        "client": "Maya Bank Collections",
        "customer_name": None,
        "product": "Loan Default",
        "language": "English",
        "agent": None,
        "team_lead": "Alok",
        "qa_lead": "Alok",
        "min_number": None,
        "min_details": [],
        "call_open_timely_manner": False,
        "call_open_timely_manner_details": [],
        "standard_opening_spiel": False,
        "standard_opening_spiel_details": [],
        "did_the_agent_state_the_product_name_current_balance_and_due_date": False,
        "did_the_agent_state_the_product_name_current_balance_and_due_date_details": [],
        "call_opening_points": "0",
        "friendly_confident_tone": False,
        "friendly_confident_tone_details": [],
        "attentive_listening": False,
        "attentive_listening_details": [],
        "customer_experience_points": "0",
        "did_the_agent_use_effective_probing_questions": False,
        "did_the_agent_use_effective_probing_questions_details": [],
        "did_the_agent_act_towards_payment_resolution": False,
        "did_the_agent_act_towards_payment_resolution_details": [],
        "did_the_agent_provide_the_consequence_of_not_paying": False,
        "did_the_agent_provide_the_consequence_of_not_paying_details": [],
        "negotiation_points": "0",
        "follow_policies_procedure": False,
        "follow_policies_procedure_details": [],
        "process_compliance_points": "0",
        "call_document": False,
        "call_document_details": [],
        "documentation_points": "0",
        "call_recap": False,
        "call_recap_details": [],
        "additional_queries": False,
        "additional_queries_details": [],
        "thank_customer": False,
        "thank_customer_details": [],
        "call_closing": False,
        "call_closing_details": [],
        "call_closing_points": "0",
        "call_record_clause": False,
        "call_record_clause_details": [],
        "pid_process": False,
        "pid_process_details": [],
        "udcp_process": False,
        "udcp_process_details": [],
        "call_avoidance": False,
        "call_avoidance_details": [],
        "misleading_information": False,
        "misleading_information_details": [],
        "data_manipulation": False,
        "data_manipulation_details": [],
        "service_compliance_points": "0",
        "probing_questions_effectiveness": False,
        "probing_questions_effectiveness_details": [],
        "payment_resolution_actions": False,
        "payment_resolution_actions_details": [],
        "payment_delay_consequences": False,
        "payment_delay_consequences_details": [],
        "payment_promptness": False,
        "payment_promptness_details": [],
        "customer_verification_accuracy": False,
        "customer_verification_accuracy_details": [],
        "total_points": "0",
        "type_of_collection": None
    }

    due_date_str = None
    for entry in transcription:
        if "due on" in entry["dialogue"]:
            due_date_str = entry["dialogue"].split("due on")[1].split(",")[0].strip()
            break

    if due_date_str:
        days_overdue = calculate_days_overdue(output["call_date"], due_date_str)
        if days_overdue < 0:
            output["type_of_collection"] = "Predues Collection"
        elif days_overdue < 30:
            output["type_of_collection"] = "Postdue Collections Less Than 30 days"
        elif 30 <= days_overdue < 60:
            output["type_of_collection"] = "Postdue Collections Greater Than 30 days"
        else:
            output["type_of_collection"] = "Late Collections Greater Than 60 days"
    else:
        output["type_of_collection"] = "Predues Collection"

    spc_violation = False
    criteria_met = {
        "call_open_timely_manner": False,
        "standard_opening_spiel": False,
        "did_the_agent_state_the_product_name_current_balance_and_due_date": False,
        "friendly_confident_tone": False,
        "attentive_listening": False,
        "did_the_agent_use_effective_probing_questions": False,
        "did_the_agent_act_towards_payment_resolution": False,
        "did_the_agent_provide_the_consequence_of_not_paying": False,
        "follow_policies_procedure": False,
        "call_document": False,
        "call_recap": False,
        "additional_queries": False,
        "thank_customer": False,
        "call_closing": False
    }

    # First pass: Extract customer details
    for entry in transcription:
        dialogue = entry["dialogue"]
        speaker = entry["speaker"]
        start_time = entry["startTime"]
        end_time = entry["endTime"]

        if speaker == "spk2" and output["customer_name"] is None and ("I'm" in dialogue or "I am" in dialogue):
            name_match = re.search(r"I'm\s+([\w\s]+)|I am\s+([\w\s]+)", dialogue)
            if name_match:
                output["customer_name"] = name_match.group(1) if name_match.group(1) else name_match.group(2)
                output["customer_name"] = output["customer_name"].split("and")[0].strip()
        elif speaker == "spk1" and output["customer_name"] is None and "Miss" in dialogue:
            name_match = re.search(r"Miss\s+([\w\s]+)", dialogue)
            if name_match:
                output["customer_name"] = name_match.group(1).strip()

        if speaker == "spk2" and output["min_number"] is None and "number is" in dialogue:
            min_match = re.search(r"number is\s+(\d+)", dialogue)
            if min_match:
                output["min_number"] = min_match.group(1).strip().replace(".", "")
        elif speaker == "spk1" and output["min_number"] is None and "digits" in dialogue:
            min_match = re.search(r"(\d{4})", dialogue)
            if min_match:
                output["min_number"] = min_match.group(1)

        if speaker == "spk1" and "this is" in dialogue and output["agent"] is None:
            output["agent"] = dialogue.split("this is")[1].split("calling")[0].strip()

    # Second pass: Evaluate criteria for Predues Collection
    if output["type_of_collection"] == "Predues Collection":
        for entry in transcription:
            dialogue = entry["dialogue"]
            speaker = entry["speaker"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]

            if not criteria_met["call_open_timely_manner"] and start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner"] = True
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["call_open_timely_manner"] = True
            elif start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })

            if not criteria_met["standard_opening_spiel"] and evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel"] = True
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from Maya Bank"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 10)
                criteria_met["standard_opening_spiel"] = True
            elif evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from Maya Bank"
                })

            if not criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] and evaluate_fuzzy_match(dialogue, ["due on", "balance", "loan"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product name, balance, and duel date", "matched_phrase": "loan balance due on April 20th"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
            elif evaluate_fuzzy_match(dialogue, ["due on", "balance", "loan"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product name, balance, and due date", "matched_phrase": "loan balance due on April 20th"
                })

            if not criteria_met["friendly_confident_tone"] and evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone"] = True
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 10)
                criteria_met["friendly_confident_tone"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })

            if not criteria_met["attentive_listening"] and evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening"] = True
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 10)
                criteria_met["attentive_listening"] = True
            elif evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })

            if not criteria_met["did_the_agent_use_effective_probing_questions"] and evaluate_fuzzy_match(dialogue, ["would you like", "receive"]):
                output["did_the_agent_use_effective_probing_questions"] = True
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_use_effective_probing_questions"] = True
                output["probing_questions_effectiveness"] = True
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })
            elif evaluate_fuzzy_match(dialogue, ["would you like", "receive"]):
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to receive your SOA"
                })

            if not criteria_met["follow_policies_procedure"] and evaluate_fuzzy_match(dialogue, ["benefits", "consequences"]):
                output["follow_policies_procedure"] = True
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policies and procedures", "matched_phrase": "benefits of paying on time"
                })
                output["process_compliance_points"] = str(float(output["process_compliance_points"]) + 15)
                criteria_met["follow_policies_procedure"] = True
            elif evaluate_fuzzy_match(dialogue, ["benefits", "consequences"]):
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policies and procedures", "matched_phrase": "benefits of paying on time"
                })

            if not criteria_met["call_document"] and evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document"] = True
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })
                output["documentation_points"] = str(float(output["documentation_points"]) + 10)
                criteria_met["call_document"] = True
            elif evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })

            if not criteria_met["call_recap"] and evaluate_fuzzy_match(dialogue, ["payment", "due date"]):
                output["call_recap"] = True
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "payment due on April 20th"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_recap"] = True
            elif evaluate_fuzzy_match(dialogue, ["payment", "due date"]):
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "payment due on April 20th"
                })

            if not criteria_met["additional_queries"] and evaluate_fuzzy_match(dialogue, ["anything else", "questions"]):
                output["additional_queries"] = True
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional questions", "matched_phrase": "Anything else I can assist with"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["additional_queries"] = True
            elif evaluate_fuzzy_match(dialogue, ["anything else", "questions"]):
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional questions", "matched_phrase": "Anything else I can assist with"
                })

            if not criteria_met["thank_customer"] and evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer"] = True
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you and have a great day"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["thank_customer"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you and have a great day"
                })

            if not criteria_met["call_closing"] and evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing"] = True
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_closing"] = True
            elif evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })

            if evaluate_fuzzy_match(dialogue, ["recorded", "quality"]):
                output["call_record_clause"] = True
                output["call_record_clause_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call record clause mentioned", "matched_phrase": "This call is recorded"
                })

            if evaluate_fuzzy_match(dialogue, ["full name", "mobile number"]):
                output["pid_process"] = True
                output["pid_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "PID process followed", "matched_phrase": "full name and registered mobile number"
                })

            if not evaluate_fuzzy_match(dialogue, ["swear", "offensive"]):
                output["udcp_process"] = True
                output["udcp_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Adhered to UDCP Prohibition", "matched_phrase": "No offensive language"
                })

            if evaluate_fuzzy_match(dialogue, ["avoid", "hang up"]):
                output["call_avoidance"] = True
                output["call_avoidance_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call avoidance detected", "matched_phrase": "avoid"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["mislead", "false"]):
                output["misleading_information"] = True
                output["misleading_information_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Misleading information detected", "matched_phrase": "mislead"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["manipulate", "alter"]):
                output["data_manipulation"] = True
                output["data_manipulation_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Data manipulation detected", "matched_phrase": "manipulate"
                })
                spc_violation = True

            if output["customer_name"] and output["min_number"] and speaker == "spk2" and output["customer_name"] in dialogue and output["min_number"] in dialogue:
                output["customer_verification_accuracy"] = True
                output["customer_verification_accuracy_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Accurate verification provided", "matched_phrase": f"{output['customer_name']} {output['min_number']}"
                })

    # Second pass: Evaluate criteria for Postdues Collection
    elif output["type_of_collection"] in ["Postdue Collections Less Than 30 days", "Postdue Collections Greater Than 30 days", "Late Collections Greater Than 60 days"]:
        for entry in transcription:
            dialogue = entry["dialogue"]
            speaker = entry["speaker"]
            start_time = entry["startTime"]
            end_time = entry["endTime"]

            if not criteria_met["call_open_timely_manner"] and start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner"] = True
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 3)
                criteria_met["call_open_timely_manner"] = True
            elif start_time <= 5 and evaluate_fuzzy_match(dialogue, ["good afternoon", "calling"]):
                output["call_open_timely_manner_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call opened within 5 seconds", "matched_phrase": "good afternoon"
                })

            if not criteria_met["standard_opening_spiel"] and evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel"] = True
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from FinTrust Services"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["standard_opening_spiel"] = True
            elif evaluate_fuzzy_match(dialogue, ["calling from", "recorded"]):
                output["standard_opening_spiel_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used standard opening spiel", "matched_phrase": "calling from FinTrust Services"
                })

            if not criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] and evaluate_fuzzy_match(dialogue, ["overdue payment", "credit card", "due on"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product, balance, and due date",
                    "matched_phrase": "overdue payment of ₹3,200 on your credit card ending with 7890, which was due on April 5th"
                })
                output["call_opening_points"] = str(float(output["call_opening_points"]) + 5)
                criteria_met["did_the_agent_state_the_product_name_current_balance_and_due_date"] = True
            elif evaluate_fuzzy_match(dialogue, ["overdue payment", "credit card", "due on"]):
                output["did_the_agent_state_the_product_name_current_balance_and_due_date_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Stated product, balance, and due date",
                    "matched_phrase": "overdue payment of ₹3,200 on your credit card ending with 7890, which was due on April 5th"
                })

            if not criteria_met["friendly_confident_tone"] and evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone"] = True
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 7)
                criteria_met["friendly_confident_tone"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "please"]):
                output["friendly_confident_tone_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Maintained friendly and confident tone", "matched_phrase": "thank you"
                })

            if not criteria_met["attentive_listening"] and evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening"] = True
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })
                output["customer_experience_points"] = str(float(output["customer_experience_points"]) + 7)
                criteria_met["attentive_listening"] = True
            elif evaluate_fuzzy_match(dialogue, ["of course", "clarification"]):
                output["attentive_listening_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Listened attentively and displayed understanding", "matched_phrase": "of course"
                })

            if not criteria_met["did_the_agent_use_effective_probing_questions"] and evaluate_fuzzy_match(dialogue, ["would you like", "anything else"]):
                output["did_the_agent_use_effective_probing_questions"] = True
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_use_effective_probing_questions"] = True
                output["probing_questions_effectiveness"] = True
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })
            elif evaluate_fuzzy_match(dialogue, ["would you like", "anything else"]):
                output["did_the_agent_use_effective_probing_questions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })
                output["probing_questions_effectiveness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Used effective probing questions", "matched_phrase": "Would you like to make the payment today"
                })

            if not criteria_met["did_the_agent_act_towards_payment_resolution"] and evaluate_fuzzy_match(dialogue, ["payment", "today", "evening"]):
                output["did_the_agent_act_towards_payment_resolution"] = True
                output["did_the_agent_act_towards_payment_resolution_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Acted towards payment resolution", "matched_phrase": "pay it online by this evening"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 20)
                criteria_met["did_the_agent_act_towards_payment_resolution"] = True
                output["payment_resolution_actions"] = True
                output["payment_resolution_actions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Payment resolution agreed", "matched_phrase": "pay it online by this evening"
                })
                output["payment_promptness"] = True
                output["payment_promptness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Customer committed to timely payment", "matched_phrase": "pay it online by this evening"
                })
            elif evaluate_fuzzy_match(dialogue, ["payment", "today", "evening"]):
                output["did_the_agent_act_towards_payment_resolution_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Acted towards payment resolution", "matched_phrase": "pay it online by this evening"
                })
                output["payment_resolution_actions_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Payment resolution agreed", "matched_phrase": "pay it online by this evening"
                })
                output["payment_promptness_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Customer committed to timely payment", "matched_phrase": "pay it online by this evening"
                })

            if not criteria_met["did_the_agent_provide_the_consequence_of_not_paying"] and evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["did_the_agent_provide_the_consequence_of_not_paying"] = True
                output["did_the_agent_provide_the_consequence_of_not_paying_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided consequence of not paying", "matched_phrase": "late fee will not increase further"
                })
                output["negotiation_points"] = str(float(output["negotiation_points"]) + 10)
                criteria_met["did_the_agent_provide_the_consequence_of_not_paying"] = True
                output["payment_delay_consequences"] = True
                output["payment_delay_consequences_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Explained delay consequence", "matched_phrase": "late fee will not increase further"
                })
            elif evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["did_the_agent_provide_the_consequence_of_not_paying_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided consequence of not paying", "matched_phrase": "late fee will not increase further"
                })
                output["payment_delay_consequences_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Explained delay consequence", "matched_phrase": "late fee will not increase further"
                })

            if not criteria_met["follow_policies_procedure"] and evaluate_fuzzy_match(dialogue, ["due amount", "principal", "late fees"]):
                output["follow_policies_procedure"] = True
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policy and procedure", "matched_phrase": "₹3,000 as the principal and ₹200 as late fees"
                })
                output["process_compliance_points"] = str(float(output["process_compliance_points"]) + 10)
                criteria_met["follow_policies_procedure"] = True
            elif evaluate_fuzzy_match(dialogue, ["due amount", "principal", "late fees"]):
                output["follow_policies_procedure_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Followed policy and procedure", "matched_phrase": "₹3,000 as the principal and ₹200 as late fees"
                })

            if not criteria_met["call_document"] and evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document"] = True
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })
                output["documentation_points"] = str(float(output["documentation_points"]) + 5)
                criteria_met["call_document"] = True
            elif evaluate_fuzzy_match(dialogue, ["document", "tools"]):
                output["call_document_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Properly documented the call", "matched_phrase": "documented in tools"
                })

            if not criteria_met["call_recap"] and evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["call_recap"] = True
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "late fee will not increase further"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_recap"] = True
            elif evaluate_fuzzy_match(dialogue, ["late fee", "increase"]):
                output["call_recap_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Provided call recap", "matched_phrase": "late fee will not increase further"
                })

            if not criteria_met["additional_queries"] and evaluate_fuzzy_match(dialogue, ["anything else", "help"]):
                output["additional_queries"] = True
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional queries", "matched_phrase": "Is there anything else I can help you with"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 3)
                criteria_met["additional_queries"] = True
            elif evaluate_fuzzy_match(dialogue, ["anything else", "help"]):
                output["additional_queries_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Asked if there are additional queries", "matched_phrase": "Is there anything else I can help you with"
                })

            if not criteria_met["thank_customer"] and evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer"] = True
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you for your time"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 2)
                criteria_met["thank_customer"] = True
            elif evaluate_fuzzy_match(dialogue, ["thank you", "great day"]):
                output["thank_customer_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Thanked the customer", "matched_phrase": "Thank you for your time"
                })

            if not criteria_met["call_closing"] and evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing"] = True
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })
                output["call_closing_points"] = str(float(output["call_closing_points"]) + 5)
                criteria_met["call_closing"] = True
            elif evaluate_fuzzy_match(dialogue, ["goodbye", "day ahead"]):
                output["call_closing_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call closed professionally", "matched_phrase": "goodbye"
                })

            if evaluate_fuzzy_match(dialogue, ["recorded", "quality"]):
                output["call_record_clause"] = True
                output["call_record_clause_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call record clause mentioned", "matched_phrase": "This call is recorded"
                })

            if evaluate_fuzzy_match(dialogue, ["full name", "mobile number"]):
                output["pid_process"] = True
                output["pid_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "PID process followed", "matched_phrase": "full name and registered mobile number"
                })

            if not evaluate_fuzzy_match(dialogue, ["swear", "offensive"]):
                output["udcp_process"] = True
                output["udcp_process_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Adhered to UDCP Prohibition", "matched_phrase": "No offensive language"
                })

            if evaluate_fuzzy_match(dialogue, ["avoid", "hang up"]):
                output["call_avoidance"] = True
                output["call_avoidance_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Call avoidance detected", "matched_phrase": "avoid"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["mislead", "false"]):
                output["misleading_information"] = True
                output["misleading_information_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Misleading information detected", "matched_phrase": "mislead"
                })
                spc_violation = True

            if evaluate_fuzzy_match(dialogue, ["manipulate", "alter"]):
                output["data_manipulation"] = True
                output["data_manipulation_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Data manipulation detected", "matched_phrase": "manipulate"
                })
                spc_violation = True

            if output["customer_name"] and output["min_number"] and speaker == "spk2" and output["customer_name"] in dialogue and output["min_number"] in dialogue:
                output["customer_verification_accuracy"] = True
                output["customer_verification_accuracy_details"].append({
                    "dialogue": dialogue, "speaker": speaker, "startTime": start_time, "endTime": end_time,
                    "reason": "Accurate verification provided", "matched_phrase": f"{output['customer_name']} {output['min_number']}"
                })

    if spc_violation or not (output["call_record_clause"] and output["pid_process"] and output["udcp_process"]):
        output["total_points"] = "0"
    else:
        total_points = (float(output["call_opening_points"]) +
                        float(output["customer_experience_points"]) +
                        float(output["negotiation_points"]) +
                        float(output["process_compliance_points"]) +
                        float(output["documentation_points"]) +
                        float(output["call_closing_points"]) +
                        float(output["service_compliance_points"]))
        output["total_points"] = str(total_points)

    return output

# Main processing function
def process_transcription(transcription, audio_file):
    output = {
        "id": "2194ae10-33b4-4e8b-9442-e7d77c493ceb",
        "call_date": "06/04/2025",
        "audit_date": "06/04/2025",
        "client": "Maya Bank Collections",
        "customer_name": None,
        "product": "Loan Default",
        "language": "English",
        "agent": None,
        "team_lead": "Alok",
        "qa_lead": "Alok",
        "min_number": None,
        "min_details": [],
        "speech": [],
        "audio_file": "",
        "cx_score": 0.0,
        "intent": 0.0,
        "problem": False,
        "resolution": False,
        "customer_sentiment": 0.0,
        "agent_sentiment": 0.0,
        "energy": [],
        "entropy": [],
        "loudness": [],
        "pitch_variation": [],
        "tone_result": {"agreeableness": 50, "disagreeableness": 50},
        "keywords": [],
        "sentiment_keywords": [],
        "call_type": "Collection",
        "category": "Loan",
        "duration": "",
        "agent_collection_view": "Promise to Pay",
        "dataklout_collection_view": "Promise to Pay",
        "agent_id": "",
        "ai_summary": {"summary": ""},
        "collection_params": {
            "dk_collection_status": "Promise to Pay",
            "progress_bar": 50,
            "payment_pct": False,
            "debtor_status": "Promise to Pay",
            "collection_score_card": 50.0
        }
    }
    
    # Process speech with emotions
    output["speech"] = analyze_emotions(transcription, audio_file)
    
    # Calculate audio metrics
    audio_metrics = calculate_audio_metrics(transcription, audio_file)
    for param in audio_metrics:
        output[param] = audio_metrics[param]
    
    # Generate summary
    output["ai_summary"]["summary"] = generate_call_summary(transcription)
    
    # Evaluate quality assurance criteria
    qa_results = evaluate_criteria(transcription)
    output.update(qa_results)
    
    # Calculate intent score
    output["intent"] = calculate_intent(qa_results, output["speech"], audio_metrics)
    
    # Calculate duration
    total_seconds = transcription[-1]["endTime"]
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    output["duration"] = f"0:{minutes:02d}:{seconds:02d}"
    
    # Extract keywords
    output["keywords"] = ["loan", "payment", "due"]
    output["sentiment_keywords"] = ["thank you", "good"]
    
    # Calculate sentiments
    agent_dialogues = [entry["dialogue"] for entry in transcription if entry["speaker"] == "spk1"]
    customer_dialogues = [entry["dialogue"] for entry in transcription if entry["speaker"] == "spk2"]
    output["agent_sentiment"] = sum(TextBlob(d).sentiment.polarity for d in agent_dialogues) / len(agent_dialogues) if agent_dialogues else 0.0
    output["customer_sentiment"] = sum(TextBlob(d).sentiment.polarity for d in customer_dialogues) / len(customer_dialogues) if customer_dialogues else 0.0
    
    return output

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import os
def main():
    try:
        with open("Relay the message Agnes Mallanao Joy Mendiguarin 611870047754 020125.json", "r") as file:
            transcription = json.load(file)
            audio_file = "Relay the message Agnes Mallanao Joy Mendiguarin 611870047754 020125.wav"
        result = process_transcription(transcription, audio_file)
        # Create output JSON file named after the audio file
        output_file = "output" + os.path.splitext(audio_file)[0] + ".json"
        with open(output_file, "w") as outfile:
            json.dump(result, outfile, indent=4, cls=NumpyEncoder)
        print(json.dumps(result, indent=4, cls=NumpyEncoder))
    except FileNotFoundError:
        print("Error: 'transcription.json' not found.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'transcription.json'.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
