import nltk
nltk.data.path.append("./nltk_data")
import os
import json
import re
import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from nltk.tokenize import word_tokenize
import textstat
import langid
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import stats

# Directories
AUDIO_DIR = "./audio"
TRANSCRIPTION_DIR = "./transcription"
OUTPUT_DIR = "./output"
PROMPTS_DIR = "./prompts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Local model paths
LOCAL_TEXT_MODEL_PATH = "./emotion_models/emotion-english-distilroberta-base"
LOCAL_AUDIO_MODEL_PATH = "./emotion_models/hubert-large-superb-er"

# Allowed Emotions
ALLOWED_EMOTIONS = [
    "anger", "frustration", "confusion", "stress", "anxiety", "resignation",
    "hopefulness", "distrust", "regret", "empathy", "defensiveness", "negotiation",
    "impatience", "contentment", "optimism", "desperation", "indifference", "pessimism",
    "curiosity", "satisfaction", "disappointment", "shame", "reassurance", "relief",
    "hopelessness", "gratitude", "hostility", "acceptance", "apathy"
]

# Emotion Mappings
TEXT_LABEL_MAPPING = {
    "anger": "anger", "joy": "curiosity", "optimism": "optimism", "sadness": "contentment",
    "fear": "anxiety", "love": "empathy", "surprise": "curiosity", "trust": "reassurance",
    "disgust": "hostility", "anticipation": "negotiation"
}

AUDIO_LABEL_MAPPING = {
    "angry": "anger", "happy": "contentment", "excited": "optimism", "sad": "contentment",
    "neutral": "indifference", "fearful": "anxiety", "disgust": "hostility", "surprised": "curiosity"
}

KEYWORD_PROMPT_MAP = {
    r"\b(idiot|stupid|shut up|jerk|moron|fool|dumb|incompetent|useless|pathetic|clueless|imbecile|loser|do your job|fix it now|this is nonsense|ridiculous service|are you deaf|sue you|take legal action|report you|get you fired|you’re an idiot|you are an idiot|you’re stupid|you are stupid|you’re incompetent|you are incompetent|you’re useless|you are useless|you’re pathetic|you are pathetic|you’re clueless|you are clueless|you’re a loser|you are a loser|you’re a jerk|you are a jerk|get it done|sort it out now|stop wasting time|what’s wrong with you|what is wrong with you|you’re not listening|you are not listening|i’ll sue|i will sue|legal action|reporting you|i’ll complain|i will complain|file a complaint|take this to court|this is absurd|what a joke|terrible service|you’re hopeless|you are hopeless|are you blind)\b": {
        "prompt_file": "CRITICAL/abuse/anger.json",
        "prompt": "Forward to manager immediately due to abusive behavior detected.",
        "priority": "CRITICAL"
    },
    r"\b(account is hacked|fraud|scammed|unauthorized|scammer|stolen|hack|hacked my account|fraudulent|identity theft|compromised|suspicious activity|phished|phishing|fake transaction|breached|stole my cards|stole my money|hacked my card|someone used my account|unrecognized charge|not my transaction|my account was hacked|someone hacked my account|fraud on my account|scam on my card|unauthorized transaction|stolen funds|account compromised|suspicious charge|identity stolen|card was hacked|not my purchase|fake charge|this is a scam|i got scammed|someone’s using my card|someone is using my card|my money’s gone|my money is gone|hack on my account)\b": {
        "prompt_file": "CRITICAL/fraud/anxiety.json",
        "prompt": "Escalate fraud report immediately and verify account details",
        "priority": "CRITICAL"
    },
    r"\b(lost my card|card is lost|misplaced my card|card’s missing|can’t find my card|i’ve lost my card|i have lost my card|my card’s gone|my card is gone|card is missing|lost the card|cannot find my card)\b": {
        "prompt_file": "CRITICAL/lost_card/anxiety.json",
        "prompt": "Escalate lost card report immediately and initiate card replacement process.",
        "priority": "CRITICAL"
    },
    r"\b(i am not paying|i am not gonna pay|i am not going to pay|why should i pay|bill wrong|wrongly charged|won’t pay|will not pay|refuse to pay|not gonna pay|i ain’t paying|i am done paying|i am not paying that|incorrect bill|overcharged|charged twice|double charged|billing mistake|wrong amount|dispute bill|give me a refund|need a refund|take it off|remove this charge|i’m not gonna pay this|i am not going to pay this|i refuse to pay|i won’t pay this|i will not pay this|no way i’m paying|no way i am paying|i’m not paying that|bill is incorrect|wrong billing|charged wrong|billed incorrectly|error on my bill|wrong charge on bill|i want my money back|refund me now|take off this charge|reverse this charge|cancel this payment|this bill is ridiculous|why am i being charged|charged for nothing|fix this bill)\b": {
        "prompt_file": "HIGH/payment/frustration.json",
        "prompt": "Address payment dispute promptly and transfer to billing if unresolved",
        "priority": "HIGH"
    },
    r"\b(balance is wrong|wrong balance|dispute charge|incorrect charge|wrong charge|account balance wrong|balance incorrect|wrong account balance|balance doesn’t match|balance does not match|balance off|charge is wrong|didn’t authorize|did not authorize|not my charge|unauthorized charge|discrepancy in balance|overdraft fee|wrong deduction|incorrect deduction|missing funds|my balance is incorrect|balance is off|account balance is wrong|balance not right|wrong balance on account|i didn’t make this charge|i did not make this charge|charge not mine|didn’t approve this|did not approve this|unauthorized deduction|wrong charge on account|missing money|funds missing|overdraft error|incorrect balance|balance discrepancy)\b": {
        "prompt_file": "HIGH/balance/frustration.json",
        "prompt": "Resolve balance dispute by reviewing account details.",
        "priority": "HIGH"
    },
    r"\b(sent to wrong number|wrong number transfer|money to wrong account|transferred to wrong number|it’s the wrong recipient|it is the wrong recipient|sent money to wrong person|wrong account transfer|i’ve sent to wrong number|i have sent to wrong number|money sent to wrong number)\b": {
        "prompt_file": "HIGH/wrong_transfer/frustration.json",
        "prompt": "Address erroneous transfer promptly and initiate recovery process.",
        "priority": "HIGH"
    },
    r"\b(app not working|app is not working|app isn’t working|app crash|transaction fail|error|bug|app keeps crashing|app won’t load|app will not load|app frozen|app error|can’t log in|cannot log in|login issue|payment failed|transfer failed|transaction error|couldn’t process payment|could not process payment|website down|system error|platform not working|technical issue|glitch|app stopped working|app doesn’t work|app does not work|can’t use the app|cannot use the app|app is down|login not working|payment didn’t go through|payment did not go through|transfer didn’t work|transfer did not work|failed transaction|couldn’t complete payment|could not complete payment|transaction not processed|site not working|website crashed|system is down|technical error|app glitching)\b": {
        "prompt_file": "MEDIUM/technical/confusion.json",
        "prompt": "Assist with technical issues and escalate if unresolved",
        "priority": "HIGH"
    }
}

# Load models
try:
    text_tokenizer = AutoTokenizer.from_pretrained(LOCAL_TEXT_MODEL_PATH)
    text_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_TEXT_MODEL_PATH)
    text_pipeline = pipeline(
        "text-classification", model=text_model, tokenizer=text_tokenizer,
        top_k=None, device=0 if torch.cuda.is_available() else -1
    )
    audio_pipeline = pipeline(
        "audio-classification", model=LOCAL_AUDIO_MODEL_PATH,
        top_k=None, device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

# Stage 0: Keyword-Based Prompt Selection
def stage_zero_prompt(dialogue):
    lowered = dialogue.lower().strip()
    matches = []
    for pattern, prompt_data in KEYWORD_PROMPT_MAP.items():
        if re.search(pattern, lowered):
            matches.append((prompt_data["priority"], prompt_data))
    if matches:
        matches.sort(key=lambda x: {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}[x[0]], reverse=True)
        return {
            "prompt": matches[0][1]["prompt"],
            "priority": matches[0][1]["priority"],
            "from_stage_zero": True
        }
    return None

# Stage 1: Emotion Analysis
def predict_text_emotion(text):
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

def predict_audio_emotion(audio_path):
    try:
        waveform, sr = torchaudio.load(audio_path)
        preds = audio_pipeline({"array": waveform[0].numpy(), "sampling_rate": sr})
        mapped = []
        for p in preds:
            lbl = p['label'].lower()
            if lbl in AUDIO_LABEL_MAPPING:
                mapped_lbl = AUDIO_LABEL_MAPPING[lbl]
                mapped.append({'label': mapped_lbl, 'score': p['score']})
        if not mapped:
            return "Unknown", 0.0
        best = max(mapped, key=lambda x: x['score'])
        return best['label'], round(best['score'], 3)
    except:
        return "Unknown", 0.0

def merge_emotions(text_emo, text_conf, audio_emo, audio_conf):
    if text_emo == "Unknown" and audio_emo == "Unknown":
        return "Unknown", 0.0
    if text_emo == audio_emo and text_emo != "Unknown":
        return text_emo, round((text_conf + audio_conf) / 2, 3)
    return (text_emo, text_conf) if text_conf >= audio_conf else (audio_emo, audio_conf)

def regex_based_emotion_override(text):
    lowered = text.lower().strip()
    if re.search(r"\b(thanks | thank you)\b", lowered):
        return "gratitude"
    if re.search(r"\b(whatever|i don't care | i do not care|as you wish)\b", lowered):
        return "apathy"
    if re.search(r"\b(don't worry|it's okay|no problem|you're fine)\b", lowered):
        return "reassurance"
    if re.search(r"\b(phew|thank goodness|i'm relieved)\b", lowered):
        return "relief"
    if re.search(r"\b(great job|well done|perfect|excellent|that works)\b", lowered):
        return "satisfaction"
    if re.search(r"\b(i guess|oh well|might as well|fine then)\b", lowered):
        return "resignation"
    if re.search(r"\b(hurry up|come on|anytime now|get to)\b", lowered):
        return "impatience"
    if re.search(r"\b(I'm Sorry | sorry|i apologize|my apologies)\b", lowered):
        return "regret"
    return None

# Stage 2: Audio Metrics
def calculate_energy(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    rms = librosa.feature.rms(y=audio_segment).mean()
    max_rms = 0.1
    return min(float(rms / max_rms * 2), 2.0)

def calculate_entropy(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    S_power = S ** 2
    S_power = S_power / (np.sum(S_power, axis=0) + 1e-10)
    entropy = -np.sum(S_power * np.log2(S_power + 1e-10), axis=0).mean()
    max_entropy = 10.0
    return min(float(entropy / max_entropy * 3), 3.0)

def calculate_loudness(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    audio_segment = audio_segment / (np.max(np.abs(audio_segment)) + 1e-10)
    rms = librosa.feature.rms(y=audio_segment, frame_length=2048, hop_length=512).mean()
    db = librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0]
    min_db, max_db = -80, 0
    loudness = (db - min_db) / (max_db - min_db)
    return min(max(float(loudness), 0.0), 1.0)

def calculate_pitch_variation(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
    pitches = pitches[magnitudes > 0]
    if len(pitches) == 0:
        return 0.5
    return min(float(np.std(pitches) / 200), 1.0)

def calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time):
    words = len(word_tokenize(dialogue))
    duration = end_time - start_time
    rate = words / duration if duration > 0 else 0
    max_rate = 6.0
    return min(float(rate / max_rate), 1.0)

def calculate_spectral_density(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))**2
    density = np.mean(S)
    max_density = max(density, 1e-10) * 1000
    return min(float(density / max_density), 1.0)

def calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time):
    syllables = textstat.syllable_count(dialogue)
    duration = end_time - start_time
    rate = syllables / duration if duration > 0 else 0
    max_rate = 8.0
    return min(float(rate / max_rate), 1.0)

def calculate_syllable_rate(audio_segment, sr, dialogue, start_time, end_time):
    syllables = textstat.syllable_count(dialogue)
    duration = librosa.get_duration(y=audio_segment, sr=sr)
    rate = syllables / duration if duration > 0 else 0
    max_rate = 8.0
    return min(float(rate / max_rate), 1.0)

def calculate_emotional_intensity(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    rms = librosa.feature.rms(y=audio_segment).mean()
    strong_words = ["urgent", "please", "immediately", "sorry", "thanks"]
    text_score = sum(1 for word in word_tokenize(dialogue.lower()) if word in strong_words) / 5
    audio_score = min(rms / 0.1, 1.0)
    return float(0.5 * text_score + 0.5 * audio_score)

def calculate_semantic_similarity(audio_segment, sr, dialogue):
    debt_terms = ["loan", "payment", "due", "balance", "debt"]
    tokens = word_tokenize(dialogue.lower())
    similarity = sum(1 for token in tokens if token in debt_terms) / len(debt_terms)
    return min(float(similarity), 1.0)

def calculate_language_identification(audio_segment, sr, dialogue):
    lang, confidence = langid.classify(dialogue)
    return float(min(max(confidence, 0.0), 1.0) if lang == "en" else 0.0)

def calculate_pronunciation_accuracy(audio_segment, sr, dialogue):
    if len(dialogue.strip()) == 0 or len(audio_segment) < 64:
        return 0.5
    return 0.5  # Placeholder

def calculate_emotional_valence(audio_segment, sr, dialogue):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(dialogue)
    compound = scores.get('compound', 0.0)
    sentiment = (compound + 1) / 2
    if len(dialogue.split()) <= 3 or abs(compound) < 0.1:
        sentiment = 0.5
    return float(max(min(sentiment, 1.0), 0.0))

def calculate_spectral_flux(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    flux = librosa.onset.onset_strength(y=audio_segment, sr=sr, S=S)
    return min(float(np.mean(flux) / 10), 1.0)

def calculate_tremor(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    pitches, _ = librosa.piptrack(y=audio_segment, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) == 0:
        return 0.5
    tremor = np.std(np.diff(pitches)) / 50
    return min(float(tremor), 1.0)

def calculate_jitter(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    pitches, _ = librosa.piptrack(y=audio_segment, sr=sr)
    pitches = pitches[pitches > 0]
    if len(pitches) < 2:
        return 0.5
    jitter = np.mean(np.abs(np.diff(pitches)) / pitches[:-1])
    return min(float(jitter / 0.05), 1.0)

def calculate_shimmer(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    rms = librosa.feature.rms(y=audio_segment)
    if len(rms) < 2:
        return 0.5
    shimmer = np.mean(np.abs(np.diff(rms)) / rms[:-1])
    return min(float(shimmer / 0.1), 1.0)

def calculate_spectral_rolloff(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr).mean()
    max_rolloff = sr / 2
    return min(float(rolloff / max_rolloff), 1.0)

def calculate_irony_detection(audio_segment, sr, dialogue):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(dialogue)['compound']
    blob = TextBlob(dialogue)
    subjectivity = blob.sentiment.subjectivity
    mismatch = abs(sentiment - subjectivity)
    return min(float(mismatch), 1.0)

def calculate_spectral_tilt(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_spectrum = np.mean(S, axis=1)
    min_len = min(len(freqs), len(mean_spectrum))
    freqs = freqs[:min_len]
    mean_spectrum = mean_spectrum[:min_len]
    if len(mean_spectrum) < 2:
        return 0.5
    tilt = np.polyfit(freqs, 10 * np.log10(mean_spectrum + 1e-10), 1)[0]
    return min(float(abs(tilt) / 0.01), 1.0)

def calculate_language_biases(audio_segment, sr, dialogue):
    blob = TextBlob(dialogue)
    polarity = blob.sentiment.polarity
    neutral_score = 0.0
    bias = abs(polarity - neutral_score)
    return min(float(bias), 1.0)

def calculate_zero_cross_rate(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    zcr = librosa.feature.zero_crossing_rate(y=audio_segment).mean()
    max_zcr = 0.5
    return min(float(zcr / max_zcr), 1.0)

def calculate_spectral_centroid(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr, n_fft=n_fft).mean()
    max_centroid = sr / 2
    return min(float(centroid / max_centroid), 1.0)

def calculate_spectral_flatness(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    flatness = librosa.feature.spectral_flatness(y=audio_segment).mean()
    return min(max(float(flatness), 0.0), 1.0)

def calculate_formant_frequencies(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peaks = freqs[np.argmax(S, axis=0)]
    avg_formant = np.mean(peaks) if len(peaks) > 0 else 500
    max_formant = 5000
    return min(float(avg_formant / max_formant), 1.0)

def calculate_spectral_kurtosis(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    if S.shape[1] < 2:
        return 0.5
    kurtosis = np.mean([stats.kurtosis(S[:, i]) for i in range(S.shape[1]) if np.any(S[:, i])])
    if np.isnan(kurtosis):
        return 0.5
    return min(float((kurtosis + 3) / 10), 1.0)

def calculate_tone(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    blob = TextBlob(dialogue)
    subjectivity = blob.sentiment.subjectivity
    n_fft = min(2048, len(audio_segment))
    chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr, n_fft=n_fft).mean()
    return min(float((subjectivity + chroma) / 2), 1.0)

def calculate_resonance(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    formants = librosa.effects.preemphasis(audio_segment)
    resonance = np.mean(np.abs(formants)) / (np.max(np.abs(formants)) + 1e-10)
    return min(max(float(resonance), 0.0), 1.0)

def calculate_spectral_skewness(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    n_fft = min(2048, len(audio_segment))
    S = np.abs(librosa.stft(audio_segment, n_fft=n_fft))
    if S.shape[1] < 2:
        return 0.5
    skewness = np.mean([stats.skew(S[:, i]) for i in range(S.shape[1]) if np.any(S[:, i])])
    if np.isnan(skewness):
        return 0.5
    return min(float((skewness + 3) / 6), 1.0)

# Stage 2 Wrapper
def compute_audio_metrics(audio_segment, sr, dialogue, start_time, end_time):
    metrics = {
        "energy": calculate_energy(audio_segment, sr, dialogue),
        "entropy": calculate_entropy(audio_segment, sr, dialogue),
        "loudness": calculate_loudness(audio_segment, sr, dialogue),
        "pitch_variation": calculate_pitch_variation(audio_segment, sr, dialogue),
        "speaking_rate": calculate_speaking_rate(audio_segment, sr, dialogue, start_time, end_time),
        "spectral_density": calculate_spectral_density(audio_segment, sr, dialogue),
        "articulation_rate": calculate_articulation_rate(audio_segment, sr, dialogue, start_time, end_time),
        "syllable_rate": calculate_syllable_rate(audio_segment, sr, dialogue, start_time, end_time),
        "emotional_intensity": calculate_emotional_intensity(audio_segment, sr, dialogue),
        "semantic_similarity": calculate_semantic_similarity(audio_segment, sr, dialogue),
        "language_identification": calculate_language_identification(audio_segment, sr, dialogue),
        "pronunciation_accuracy": calculate_pronunciation_accuracy(audio_segment, sr, dialogue),
        "emotional_valence": calculate_emotional_valence(audio_segment, sr, dialogue),
        "spectral_flux": calculate_spectral_flux(audio_segment, sr, dialogue),
        "tremor": calculate_tremor(audio_segment, sr, dialogue),
        "jitter": calculate_jitter(audio_segment, sr, dialogue),
        "shimmer": calculate_shimmer(audio_segment, sr, dialogue),
        "spectral_rolloff": calculate_spectral_rolloff(audio_segment, sr, dialogue),
        "irony_detection": calculate_irony_detection(audio_segment, sr, dialogue),
        "spectral_tilt": calculate_spectral_tilt(audio_segment, sr, dialogue),
        "language_biases": calculate_language_biases(audio_segment, sr, dialogue),
        "zero_cross_rate": calculate_zero_cross_rate(audio_segment, sr, dialogue),
        "spectral_centroid": calculate_spectral_centroid(audio_segment, sr, dialogue),
        "spectral_flatness": calculate_spectral_flatness(audio_segment, sr, dialogue),
        "formant_frequencies": calculate_formant_frequencies(audio_segment, sr, dialogue),
        "spectral_kurtosis": calculate_spectral_kurtosis(audio_segment, sr, dialogue),
        "tone": calculate_tone(audio_segment, sr, dialogue),
        "resonance": calculate_resonance(audio_segment, sr, dialogue),
        "spectral_skewness": calculate_spectral_skewness(audio_segment, sr, dialogue)
    }
    keywords = []
    for metric, value in metrics.items():
        if value > 0.7:
            keywords.append(f"{metric} high")
        elif value < 0.3:
            keywords.append(f"{metric} low")
        else:
            keywords.append(f"{metric} medium")
    return metrics, keywords

# Stage 3: Situation Detection
SITUATIONS = [
    "abuse risk", "payment issues", "escalation need", "technical problems", "overlapping voices",
    "fraud report", "loan inquiry", "balance dispute", "transfer request", "satisfaction feedback",
    "account closure", "identity verification", "lost card", "money sent to wrong number"
]

SITUATION_REGEX = {
    "abuse risk": r"\b(idiot|stupid|threat|insult|shut up)\b",
    "payment issues": r"\b(deny|refuse|not pay|bill wrong|refund|due|debt)\b",
    "escalation need": r"\b(manager|supervisor|escalate|higher up)\b",
    "technical problems": r"\b(error|bug|not working|app crash|transaction fail)\b",
    "overlapping voices": None,
    "fraud report": r"\b(fraud|unauthorized|stolen|hack|scam)\b",
    "loan inquiry": r"\b(loan|application|interest rate|borrow)\b",
    "balance dispute": r"\b(wrong balance|dispute charge|overdraft)\b",
    "transfer request": r"\b(transfer|wire|send money|move funds)\b",
    "satisfaction feedback": r"\b(great service|bad experience|thank you|complaint)\b",
    "account closure": r"\b(close account|cancel|terminate)\b",
    "identity verification": r"\b(verify|ID|security question|password)\b",
    "lost card": r"\b(lost.*card|card.*lost|misplaced.*card|card.*missing)\b",
    "money sent to wrong number": r"\b(wrong.*number|wrong.*recipient|wrong.*account|sent.*wrong)\b"
}

def detect_situations(dialogue, emotion, emotion_score, metrics):
    detected = []
    lowered = dialogue.lower()
    for situation, pattern in SITUATION_REGEX.items():
        match = False
        if pattern and re.search(pattern, lowered):
            match = True
        if situation == "abuse risk" and match and emotion in ["anger", "hostility"] and emotion_score > 0.7 and metrics["loudness"] > 0.7:
            detected.append(situation)
        elif situation == "payment issues" and match and emotion in ["frustration", "anger"] and metrics["energy"] > 0.7:
            detected.append(situation)
        elif situation == "escalation need" and match and emotion in ["impatience", "desperation"] and metrics["spectral_flux"] > 0.7:
            detected.append(situation)
        elif situation == "technical problems" and match and emotion == "confusion" and metrics["entropy"] < 0.3:
            detected.append(situation)
        elif situation == "overlapping voices" and metrics["spectral_flux"] > 0.7 and metrics["zero_cross_rate"] > 0.5:
            detected.append(situation)
        elif situation == "fraud report" and match and emotion == "anxiety" and metrics["emotional_intensity"] > 0.7:
            detected.append(situation)
        elif situation == "loan inquiry" and match and emotion == "curiosity" and metrics["speaking_rate"] < 0.5:
            detected.append(situation)
        elif situation == "balance dispute" and match and emotion == "frustration" and metrics["speaking_rate"] > 0.7:
            detected.append(situation)
        elif situation == "transfer request" and match and emotion == "negotiation" and metrics["pitch_variation"] > 0.5:
            detected.append(situation)
        elif situation == "satisfaction feedback" and match and emotion in ["gratitude", "satisfaction"] and metrics["emotional_intensity"] < 0.3:
            detected.append(situation)
        elif situation == "account closure" and match and emotion == "resignation" and metrics["energy"] < 0.3:
            detected.append(situation)
        elif situation == "identity verification" and match and emotion == "defensiveness" and metrics["resonance"] > 0.7:
            detected.append(situation)
        elif situation == "lost card" and match and emotion == "anxiety" and metrics["emotional_intensity"] > 0.7:
            detected.append(situation)
        elif situation == "money sent to wrong number" and match and emotion == "frustration" and metrics["speaking_rate"] > 0.7:
            detected.append(situation)
    return detected

# Prompt Directory Management
def build_prompt_directory():
    directory = {}
    priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    for priority in priorities:
        priority_path = os.path.join(PROMPTS_DIR, priority)
        if not os.path.exists(priority_path):
            continue
        for subdir in os.listdir(priority_path):
            subdir_path = os.path.join(priority_path, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for file in os.listdir(subdir_path):
                if not file.endswith(".json"):
                    continue
                file_path = os.path.join(priority, subdir, file)
                with open(os.path.join(subdir_path, file), 'r') as f:
                    directory[file_path] = json.load(f)
    return directory

def build_keyword_index(directory):
    index = {}
    for path, data in directory.items():
        for kw in data["keywords"]:
            if kw not in index:
                index[kw] = []
            index[kw].append(path)
    with open(os.path.join(PROMPTS_DIR, "keyword_index.json"), 'w') as f:
        json.dump(index, f, indent=2)
    return index

# Rules for Lookup
RULES = {
    "priorities": {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1},
    "weights": {"situation": 5, "emotion": 3, "audio_metric": 2},
    "overrides": {"abuse risk": "CRITICAL", "fraud report": "CRITICAL", "lost card": "CRITICAL"}
}

# Stage 4: Prompt Generation
def extract_keywords(emotion, audio_keywords, situations):
    keywords = [emotion] + audio_keywords + situations
    return keywords

def lookup_prompt(keywords, directory, index):
    candidate_paths = set()
    for kw in keywords:
        if kw in index:
            if not candidate_paths:
                candidate_paths = set(index[kw])
            else:
                candidate_paths &= set(index[kw])
    
    if not candidate_paths:
        candidate_paths = set([p for p in directory if "general/default" in p])
    
    scores = {}
    for path in candidate_paths:
        priority_score = RULES["priorities"].get(directory[path]["priority"], 1)
        kw_match_count = sum(1 for kw in keywords if kw in directory[path]["keywords"])
        score = priority_score * 10 + kw_match_count
        for kw in keywords:
            if kw in RULES["overrides"]:
                score += 50
            if "high" in kw:
                score += RULES["weights"]["audio_metric"]
        scores[path] = score
    
    if not scores:
        best_path = "LOW/general/default.json"
    else:
        best_path = max(scores, key=scores.get)
    
    return directory[best_path]["prompt_template"], directory[best_path]["priority"]

# Process Single Utterance
def process_utterance(audio_path, transcription_path, directory, index):
    y, sr = torchaudio.load(audio_path)
    with open(transcription_path, 'r') as f:
        segment = json.load(f)[0]  # Single utterance
    start, end = segment['startTime'], segment['endTime']
    dialogue = segment.get('dialogue', '')
    y_seg = y[0][int(start * sr):int(end * sr)]
    tmp_path = os.path.join("/tmp", "utterance_tmp.wav")
    torchaudio.save(tmp_path, y_seg.unsqueeze(0), sr)

    # Stage 0: Keyword-Based Prompt
    stage_zero_result = stage_zero_prompt(dialogue)
    if stage_zero_result:
        os.remove(tmp_path)
        return {
            "sentence": dialogue,
            "emotion": "N/A (Stage 0)",
            "emotion_score": 0.0,
            "audio_metrics": {},
            "situations": [],
            "prompt": stage_zero_result["prompt"],
            "priority": stage_zero_result["priority"],
            "startTime": round(start, 2),
            "endTime": round(end, 2)
        }

    # Stage 1: Emotion
    t_emo, t_conf = predict_text_emotion(dialogue)
    a_emo, a_conf = predict_audio_emotion(tmp_path)
    m_emo, m_conf = merge_emotions(t_emo, t_conf, a_emo, a_conf)
    regex_emo = regex_based_emotion_override(dialogue)
    if regex_emo and m_emo == "Unknown":
        m_emo = regex_emo
        m_conf = 0.8

    # Stage 2: Audio Metrics
    metrics, audio_keywords = compute_audio_metrics(y_seg.numpy(), sr, dialogue, start, end)

    # Stage 3: Situations
    situations = detect_situations(dialogue, m_emo, m_conf, metrics)

    # Stage 4: Prompt
    keywords = extract_keywords(m_emo, audio_keywords, situations)
    prompt, priority = lookup_prompt(keywords, directory, index)

    os.remove(tmp_path)
    # Final Result
    return {
        "sentence": dialogue,
        "emotion": m_emo,
        "emotion_score": m_conf,
        "audio_metrics": metrics,
        "situations": situations,
        "prompt": prompt,
        "priority": priority,
        "startTime": round(start, 2),
        "endTime": round(end, 2)
    }

# Main Execution
if __name__ == "__main__":
    audio_path = os.path.join(AUDIO_DIR, "6.wav")
    transcription_path = os.path.join(TRANSCRIPTION_DIR, "6.json")
    
    if not os.path.isfile(audio_path) or not os.path.isfile(transcription_path):
        print("Error: utterance.wav or utterance.json not found.")
        exit(1)

    directory = build_prompt_directory()
    index = build_keyword_index(directory)
    
    result = process_utterance(audio_path, transcription_path, directory, index)
    
    out_path = os.path.join(OUTPUT_DIR, "utterance_results.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Processed utterance. Wrote to {out_path}")