import os
import json
import re
import torch
import torchaudio
import librosa
import numpy as np


# Directories
AUDIO_DIR = "./audio"
TRANSCRIPTION_DIR = "./transcription"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Keyword Prompt Map (Customer full set)
KEYWORD_PROMPT_MAP = {
    r"\b(idiot|stupid|shut up|jerk|moron|fool|dumb|incompetent|pathetic|imbecile|loser|do your job|nonsense|ridiculous|deaf|sue you|report you|get you fired|you’re a loser|you are a loser|reporting you|i’ll complain|i will complain|file a complaint|take this to court|this is absurd|what a joke|are you blind)\b": {
        "prompt": "Escalate to manager for abusive behavior.",
        "priority": "CRITICAL"
    },
    r"\b(account is hacked|heacked|fraud|scammed|unauthorized|scammer|stolen|hack|hacked my account|fraudulent|identity theft|suspicious activity|phished|phishing|fake transaction|breached|stole my cards|stole my money|hacked my card|someone used my account|unrecognized charge|not my transaction|my account was hacked|someone hacked my account|fraud on my account|scam on my card|unauthorized transaction|stolen funds|account compromised|suspicious charge|identity stolen|card was hacked|not my purchase|fake charge|this is a scam|i got scammed|someone’s using my card|someone is using my card|my money’s gone|my money is gone|hack on my account|stolen|stole)\b": {
        "prompt": "Escalate fraud, verify account.",
        "priority": "CRITICAL"
    },
    r"\b(lost my card|card is lost|lost card|misplaced my card|card’s missing|card is missing|can’t find my card|cannot find my card|i’ve lost my card|i have lost my card|my card’s gone|my card is gone|card is missing|lost the card|cannot find my card)\b": {
        "prompt": "Escalate lost card, replace card.",
        "priority": "CRITICAL"
    },
    r"\b(i am not paying|i am not gonna pay|i am not going to pay|why should i pay|bill is wrong|wrongly charged|won’t pay|will not pay|refuse to pay|not gonna pay|i ain’t paying|i am done paying|incorrect bill|overcharged|charged twice|double charged|billing mistake|wrong amount|dispute bill|give me a refund|need a refund|take it off|remove this charge|i’m not gonna pay this|i am not going to pay this|i refuse to pay|i won’t pay this|i will not pay this|no way i’m paying|no way i am paying|i’m not paying that|bill is incorrect|wrong billing|charged wrong|billed incorrectly|error on my bill|wrong charge on bill|i want my money back|refund me now|take off this charge|reverse this charge|cancel this payment|this bill is ridiculous|why am i being charged|charged for nothing|fix this bill)\b": {
        "prompt": "Resolve payment dispute, escalate to billing.",
        "priority": "HIGH"
    },
    r"\b(balance is wrong|wrong balance|dispute charge|incorrect charge|wrong charge|account balance wrong|balance incorrect|wrong account balance|balance doesn’t match|balance does not match|balance off|charge is wrong|didn’t authorize|did not authorize|not my charge|unauthorized charge|discrepancy in balance|overdraft fee|wrong deduction|incorrect deduction|missing funds|my balance is incorrect|balance is off|account balance is wrong|balance not right|wrong balance on account|i didn’t make this charge|i did not make this charge|charge not mine|didn’t approve this|did not approve this|unauthorized deduction|wrong charge on account|missing money|funds missing|overdraft error|incorrect balance|balance discrepancy)\b": {
        "prompt": "Review balance dispute.",
        "priority": "HIGH"
    },
    r"\b(sent to wrong number|wrong number transfer|money to wrong account|transferred to wrong number|it’s the wrong recipient|it is the wrong recipient|sent money to wrong person|wrong account transfer|i’ve sent to wrong number|i have sent to wrong number|money sent to wrong number)\b": {
        "prompt": "Resolve wrong transfer, initiate recovery.",
        "priority": "HIGH"
    },
    r"\b(app not working|app is not working|app isn’t working|app crash|transaction fail|error|bug|app keeps crashing|app won’t load|app will not load|app frozen|app error|can’t log in|cannot log in|login issue|payment failed|transfer failed|transaction error|couldn’t process payment|could not process payment|website down|system error|platform not working|technical issue|glitch|app stopped working|app doesn’t work|app does not work|can’t use the app|cannot use the app|app is down|login not working|site not working|website crashed|system is down|technical error|app glitching)\b": {
        "prompt": "Assist technical issue, escalate if unresolved.",
        "priority": "HIGH"
    }
}


# Abusive-only regex for agents
ABUSIVE_PATTERN = list(KEYWORD_PROMPT_MAP.keys())[0]


# Stage 0: Keyword-Based Prompt Selection
def stage_zero_prompt(dialogue, speaker):
    lowered = dialogue.lower().strip()
    matches = []

    if speaker.lower() == "customer":
        for pattern, prompt_data in KEYWORD_PROMPT_MAP.items():
            if re.search(pattern, lowered):
                matches.append((prompt_data["priority"], prompt_data))
        if matches:
            matches.sort(key=lambda x: {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}[x], reverse=True)
            top_prompt = matches
            return {
                "prompt": top_prompt["prompt"],
                "priority": top_prompt["priority"],
                "from_stage_zero": True
            }

    elif speaker.lower() == "agent":
        if re.search(ABUSIVE_PATTERN, lowered):
            return {
                "prompt": "Abusive behavior by agent.",
                "priority": "CRITICAL",
                "from_stage_zero": True
            }

    return None


# Stage 1: Loudness Check
def calculate_loudness(audio_segment, sr, dialogue):
    if len(audio_segment) < 64:
        return 0.5
    max_amplitude = np.max(np.abs(audio_segment))
    raw_rms = np.sqrt(np.mean(audio_segment**2))
    rms = librosa.feature.rms(y=audio_segment, frame_length=2048, hop_length=512).mean()
    db = librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0]
    min_db, max_db = -80, 0
    loudness = (db - min_db) / (max_db - min_db)
    print(f"Debug: MaxAmp={max_amplitude:.6f}, RawRMS={raw_rms:.6f}, LibrosaRMS={rms:.6f}, dB={db:.2f}, Loudness={loudness:.3f}")
    return min(max(float(loudness), 0.0), 1.0)


# Process Single Utterance
def process_utterance(y, sr, segment):
    start = segment.get("startTime", 0.0)
    end = segment.get("endTime", 0.0)
    dialogue = segment.get("dialogue", "")
    speaker = segment.get("speaker", "Customer")

    # Slice audio
    y_seg = y[:, int(start * sr):int(end * sr)][0]  # pick first channel

    # Stage 0 match
    stage_zero_result = stage_zero_prompt(dialogue, speaker)

    # Stage 1 loudness
    loudness = calculate_loudness(y_seg.numpy(), sr, dialogue)
    if loudness > 0.8:
        if speaker.lower() == "customer":
            return {
                "sentence": dialogue,
                "speaker": speaker,
                "loudness": round(loudness, 3),
                "outlier": True,
                "prompt": "Calm down the customer",
                "priority": "HIGH",
                "startTime": round(start, 2),
                "endTime": round(end, 2)
            }
        elif speaker.lower() == "agent":
            return {
                "sentence": dialogue,
                "speaker": speaker,
                "loudness": round(loudness, 3),
                "outlier": True,
                "prompt": "Agent is being too aggressive",
                "priority": "HIGH",
                "startTime": round(start, 2),
                "endTime": round(end, 2)
            }

    # Return Stage 0 or default
    if stage_zero_result:
        return {
            "sentence": dialogue,
            "speaker": speaker,
            "loudness": round(loudness, 3),
            "outlier": True,
            "prompt": stage_zero_result["prompt"],
            "priority": stage_zero_result["priority"],
            "startTime": round(start, 2),
            "endTime": round(end, 2)
        }

    return {
        "sentence": dialogue,
        "speaker": speaker,
        "loudness": round(loudness, 3),
        "outlier": False,
        "prompt": "No urgent prompt",
        "priority": "LOW",
        "startTime": round(start, 2),
        "endTime": round(end, 2)
    }


# Main Execution
if __name__ == "__main__":
    audio_path = os.path.join(AUDIO_DIR, "1.wav")
    transcription_path = os.path.join(TRANSCRIPTION_DIR, "1.json")

    if not os.path.isfile(audio_path) or not os.path.isfile(transcription_path):
        print("Error: audio or transcription file not found.")
        exit(1)

    # Load audio once
    y, sr = torchaudio.load(audio_path)
    # Load multiple utterances
    with open(transcription_path, "r") as f:
        transcription = json.load(f)

    if not isinstance(transcription, list):
        transcription = [transcription]

    results = []
    for segment in transcription:
        result = process_utterance(y, sr, segment)
        results.append(result)

    out_path = os.path.join(OUTPUT_DIR, "utterance_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Processed {len(results)} utterance(s). Wrote to {out_path}")
