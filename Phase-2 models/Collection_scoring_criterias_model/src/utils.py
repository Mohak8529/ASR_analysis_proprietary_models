import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import json

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load and resample audio to mono."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze()

def compute_mel_spectrogram(waveform: torch.Tensor, sample_rate: int, n_mels: int) -> torch.Tensor:
    """Compute mel-spectrogram with tonal features."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=400, hop_length=160
    )
    spec = mel_transform(waveform)
    spec = torch.log(spec + 1e-10)  # Log-mel for stability
    return spec

def extract_tonal_features(waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract pitch and energy for tonal analysis."""
    # Simple pitch estimation (zero-crossing rate as proxy)
    zcr = torchaudio.transforms.compute_deltas(waveform.unsqueeze(0)).squeeze()
    # Energy (RMS)
    energy = torch.sqrt(torch.mean(waveform**2, dim=-1, keepdim=True))
    return zcr, energy

def build_vocabulary(text_files: List[Path], max_vocab_size: int = 30000) -> Dict[str, int]:
    """Build vocabulary from text files."""
    word_counts = {}
    for file in text_files:
        with open(file, 'r') as f:
            text = f.read().lower().split()
            for word in text:
                word_counts[word] = word_counts.get(word, 0) + 1
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size-2])}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

def tokenize_text(text: str, vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """Tokenize text to indices."""
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens][:max_len]
    if len(indices) < max_len:
        indices += [vocab['<pad>']] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)

def load_criteria(json_path: str) -> torch.Tensor:
    """Load 22 binary labels from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    labels = [
        data['call_closed_properly'],
        data['call_open_timely_manner'],
        data['standard_opening_spiel'],
        data['verification_of_account_security'],
        data['friendly_confident_tone'],
        data['attentive_listening'],
        data['call_control_efficiency'],
        data['follow_policies_procedure'],
        data['service_reminder'],
        data['customer_alternate_number'],
        data['call_record_clause'],
        data['pid_process'],
        data['udcp_process'],
        data['call_avoidance'],
        data['misleading_information'],
        data['data_manipulation'],
        data['call_recap'],
        data['ask_additional_number'],
        data['probing_questions_effectiveness'],
        data['payment_resolution_actions'],
        data['payment_delay_consequences'],
        data['properly_document_the_call']
    ]
    return torch.tensor(labels, dtype=torch.float32)