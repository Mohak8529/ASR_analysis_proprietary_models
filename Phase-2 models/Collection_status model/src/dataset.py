import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from collections import defaultdict

class CollectionStatusDataset(Dataset):
    def __init__(self, data_dir, max_text_len=512):
        self.data_dir = data_dir
        self.max_text_len = max_text_len    # Tokens
        self.label_map = {
            "PreDue": 0,
            "PostDue < 30": 1,
            "PostDue > 30": 2,
            "PostDue > 60": 3
        }
        self.syntactic_cues = {
            0: ["due soon", "reminder call", "prepare for payment", "statement sent", "confirm due date"],
            1: ["recently overdue", "late fee risk", "bring current", "short delay", "prompt payment"],
            2: ["month overdue", "credit reporting", "arrange plan", "delinquent status", "escalation warning"],
            3: ["collections referral", "legal proceedings", "account closure", "final demand", "credit bureau notice"]
        }
        self.files = self._load_files()

    def _load_files(self):
        audio_dir = os.path.join(self.data_dir, "audio")
        files = [f.split(".")[0] for f in os.listdir(audio_dir) if f.endswith(".wav")]
        return sorted(files)

    def _load_audio(self, file_id):
        path = os.path.join(self.data_dir, "audio", f"{file_id}.wav")
        audio, sr = librosa.load(path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]
        return mel_spec, mel_spec.size(-1)  # Return spectrogram and time length

    def _load_text(self, file_id):
        path = os.path.join(self.data_dir, "translation", f"{file_id}_translation.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip().lower()
        tokens = text.split()[:self.max_text_len]
        if len(tokens) < self.max_text_len:
            tokens += ["<pad>"] * (self.max_text_len - len(tokens))
        vocab = defaultdict(lambda: len(vocab))
        vocab["<pad>"] = 0
        token_ids = [vocab[token] for token in tokens]
        cue_features = [0] * len(self.label_map)
        for label, cues in self.syntactic_cues.items():
            if any(cue in text for cue in cues):
                cue_features[label] = 1
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(cue_features, dtype=torch.float32)

    def _load_label(self, file_id):
        path = os.path.join(self.data_dir, "collection_status", f"{file_id}_collection_status.txt")
        with open(path, "r", encoding="utf-8") as f:
            label = f.read().strip()
        return self.label_map[label]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]
        audio, audio_len = self._load_audio(file_id)
        text, cue_features = self._load_text(file_id)
        label = self._load_label(file_id)
        return audio, audio_len, text, cue_features, label