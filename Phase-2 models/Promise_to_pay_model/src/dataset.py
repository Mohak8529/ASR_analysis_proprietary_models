import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from collections import defaultdict

class PromiseToPayDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_map = {
            "Complete Settlement": 0,
            "Partial Settlement": 1,
            "Promise Broken": 2,
            "Denial": 3
        }
        self.files = self._load_files()
        self.vocab = self._build_vocab()

    def _load_files(self):
        audio_dir = os.path.join(self.data_dir, "audio")
        files = [f.split(".")[0] for f in os.listdir(audio_dir) if f.endswith(".wav")]
        return sorted(files)

    def _build_vocab(self):
        vocab = defaultdict(lambda: len(vocab))
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        for file_id in self.files:
            for folder in ["translation", "transcription"]:
                path = os.path.join(self.data_dir, folder, f"{file_id}_{folder}.txt")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read().strip().lower()
                        tokens = text.split()
                        for token in tokens:
                            _ = vocab[token]
        return vocab

    def _load_audio(self, file_id):
        path = os.path.join(self.data_dir, "audio", f"{file_id}.wav")
        audio, sr = librosa.load(path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # [1, n_mels, time]
        audio_len = mel_spec.size(-1)
        return mel_spec, audio_len

    def _load_text(self, file_id, folder="translation"):
        path = os.path.join(self.data_dir, folder, f"{file_id}_{folder}.txt")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip().lower()
        tokens = text.split()
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long), len(tokens)

    def _load_label(self, file_id):
        path = os.path.join(self.data_dir, "promise_to_pay_category", f"{file_id}_promise_to_pay.txt")
        with open(path, "r", encoding="utf-8") as f:
            label = f.read().strip()
        return self.label_map[label]

    def _load_keywords(self, file_id):
        path = os.path.join(self.data_dir, "critical_keywords", f"{file_id}_critical_keywords.txt")
        with open(path, "r", encoding="utf-8") as f:
            keywords = [kw.strip().lower() for kw in f.read().split(",")]
        trans_path = os.path.join(self.data_dir, "translation", f"{file_id}_translation.txt")
        with open(trans_path, "r", encoding="utf-8") as f:
            text = f.read().strip().lower()
        tokens = text.split()
        labels = ["O"] * len(tokens)
        for keyword in keywords:
            kw_tokens = keyword.split()
            for i in range(len(tokens) - len(kw_tokens) + 1):
                if tokens[i:i+len(kw_tokens)] == kw_tokens:
                    labels[i] = "B-KEY"
                    for j in range(i+1, i+len(kw_tokens)):
                        labels[j] = "I-KEY"
        label_ids = [0 if l == "O" else 1 if l == "B-KEY" else 2 for l in labels]
        return torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]
        audio, audio_len = self._load_audio(file_id)
        trans_tokens, trans_len = self._load_text(file_id, "translation")
        transcr_tokens, transcr_len = self._load_text(file_id, "transcription")
        label = self._load_label(file_id)
        keyword_labels = self._load_keywords(file_id)
        return audio, audio_len, trans_tokens, transcr_tokens, trans_len, transcr_len, label, keyword_labels