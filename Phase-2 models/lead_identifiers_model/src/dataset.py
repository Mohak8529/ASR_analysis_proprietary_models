import os
import torch
import librosa
import numpy as np

class LeadIdentifiersDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        self.transcr_dir = os.path.join(data_dir, "transcription")
        self.trans_dir = os.path.join(data_dir, "translation")
        self.identifiers_dir = os.path.join(data_dir, "lead_identifiers")
        self.file_ids = [f.split(".")[0] for f in os.listdir(self.audio_dir) if f.endswith(".wav")]
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for file_id in self.file_ids:
            for dir_path in [self.transcr_dir, self.trans_dir]:
                file_path = os.path.join(dir_path, f"{file_id}_{'transcription' if dir_path == self.transcr_dir else 'translation'}.txt")
                with open(file_path, "r", encoding="utf-8") as f:
                    words = f.read().strip().lower().split()
                    for word in words:
                        if word not in vocab:
                            vocab[word] = idx
                            idx += 1
        return vocab

    def _load_audio(self, file_id):
        audio_path = os.path.join(self.audio_dir, f"{file_id}.wav")
        audio, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0), len(mel_spec[0])

    def _load_text(self, file_id, dir_path, file_type):
        file_path = os.path.join(dir_path, f"{file_id}_{file_type}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            words = f.read().strip().lower().split()
        tokens = [self.vocab.get(word, self.vocab["<unk>"]) for word in words]
        return torch.tensor(tokens, dtype=torch.long), len(tokens)

    def _load_identifiers(self, file_id, trans_len):
        file_path = os.path.join(self.identifiers_dir, f"{file_id}_lead_identifiers.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            identifiers = f.read().strip().lower().split(", ")
        trans_file = os.path.join(self.trans_dir, f"{file_id}_translation.txt")
        with open(trans_file, "r", encoding="utf-8") as f:
            words = f.read().strip().lower().split()
        labels = [0] * trans_len  # 0: O (not a key identifier)
        for identifier in identifiers:
            id_words = identifier.split()
            for i in range(len(words) - len(id_words) + 1):
                if words[i:i+len(id_words)] == id_words:
                    labels[i] = 1  # B-KEY
                    for j in range(i+1, i+len(id_words)):
                        labels[j] = 2  # I-KEY
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        audio, audio_len = self._load_audio(file_id)
        trans_tokens, trans_len = self._load_text(file_id, self.trans_dir, "translation")
        transcr_tokens, transcr_len = self._load_text(file_id, self.transcr_dir, "transcription")
        identifier_labels = self._load_identifiers(file_id, trans_len)
        return audio, audio_len, trans_tokens, transcr_tokens, trans_len, transcr_len, identifier_labels

    def collate_fn(self, batch):
        audios, audio_lengths, trans_tokens, transcr_tokens, trans_lengths, transcr_lengths, identifier_labels = zip(*batch)
        max_audio_len = max(audio_lengths)
        padded_audios = torch.zeros(len(audios), 1, audios[0].size(1), max_audio_len)
        for i, audio in enumerate(audios):
            padded_audios[i, :, :, :audio.size(-1)] = audio
        max_text_len = max(max(trans_lengths), max(transcr_lengths))
        padded_trans = torch.zeros(len(trans_tokens), max_text_len, dtype=torch.long)
        padded_transcr = torch.zeros(len(transcr_tokens), max_text_len, dtype=torch.long)
        padded_identifiers = torch.zeros(len(identifier_labels), max_text_len, dtype=torch.long)
        for i in range(len(trans_tokens)):
            padded_trans[i, :len(trans_tokens[i])] = trans_tokens[i]
            padded_transcr[i, :len(transcr_tokens[i])] = transcr_tokens[i]
            padded_identifiers[i, :len(identifier_labels[i])] = identifier_labels[i]
        return (
            padded_audios,
            torch.tensor(audio_lengths),
            padded_trans,
            padded_transcr,
            torch.tensor(trans_lengths),
            torch.tensor(transcr_lengths),
            padded_identifiers
        )