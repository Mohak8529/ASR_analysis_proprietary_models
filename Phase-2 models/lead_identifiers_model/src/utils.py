import os
import torch
import numpy as np
import librosa

def add_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    noisy_audio = audio + noise
    return noisy_audio

def validate_data(data_dir, file_id):
    files = [
        os.path.join(data_dir, "audio", f"{file_id}.wav"),
        os.path.join(data_dir, "transcription", f"{file_id}_transcription.txt"),
        os.path.join(data_dir, "translation", f"{file_id}_translation.txt"),
        os.path.join(data_dir, "lead_identifiers", f"{file_id}_lead_identifiers.txt")
    ]
    for file_path in files:
        if not os.path.exists(file_path):
            return False, f"Missing file: {file_path}"
    try:
        audio, _ = librosa.load(files[0], sr=16000)
        if len(audio) == 0:
            return False, f"Empty audio: {files[0]}"
        for txt_file in files[1:]:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return False, f"Empty file: {txt_file}"
    except Exception as e:
        return False, f"Error in {file_path}: {str(e)}"
    return True, "All files valid"

def evaluate_model(model, dataloader, device):
    model.eval()
    identifier_correct = 0
    identifier_total = 0
    with torch.no_grad():
        for audio, audio_len, trans, transcr, trans_len, transcr_len, id_labels in dataloader:
            audio, audio_len, trans, transcr, trans_len, transcr_len, id_labels = (
                audio.to(device),
                audio_len.to(device),
                trans.to(device),
                transcr.to(device),
                trans_len.to(device),
                transcr_len.to(device),
                id_labels.to(device)
            )
            identifier_logits = model(audio, audio_len, trans, transcr, trans_len)
            _, id_predicted = torch.max(identifier_logits, 2)
            mask = (id_labels != 0)
            identifier_correct += ((id_predicted == id_labels) & mask).sum().item()
            identifier_total += mask.sum().item()
    identifier_accuracy = identifier_correct / identifier_total if identifier_total > 0 else 0
    return identifier_accuracy

def extract_identifiers(trans_tokens, identifier_logits, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    _, id_predicted = torch.max(identifier_logits, 2)
    identifiers = []
    for i, seq in enumerate(trans_tokens):
        words = [inv_vocab.get(token.item(), "<unk>") for token in seq]
        labels = id_predicted[i]
        seq_identifiers = []
        current_identifier = []
        for word, label in zip(words, labels):
            if label.item() == 1:  # B-KEY
                if current_identifier:
                    seq_identifiers.append(" ".join(current_identifier))
                    current_identifier = []
                current_identifier.append(word)
            elif label.item() == 2:  # I-KEY
                current_identifier.append(word)
            elif label.item() == 0 and current_identifier:  # O
                seq_identifiers.append(" ".join(current_identifier))
                current_identifier = []
        if current_identifier:
            seq_identifiers.append(" ".join(current_identifier))
        identifiers.append(seq_identifiers)
    return identifiers