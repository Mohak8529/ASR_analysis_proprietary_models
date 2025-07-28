import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
from src.utils import load_audio, compute_mel_spectrogram, extract_tonal_features, tokenize_text, load_criteria, build_vocabulary

class CollectionScoringDataset(Dataset):
    def __init__(self, dataset_path: str, vocab: Dict[str, int], config: Dict):
        self.dataset_path = Path(dataset_path)
        self.audio_dir = self.dataset_path / 'audio'
        self.transcription_dir = self.dataset_path / 'transcription'
        self.translation_dir = self.dataset_path / 'translation'
        self.criteria_dir = self.dataset_path / 'collection_scoring_criteria'
        self.sample_rate = config['data']['audio_sample_rate']
        self.n_mels = config['data']['mel_bands']
        self.max_seq_len = config['model']['max_seq_len']
        self.vocab = vocab

        self.audio_files = sorted(self.audio_dir.glob('*.wav'))
        self.transcription_files = sorted(self.transcription_dir.glob('*.txt'))
        self.translation_files = sorted(self.translation_dir.glob('*.txt'))
        self.criteria_files = sorted(self.criteria_dir.glob('*.json'))

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load audio
        audio_path = self.audio_files[idx]
        waveform = load_audio(audio_path, self.sample_rate)
        mel_spec = compute_mel_spectrogram(waveform, self.sample_rate, self.n_mels)
        zcr, energy = extract_tonal_features(waveform, self.sample_rate)

        # Load text
        transcription_path = self.transcription_files[idx]
        translation_path = self.translation_files[idx]
        with open(transcription_path, 'r') as f:
            transcription = f.read()
        with open(translation_path, 'r') as f:
            translation = f.read()
        # Concatenate with prompt for criterion-specific guidance
        text = f"{transcription} [SEP] {translation} [SEP] Check criteria"
        text_tokens = tokenize_text(text, self.vocab, self.max_seq_len)

        # Load labels
        criteria_path = self.criteria_files[idx]
        labels = load_criteria(criteria_path)

        return mel_spec, text_tokens, labels, zcr, energy

def get_dataloaders(dataset_path: str, config: Dict, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    # Build vocabulary
    transcription_files = sorted(Path(dataset_path + '/transcription').glob('*.txt'))
    translation_files = sorted(Path(dataset_path + '/translation').glob('*.txt'))
    text_files = transcription_files + translation_files
    vocab = build_vocabulary(text_files, config['model']['vocab_size'])

    # Full dataset
    full_dataset = CollectionScoringDataset(dataset_path, vocab, config)
    
    # Train/validation split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, vocab