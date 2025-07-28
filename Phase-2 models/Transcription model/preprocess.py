import os
import torch
import torchaudio
import numpy as np
import librosa
from tokenizers import Tokenizer
from datasets import Dataset
from pathlib import Path
import pickle

def load_data(audio_dir, trans_dir, text_dir):
    """Load Taglish audio-transcription pairs and bank call text."""
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    trans_files = sorted(Path(trans_dir).glob("*_transcription.txt"))
    text_file = Path(text_dir) / "bank_call_text.txt"

    # Pair audio and transcriptions
    data = []
    for audio_path in audio_files:
        trans_path = Path(trans_dir) / f"{audio_path.stem}_transcription.txt"
        if trans_path.exists():
            with open(trans_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            data.append({"audio_path": str(audio_path), "text": text})
        else:
            print(f"Warning: Transcription file {trans_path} not found for {audio_path}")

    # Load LM text
    lm_texts = []
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lm_texts.extend(line.strip() for line in lines if line.strip())
    else:
        print(f"Warning: {text_file} not found, LM dataset will be empty.")

    return data, lm_texts

def process_audio(audio_path, target_sr=16000):
    """Convert audio to mel-spectrogram, like Whisper."""
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    if waveform.shape[0] > 1:  # Stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = waveform.squeeze().numpy()
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=target_sr, n_mels=80, fmax=8000, hop_length=160
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    return log_mel.T

def split_tokens(tokens, max_length, overlap=50):
    """Split token sequence into chunks with overlap."""
    if len(tokens) <= max_length:
        return [tokens]
    
    chunks = []
    step = max_length - overlap
    for i in range(0, len(tokens), step):
        chunk = tokens[i:i + max_length]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

def preprocess_data(data, lm_texts, tokenizer_path="custom_taglish_tokenizer.json", max_length=2048, chunk_length=1024, overlap=50):
    """Tokenize text and process audio, return datasets with checkpointing."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load checkpoint if exists
    checkpoint_file = "processed_data/preprocess_checkpoint.pkl"
    processed_data = []
    processed_indices = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            processed_data = checkpoint["processed_data"]
            processed_indices = set(checkpoint["processed_indices"])
            print(f"Resumed preprocessing from checkpoint: {len(processed_data)} items already processed.")

    try:
        for idx, item in enumerate(data):
            if idx in processed_indices:
                continue  # Skip already processed items
            
            # Process audio
            log_mel = process_audio(item["audio_path"])
            
            # Tokenize and split transcription
            encoding = tokenizer.encode(item["text"])
            tokens = encoding.ids
            if len(tokens) > max_length:
                print(f"Info: Transcription for {item['audio_path']} (length {len(tokens)}) will be split into chunks of {chunk_length} tokens.")
                token_chunks = split_tokens(tokens, chunk_length, overlap)
                for chunk_idx, token_chunk in enumerate(token_chunks):
                    processed_data.append({
                        "log_mel": log_mel,  # Same audio for all chunks
                        "tokens": token_chunk,
                        "text": tokenizer.decode(token_chunk),
                        "chunk_idx": chunk_idx
                    })
            else:
                processed_data.append({
                    "log_mel": log_mel,
                    "tokens": tokens,
                    "text": item["text"],
                    "chunk_idx": 0
                })
            
            # Save checkpoint after each item
            processed_indices.add(idx)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    "processed_data": processed_data,
                    "processed_indices": processed_indices
                }, f)
    
    except KeyboardInterrupt:
        print("Preprocessing interrupted, saving checkpoint...")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                "processed_data": processed_data,
                "processed_indices": processed_indices
            }, f)
        print(f"Checkpoint saved to {checkpoint_file}")
        exit(0)

    # Process LM texts
    lm_tokens = []
    for text in lm_texts:
        encoding = tokenizer.encode(text)
        tokens = encoding.ids
        if len(tokens) > max_length:
            print(f"Warning: LM text truncated from {len(tokens)} to {max_length} tokens: {text[:50]}...")
            tokens = tokens[:max_length]
        lm_tokens.append(tokens)
    
    # Create datasets
    stt_dataset = Dataset.from_list(processed_data)
    lm_dataset = Dataset.from_dict({"tokens": lm_tokens})
    
    return stt_dataset, lm_dataset, tokenizer

def main():
    audio_dir = "Dataset/Audio"
    trans_dir = "Dataset/Transcription"
    text_dir = "Dataset/Text_data"
    
    data, lm_texts = load_data(audio_dir, trans_dir, text_dir)
    if not data:
        raise ValueError("No audio-transcription pairs found. Check file naming and paths.")
    
    stt_dataset, lm_dataset, tokenizer = preprocess_data(data, lm_texts)
    
    # Save datasets
    os.makedirs("processed_data", exist_ok=True)
    stt_dataset.save_to_disk("processed_data/stt_dataset")
    lm_dataset.save_to_disk("processed_data/lm_dataset")
    os.makedirs("processed_data/tokenizer", exist_ok=True)
    tokenizer.save("processed_data/tokenizer/custom_taglish_tokenizer.json")
    
    # Clean up checkpoint file
    checkpoint_file = "processed_data/preprocess_checkpoint.pkl"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed checkpoint file {checkpoint_file}")
    
    print(f"Processed {len(stt_dataset)} audio-text pairs and {len(lm_dataset)} LM texts.")

if __name__ == "__main__":
    os.makedirs("processed_data", exist_ok=True)
    main()