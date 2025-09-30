import os
import torch
import soundfile as sf
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

MODEL_DIR = "speecht5_model"
SPEAKER_EMB_FILE = "speaker.npy"
TEXT = "Hello, this is a test using SpeechT5 with speaker embedding."
OUTPUT_WAV = "output.wav"

def load_model(model_dir=MODEL_DIR):
    proc = SpeechT5Processor.from_pretrained(model_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_dir)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return proc, model, vocoder

def load_speaker_embedding(path=SPEAKER_EMB_FILE):
    emb = np.load(path)
    # The embedding from speaker_embed.py is already in shape (1, 512)
    # We just need to convert to tensor without adding extra dimensions
    tensor = torch.tensor(emb, dtype=torch.float32)
    
    # Ensure it's in the correct shape: [batch_size, embedding_dim]
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # Shape: [1, 512]
    elif tensor.dim() == 2 and tensor.size(0) == 1:
        pass  # Already correct shape: [1, 512]
    else:
        # If it's somehow a different shape, reshape it
        tensor = tensor.view(1, -1)
    
    print(f"Speaker embedding shape: {tensor.shape}")
    return tensor

def text_to_speech(proc, model, vocoder, text, speaker_emb, output_wav):
    inputs = proc(text=text, return_tensors="pt")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Speaker embedding shape: {speaker_emb.shape}")
    
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=vocoder)
    
    wav = speech.cpu().numpy().squeeze()
    sf.write(output_wav, wav, samplerate=16000)
    print(f"Saved speech to '{output_wav}'")

def main():
    if not os.path.isdir(MODEL_DIR):
        print("Run download_model.py first.")
        return
    if not os.path.isfile(SPEAKER_EMB_FILE):
        print(f"Run speaker_embed.py to create '{SPEAKER_EMB_FILE}' before using this.")
        return
        
    proc, model, vocoder = load_model()
    speaker_emb = load_speaker_embedding()
    text_to_speech(proc, model, vocoder, TEXT, speaker_emb, OUTPUT_WAV)

if __name__ == "__main__":
    main()