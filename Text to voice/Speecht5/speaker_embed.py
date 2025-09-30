import os
import sys
import torch
import numpy as np
import torchaudio
from speechbrain.inference import EncoderClassifier

EMBED_MODEL = "speechbrain/spkrec-xvect-voxceleb"
CACHE_DIR = os.path.join("speecht5_model", "spkrec-xvect-voxceleb")

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = EncoderClassifier.from_hparams(
    source=EMBED_MODEL,
    savedir=CACHE_DIR,
    run_opts={"device": device}
)

def extract_embedding(audio_filepath: str) -> torch.Tensor:
    signal, sr = torchaudio.load(audio_filepath)
    signal = signal.mean(dim=0, keepdim=True) if signal.ndim > 1 else signal
    embedding = classifier.encode_batch(signal)  # shape (1,1,512)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding.squeeze(1).to(device)      # (1,512)

def save_embedding(audio_filepath: str, output_numpy: str = "speaker.npy"):
    emb = extract_embedding(audio_filepath).cpu().numpy()
    np.save(output_numpy, emb, allow_pickle=False)
    print(f"Saved speaker embedding to '{output_numpy}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python speaker_embed.py <audio_file>")
        print("Example: python speaker_embed.py reference_audio_male.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        sys.exit(1)
    
    save_embedding(audio_file)