
import torch
import argparse
from dataset import CollectionStatusDataset
from model import CollectionStatusModel
import os

def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {filename}")
    else:
        raise FileNotFoundError(f"Checkpoint {filename} not found")
    return model

def infer(audio_path, translation_path, checkpoint_path="../checkpoints/latest.pth"):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize dataset for preprocessing
    dataset = CollectionStatusDataset(data_dir="../dataset")
    
    # Load audio
    audio, audio_len = dataset._load_audio(os.path.basename(audio_path).split(".")[0])
    audio = audio.to(device)  # [1, n_mels, time]
    audio_len = torch.tensor([audio_len], dtype=torch.long).to(device)  # [1]
    
    # Load translation
    text, cue_features = dataset._load_text(os.path.basename(translation_path).split("_")[0])
    text = text.unsqueeze(0).to(device)  # [1, max_text_len]
    cue_features = cue_features.unsqueeze(0).to(device)  # [1, 4]
    
    # Initialize model
    model = CollectionStatusModel().to(device)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    
    # Inference
    with torch.no_grad():
        logits = model(audio.unsqueeze(0), audio_len, text, cue_features)  # [1, 4]
        probabilities = torch.softmax(logits, dim=-1)  # [1, 4]
        prediction = torch.argmax(logits, dim=-1).item()  # Scalar
        confidence = probabilities[0, prediction].item()
    
    # Map prediction to label
    label_map = {v: k for k, v in dataset.label_map.items()}
    predicted_label = label_map[prediction]
    
    # Output results
    print(f"Predicted Collection Status: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print("Probabilities for all classes:")
    for i, label in label_map.items():
        print(f"  {label}: {probabilities[0, i].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer collection status from audio and translation")
    parser.add_argument("audio_path", type=str, help="Path to audio file (.wav)")
    parser.add_argument("translation_path", type=str, help="Path to translation file (.txt)")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/latest.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    infer(args.audio_path, args.translation_path, args.checkpoint)
