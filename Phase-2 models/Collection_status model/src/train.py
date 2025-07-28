import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import signal
import sys
from dataset import CollectionStatusDataset
from model import CollectionStatusModel

def collate_fn(batch):
    audios, audio_lengths, texts, cue_features, labels = zip(*batch)
    max_audio_len = max(audio_lengths)
    padded_audios = torch.zeros(len(audios), 1, audios[0].size(1), max_audio_len)
    for i, (audio, length) in enumerate(zip(audios, audio_lengths)):
        padded_audios[i, :, :, :length] = audio
    texts = torch.stack(texts)
    cue_features = torch.stack(cue_features)
    labels = torch.tensor(labels, dtype=torch.long)
    audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
    return padded_audios, audio_lengths, texts, cue_features, labels

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')

def train():
    # Config
    data_dir = "../dataset"
    batch_size = 8  # Reduced for memory
    num_epochs = 50
    learning_rate = 1e-4
    checkpoint_dir = "../checkpoints"
    log_file = "../logs/training_log.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Data
    dataset = CollectionStatusDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model and optimizer
    model = CollectionStatusModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, "latest.pth")
    start_epoch, best_loss = load_checkpoint(model, optimizer, latest_checkpoint)

    # Signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\nSaving checkpoint before exit...")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(dataloader)
        }, latest_checkpoint)
        with open(log_file, "a") as f:
            f.write(f"Interrupted at epoch {epoch + 1}, loss: {running_loss / len(dataloader):.4f}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for audio, audio_lengths, text, cue_features, labels in progress_bar:
            audio, audio_lengths, text, cue_features, labels = (
                audio.to(device),
                audio_lengths.to(device),
                text.to(device),
                cue_features.to(device),
                labels.to(device)
            )
            
            optimizer.zero_grad()
            outputs = model(audio, audio_lengths, text, cue_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        
        # Save per-epoch checkpoint
        epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, epoch_checkpoint)
        
        # Update latest checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, latest_checkpoint)
        
        # Log
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}\n")
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train()