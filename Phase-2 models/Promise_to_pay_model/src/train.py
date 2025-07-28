import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import signal
import sys
from dataset import PromiseToPayDataset
from model import PromiseToPayModel

def collate_fn(batch):
    audios, audio_lengths, trans_tokens, transcr_tokens, trans_lengths, transcr_lengths, labels, keyword_labels = zip(*batch)
    # Pad audio to longest in batch
    max_audio_len = max(audio_lengths)
    padded_audios = torch.zeros(len(audios), 1, audios[0].size(1), max_audio_len)
    for i, audio in enumerate(audios):
        padded_audios[i, :, :, :audio.size(-1)] = audio
    # Pad text and keyword labels to longest in batch
    max_text_len = max(max(trans_lengths), max(transcr_lengths))
    padded_trans = torch.zeros(len(trans_tokens), max_text_len, dtype=torch.long)
    padded_transcr = torch.zeros(len(transcr_tokens), max_text_len, dtype=torch.long)
    padded_keywords = torch.zeros(len(keyword_labels), max_text_len, dtype=torch.long)
    for i in range(len(trans_tokens)):
        padded_trans[i, :len(trans_tokens[i])] = trans_tokens[i]
        padded_transcr[i, :len(transcr_tokens[i])] = transcr_tokens[i]
        padded_keywords[i, :len(keyword_labels[i])] = keyword_labels[i]
    return (
        padded_audios,
        torch.tensor(audio_lengths),
        padded_trans,
        padded_transcr,
        torch.tensor(trans_lengths),
        torch.tensor(transcr_lengths),
        torch.tensor(labels, dtype=torch.long),
        padded_keywords
    )

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
    # Settings
    data_dir = "../dataset"
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.0001
    checkpoint_dir = "../checkpoints"
    log_file = "../logs/training_log.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make folders for checkpoints and logs
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Load data
    dataset = PromiseToPayDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Set up model and optimizer
    model = PromiseToPayModel(vocab_size=len(dataset.vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    class_criterion = nn.CrossEntropyLoss()
    keyword_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    # Load latest checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, "latest.pth")
    start_epoch, best_loss = load_checkpoint(model, optimizer, latest_checkpoint)

    # Handle Ctrl+C to save progress
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

        for audio, audio_len, trans, transcr, trans_len, transcr_len, labels, kw_labels in progress_bar:
            # Move data to GPU/CPU
            audio, audio_len, trans, transcr, trans_len, transcr_len, labels, kw_labels = (
                audio.to(device),
                audio_len.to(device),
                trans.to(device),
                transcr.to(device),
                trans_len.to(device),
                transcr_len.to(device),
                labels.to(device),
                kw_labels.to(device)
            )

            # Reset model
            optimizer.zero_grad()
            # Make predictions
            class_logits, keyword_logits = model(audio, audio_len, trans, transcr, trans_len)
            # Check errors
            class_loss = class_criterion(class_logits, labels)
            keyword_loss = keyword_criterion(keyword_logits.view(-1, 3), kw_labels.view(-1))
            loss = class_loss + keyword_loss
            # Learn from errors
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Save checkpoint for this epoch
        epoch_loss = running_loss / len(dataloader)
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

        # Write to log file
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}\n")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train()

