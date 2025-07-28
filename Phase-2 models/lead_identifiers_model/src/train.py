import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import signal
import sys
from dataset import LeadIdentifiersDataset
from model import LeadIdentifiersModel

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
    data_dir = "../dataset"
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.0001
    checkpoint_dir = "../checkpoints"
    log_file = "../logs/training_log.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    dataset = LeadIdentifiersDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model = LeadIdentifiersModel(vocab_size=len(dataset.vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    identifier_criterion = nn.CrossEntropyLoss(ignore_index=0)

    latest_checkpoint = os.path.join(checkpoint_dir, "latest.pth")
    start_epoch, best_loss = load_checkpoint(model, optimizer, latest_checkpoint)

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

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for audio, audio_len, trans, transcr, trans_len, transcr_len, id_labels in progress_bar:
            audio, audio_len, trans, transcr, trans_len, transcr_len, id_labels = (
                audio.to(device),
                audio_len.to(device),
                trans.to(device),
                transcr.to(device),
                trans_len.to(device),
                transcr_len.to(device),
                id_labels.to(device)
            )

            optimizer.zero_grad()
            identifier_logits = model(audio, audio_len, trans, transcr, trans_len)
            identifier_loss = identifier_criterion(identifier_logits.view(-1, 3), id_labels.view(-1))
            identifier_loss.backward()
            optimizer.step()
            running_loss += identifier_loss.item()
            progress_bar.set_postfix(loss=identifier_loss.item())

        epoch_loss = running_loss / len(dataloader)
        epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, epoch_checkpoint)
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, latest_checkpoint)
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}\n")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train() 