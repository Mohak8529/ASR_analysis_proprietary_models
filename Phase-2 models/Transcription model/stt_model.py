import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tokenizers import Tokenizer
import os
from pathlib import Path

class STTModel(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=1024, dropout=0.1),
                n_layers
            )
        )
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        return self.decoder(x)

def train_stt(dataset, tokenizer, epochs=10, batch_size=2, max_length=2048):
    vocab_size = tokenizer.get_vocab_size()
    model = STTModel(vocab_size).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=0)
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path("checkpoints_stt")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoints to resume training
    start_epoch = 0
    checkpoints = list(checkpoint_dir.glob("stt_model_epoch_*.pt"))
    if checkpoints:
        latest_checkpoint = max(
            checkpoints,
            key=lambda x: int(x.stem.split("_")[-1])
        )
        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # DataLoader for training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    try:
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            for batch in dataloader:
                log_mel = batch["log_mel"].to("cpu").float()
                tokens = batch["tokens"].to("cpu")
                
                # Validate token sequence length
                batch_valid = True
                for token_seq in tokens:
                    if len(token_seq) > max_length:
                        print(f"Warning: Token sequence length {len(token_seq)} exceeds max_length {max_length}. Skipping batch.")
                        batch_valid = False
                        break
                if not batch_valid:
                    continue
                
                optimizer.zero_grad()
                output = model(log_mel)
                output = output.log_softmax(2).transpose(0, 1)
                
                input_lengths = torch.full((log_mel.size(0),), log_mel.size(1), dtype=torch.long)
                target_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
                loss = criterion(output, tokens, input_lengths, target_lengths)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_path = checkpoint_dir / f"stt_model_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save final model
        torch.save(model.state_dict(), "stt_model.pt")
        print("STT model saved to stt_model.pt")
    
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        checkpoint_path = checkpoint_dir / f"stt_model_epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(dataloader) if total_loss > 0 else 0.0
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        exit(0)

def main():
    dataset = load_from_disk("processed_data/stt_dataset")
    tokenizer = Tokenizer.from_file("processed_data/tokenizer/custom_taglish_tokenizer.json")
    train_stt(dataset, tokenizer)

if __name__ == "__main__":
    main()