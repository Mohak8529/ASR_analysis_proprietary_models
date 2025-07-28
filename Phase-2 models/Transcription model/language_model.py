import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_from_disk
import os
from pathlib import Path

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=1024, dropout=0.1),
            n_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc_out(x)

def collate_fn(batch, pad_token_id, max_length):
    """Pad sequences in the batch to the same length."""
    # Get the length of the longest sequence in the batch (up to max_length)
    max_len = min(max(len(item["tokens"]) for item in batch), max_length)
    
    # Pad each sequence to max_len
    padded_tokens = []
    for item in batch:
        tokens = item["tokens"]
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        padded_length = max_len - len(tokens)
        padded_tokens.append(tokens + [pad_token_id] * padded_length)
    
    # Convert to tensor
    return {
        "tokens": torch.tensor(padded_tokens, dtype=torch.long)
    }

def train_lm(dataset, tokenizer, epochs=10, batch_size=32, max_length=2048):
    if len(dataset) == 0:
        print("Error: LM dataset is empty, check bank_call_text.txt.")
        return
    
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        raise ValueError("PAD token not found in tokenizer vocabulary.")
    
    model = LanguageModel(vocab_size).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore PAD tokens in loss
    
    # Ensure checkpoint directory exists
    checkpoint_dir = Path("checkpoints_lm")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoints to resume training
    start_epoch = 0
    checkpoints = list(checkpoint_dir.glob("lm_model_epoch_*.pt"))
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

    # DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id, max_length)
    )
    
    model.train()
    try:
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            for batch in dataloader:
                tokens = batch["tokens"].to("cpu")
                
                # Validate token sequence length (already handled by collate_fn, but keeping for safety)
                batch_valid = True
                for token_seq in tokens:
                    if len(token_seq) > max_length:
                        print(f"Warning: Token sequence length {len(token_seq)} exceeds max_length {max_length}. Skipping batch.")
                        batch_valid = False
                        break
                if not batch_valid:
                    continue
                
                optimizer.zero_grad()
                output = model(tokens[:, :-1])
                loss = criterion(output.reshape(-1, vocab_size), tokens[:, 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_path = checkpoint_dir / f"lm_model_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save final model
        torch.save(model.state_dict(), "lm_model.pt")
        print("Language model saved to lm_model.pt")
    
    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        checkpoint_path = checkpoint_dir / f"lm_model_epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(dataloader) if total_loss > 0 else 0.0
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        exit(0)

def main():
    dataset = load_from_disk("processed_data/lm_dataset")
    tokenizer = Tokenizer.from_file("processed_data/tokenizer/custom_taglish_tokenizer.json")
    train_lm(dataset, tokenizer)

if __name__ == "__main__":
    main()