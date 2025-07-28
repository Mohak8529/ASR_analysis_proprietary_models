import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from src.data import get_dataloaders
from src.model import CollectionScoringModel
from src.utils import load_config
import os
from pathlib import Path

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: str):
    """Save model, optimizer, epoch, and loss to a checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str, device: torch.device) -> Tuple[int, float]:
    """Load checkpoint and return starting epoch and loss."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def train():
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, val_loader, vocab = get_dataloaders(
        config['data']['dataset_path'], config, config['training']['batch_size']
    )

    # Model and optimizer
    model = CollectionScoringModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = nn.BCELoss()

    # Checkpointing setup
    os.makedirs('checkpoints', exist_ok=True)
    latest_checkpoint_path = 'checkpoints/latest_checkpoint.pth'
    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from latest checkpoint if exists
    if Path(latest_checkpoint_path).exists():
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, latest_checkpoint_path, device)
        print(f"Resuming training from epoch {start_epoch + 1}, best validation loss: {best_val_loss:.4f}")

    # Training loop
    try:
        for epoch in range(start_epoch, config['training']['epochs']):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                spec, tokens, labels, zcr, energy = batch
                spec, tokens, labels = spec.to(device), tokens.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(spec, tokens)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    spec, tokens, labels, zcr, energy = batch
                    spec, tokens, labels = spec.to(device), tokens.to(device), labels.to(device)
                    logits = model(spec, tokens)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            # Save per-epoch checkpoint
            epoch_checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1:03d}.pth'
            save_checkpoint(model, optimizer, epoch + 1, val_loss, epoch_checkpoint_path)

            # Save latest checkpoint
            save_checkpoint(model, optimizer, epoch + 1, val_loss, latest_checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    except KeyboardInterrupt:
        print("Training interrupted. Saving latest checkpoint...")
        save_checkpoint(model, optimizer, epoch + 1, val_loss, latest_checkpoint_path)
        print(f"Checkpoint saved to {latest_checkpoint_path}")
        exit(0)

if __name__ == '__main__':
    train()