import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import sentencepiece as spm
import os
from torchtext.data.metrics import bleu_score

# Define project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, csv_file, spm_model):
        self.data = pd.read_csv(csv_file)
        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.data["target"] = self.data["target"].apply(lambda x: f"<start> {x} <end>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.sp.encode(self.data.iloc[idx]["source"], out_type=int)
        tgt = self.sp.encode(self.data.iloc[idx]["target"], out_type=int)
        return torch.tensor(src), torch.tensor(tgt)

# Transformer Model (inspired by SeamlessM4T)
class TranslationTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dropout=dropout
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src = src.transpose(0, 1)  # [seq_len, batch, d_model]
        tgt = tgt.transpose(0, 1)
        output = self.transformer(src, tgt)
        output = output.transpose(0, 1)  # [batch, seq_len, d_model]
        return self.fc(output)

# Load data
train_dataset = TranslationDataset(os.path.join(project_root, "Dataset", "processed", "train.csv"), os.path.join(project_root, "models", "spm.model"))
val_dataset = TranslationDataset(os.path.join(project_root, "Dataset", "processed", "val.csv"), os.path.join(project_root, "models", "spm.model"))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: pad_collate(x))
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=lambda x: pad_collate(x))

def pad_collate(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, padding_value=0, batch_first=True)
    tgt = nn.utils.rnn.pad_sequence(tgt, padding_value=0, batch_first=True)
    return src, tgt

# Model parameters
vocab_size = 32000
d_model = 512
nhead = 8
num_layers = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = TranslationTransformer(vocab_size, d_model, nhead, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
scaler = GradScaler()

# Load existing weights if available
model_path = os.path.join(project_root, "models", "translation_model.pt")
initial_epoch = 0
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    initial_epoch = checkpoint["epoch"]
    print(f"Loaded model from epoch {initial_epoch}")

# Training loop
epochs = 50
sp = spm.SentencePieceProcessor(model_file=os.path.join(project_root, "models", "spm.model"))
for epoch in range(initial_epoch, epochs):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # Validation BLEU
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            preds = torch.argmax(output, dim=-1)
            val_preds.extend([sp.decode(pred.tolist()) for pred in preds])
            val_targets.extend([sp.decode(t.tolist()) for t in tgt[:, 1:]])
    bleu = bleu_score(val_preds, [[t] for t in val_targets])
    print(f"Validation BLEU: {bleu:.4f}")
    
    # Save checkpoint
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, model_path)
    
    # Early stopping
    if epoch > 5 and bleu < 0.01:
        print("Early stopping due to low BLEU")
        break