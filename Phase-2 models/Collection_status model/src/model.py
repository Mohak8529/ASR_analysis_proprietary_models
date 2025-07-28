import torch
import torch.nn as nn
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class CollectionStatusModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4, dropout=0.3):
        super().__init__()
        # Audio branch: CNN + Transformer
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.audio_flatten = nn.Linear(128 * 32, d_model)  # Adjusted for variable time
        self.audio_pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 time steps
        self.audio_transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Text branch: Embedding + Transformer
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.text_pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))
        self.text_transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

        # Cue features
        self.cue_linear = nn.Linear(4, d_model)

        # Fusion and classification
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.classifier = nn.Linear(d_model, 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, audio_lengths, text, cue_features):
        # Audio processing
        batch_size = audio.size(0)
        audio = self.audio_cnn(audio)  # [batch, 128, h, w]
        audio = audio.permute(0, 2, 3, 1)  # [batch, h, w, 128]
        audio = audio.reshape(batch_size, -1, 128 * 32)  # [batch, h*w, 128*32]
        audio = self.audio_flatten(audio)  # [batch, h*w, d_model]

        # Positional encoding and masking
        max_len = audio.size(1)
        audio = audio + self.audio_pos_encoder[:, :max_len, :]
        key_padding_mask = torch.zeros(batch_size, max_len, device=audio.device).bool()
        for i, length in enumerate(audio_lengths):
            key_padding_mask[i, length:] = True  # Mask padded regions
        for layer in self.audio_transformer:
            audio = layer(audio.transpose(0, 1), src_key_padding_mask=key_padding_mask).transpose(0, 1)
        audio = audio.mean(dim=1)  # Mean pooling: [batch, d_model]

        # Text processing
        text = self.text_embedding(text) + self.text_pos_encoder
        for layer in self.text_transformer:
            text = layer(text.transpose(0, 1)).transpose(0, 1)
        text = text.mean(dim=1)  # [batch, d_model]

        # Cue features
        cues = self.cue_linear(cue_features)  # [batch, d_model]

        # Fusion
        fused = torch.cat([audio, text, cues], dim=-1)  # [batch, d_model * 3]
        fused = self.fusion(fused)  # [batch, d_model]
        fused = self.dropout(fused)
        logits = self.classifier(fused)  # [batch, 4]
        return logits