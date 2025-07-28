import torch
import torch.nn as nn
import math
from typing import Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class AudioEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, n_mels: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.proj = nn.Linear(128 * (n_mels // 4), d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max 100 frames
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.cnn(spec.unsqueeze(1))  # [batch, 128, mel//4, time//4]
        x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3)).transpose(1, 2)  # [batch, time, feat]
        x = self.proj(x)  # [batch, time, d_model]
        x = x + self.pos_encoding[:, :x.size(1)]
        for layer in self.transformer:
            x = layer(x)
        return x.mean(dim=1)  # [batch, d_model]

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)  # [batch, seq_len, d_model]
        x = x + self.pos_encoding[:, :x.size(1)]
        for layer in self.transformer:
            x = layer(x)
        return x.mean(dim=1)  # [batch, d_model]

class CollectionScoringModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.audio_encoder = AudioEncoder(
            config['model']['audio_dim'], config['model']['num_heads'],
            config['model']['num_layers'], config['data']['mel_bands']
        )
        self.text_encoder = TextEncoder(
            config['model']['vocab_size'], config['model']['text_dim'],
            config['model']['num_heads'], config['model']['num_layers'],
            config['model']['max_seq_len']
        )
        self.fusion = nn.Linear(config['model']['audio_dim'] + config['model']['text_dim'], 128)
        self.cross_attn = TransformerEncoderLayer(128, config['model']['num_heads'])
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(64, 22),
            nn.Sigmoid()
        )

    def forward(self, spec: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        audio_feat = self.audio_encoder(spec)  # [batch, audio_dim]
        text_feat = self.text_encoder(tokens)  # [batch, text_dim]
        fused = torch.cat([audio_feat, text_feat], dim=-1)  # [batch, audio_dim + text_dim]
        fused = self.fusion(fused).unsqueeze(1)  # [batch, 1, 128]
        fused = self.cross_attn(fused).squeeze(1)  # [batch, 128]
        logits = self.classifier(fused)  # [batch, 22]
        return logits