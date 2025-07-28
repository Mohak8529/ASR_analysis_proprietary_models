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

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        output = query + self.dropout(attn_output)
        return self.norm(output)

class LeadIdentifiersModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, nhead=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        # Audio branch
        self.audio_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.audio_flatten = nn.Linear(128 * 32, d_model)
        self.audio_pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        self.audio_transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        # Translation branch
        self.trans_embedding = nn.Embedding(vocab_size, d_model)
        self.trans_pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))
        self.trans_transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        # Transcription branch
        self.transcr_embedding = nn.Embedding(vocab_size, d_model)
        self.transcr_pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))
        self.transcr_transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        # Cross-modal attention
        self.audio_to_trans = CrossModalAttention(d_model, nhead, dropout)
        self.trans_to_audio = CrossModalAttention(d_model, nhead, dropout)
        self.transcr_to_trans = CrossModalAttention(d_model, nhead, dropout)
        # Identifier classification
        self.identifier_classifier = nn.Linear(d_model, 3)  # B-KEY, I-KEY, O
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio, audio_lengths, trans_tokens, transcr_tokens, trans_len):
        batch_size = audio.size(0)
        # Audio processing
        audio = self.audio_cnn(audio)
        audio = audio.permute(0, 2, 3, 1).reshape(batch_size, -1, 128 * 32)
        audio = self.audio_flatten(audio)
        audio = audio + self.audio_pos_encoder[:, :audio.size(1), :]
        audio_mask = torch.zeros(batch_size, audio.size(1), device=audio.device).bool()
        for i, length in enumerate(audio_lengths):
            audio_mask[i, length:] = True
        for layer in self.audio_transformer:
            audio = layer(audio.transpose(0, 1), src_key_padding_mask=audio_mask).transpose(0, 1)
        audio_repr = audio.mean(dim=1)
        # Translation processing
        trans = self.trans_embedding(trans_tokens) + self.trans_pos_encoder
        trans_mask = (trans_tokens == 0).to(trans.device)
        for layer in self.trans_transformer:
            trans = layer(trans.transpose(0, 1), src_key_padding_mask=trans_mask).transpose(0, 1)
        trans_repr = trans.mean(dim=1)
        # Transcription processing
        transcr = self.transcr_embedding(transcr_tokens) + self.transcr_pos_encoder
        transcr_mask = (transcr_tokens == 0).to(transcr.device)
        for layer in self.transcr_transformer:
            transcr = layer(transcr.transpose(0, 1), src_key_padding_mask=transcr_mask).transpose(0, 1)
        transcr_repr = transcr.mean(dim=1)
        # Cross-modal attention
        audio_fused = self.audio_to_trans(audio_repr.unsqueeze(0), trans, trans, key_padding_mask=trans_mask)
        trans_fused = self.trans_to_audio(trans_repr.unsqueeze(0), audio, audio, key_padding_mask=audio_mask)
        transcr_fused = self.transcr_to_trans(transcr_repr.unsqueeze(0), trans, trans, key_padding_mask=trans_mask)
        audio_fused = audio_fused.squeeze(0)
        trans_fused = trans_fused.squeeze(0)
        # Combine for translation enhancement
        fused = torch.cat([audio_fused, trans_fused, transcr_fused.squeeze(0)], dim=-1)
        fused = self.dropout(fused)
        # Identifier classification
        identifier_logits = self.identifier_classifier(trans)
        return identifier_logits