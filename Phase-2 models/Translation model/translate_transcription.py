import torch
import sentencepiece as spm
import json
import os

# Define project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(project_root, "models", "translation_model.pt")
spm_model = os.path.join(project_root, "models", "spm.model")

# Load model (same as in translate_call.py)
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
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        output = self.transformer(src, tgt)
        output = output.transpose(0, 1)
        return self.fc(output)

# Beam search (same as in translate_call.py)
def beam_search(model, src, sp, beam_size=5, max_len=50):
    device = next(model.parameters()).device
    start_token = sp.piece_to_id("<start>")
    end_token = sp.piece_to_id("<end>")
    src = torch.tensor([sp.encode(src, out_type=int)], dtype=torch.long).to(device)
    
    beams = [(torch.tensor([[start_token]], dtype=torch.long).to(device), 0, [])]
    completed = []
    
    for _ in range(max_len):
        new_beams = []
        for tgt, score, tokens in beams:
            if len(tokens) > 0 and tokens[-1] == end_token:
                completed.append((tgt, score, tokens))
                continue
            output = model(src, tgt)
            probs = torch.softmax(output[:, -1, :], dim=-1)
            top_probs, top_tokens = probs.topk(beam_size)
            for prob, token in zip(top_probs[0], top_tokens[0]):
                new_tgt = torch.cat([tgt, token.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + torch.log(prob).item()
                new_tokens = tokens + [token.item()]
                new_beams.append((new_tgt, new_score, new_tokens))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if len(completed) >= beam_size:
            break
    
    best_beam = sorted(completed + beams, key=lambda x: x[1], reverse=True)[0]
    return sp.decode(best_beam[2])

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TranslationTransformer(vocab_size=32000).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
sp = spm.SentencePieceProcessor(model_file=spm_model)

# Translate JSON dialogue
input_file = os.path.join(project_root, "src", "filipino", "1697712020_3_1.json")
with open(input_file, "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

for item in transcript_data["serializedTranscription"]:
    original_dialogue = item["dialogue"]
    translated_dialogue = beam_search(model, original_dialogue, sp)
    item["dialogue"] = translated_dialogue

# Save output
output_file = os.path.splitext(input_file)[0] + "_translated.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(transcript_data, f, indent=2, ensure_ascii=False)

print(f"Translated transcription saved to {output_file}")