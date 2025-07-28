import torch
import torchaudio
import numpy as np
from tokenizers import Tokenizer
import json
import os
import argparse
from preprocess import process_audio
from stt_model import STTModel
from language_model import LanguageModel

def ctc_decode(logits, tokenizer, beam_size=5):
    def beam_search(logits, beam_size):
        beams = [((), 0)]
        for t in range(logits.shape[0]):
            new_beams = []
            for prefix, score in beams:
                top_k = torch.topk(logits[t], beam_size)[1]
                for idx in top_k:
                    new_prefix = prefix + (idx.item(),)
                    new_score = score + logits[t, idx].item()
                    new_beams.append((new_prefix, new_score))
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
        
        best_seq = beams[0][0]
        decoded = []
        prev = None
        for idx in best_seq:
            if idx != 0 and idx != prev:
                decoded.append(idx)
            prev = idx
        return decoded

    logits = torch.softmax(logits, dim=-1).numpy()
    decoded = beam_search(logits, beam_size)
    text = tokenizer.decode(decoded)
    
    # Energy-based segmentation
    energy = np.sum(logits, axis=1)
    pause_threshold = np.percentile(energy, 10)
    segments = []
    start = 0
    for t in range(1, len(energy)):
        if energy[t] < pause_threshold and energy[t-1] >= pause_threshold:
            segment_text = tokenizer.decode(beam_search(logits[start:t], beam_size))
            if segment_text.strip():
                segments.append({
                    "text": segment_text,
                    "start": start * 0.01,
                    "end": t * 0.01
                })
            start = t
    
    if start < len(energy):
        segment_text = tokenizer.decode(beam_search(logits[start:], beam_size))
        if segment_text.strip():
            segments.append({
                "text": segment_text,
                "start": start * 0.01,
                "end": len(energy) * 0.01
            })
    
    return {"segments": segments}

def transcribe(audio_path, stt_model_path="stt_model.pt", lm_model_path="lm_model.pt"):
    tokenizer = Tokenizer.from_file("processed_data/tokenizer/custom_taglish_tokenizer.json")
    
    stt_model = STTModel(tokenizer.get_vocab_size())
    stt_model.load_state_dict(torch.load(stt_model_path, map_location="cpu"))
    stt_model.eval()
    
    lm_model = LanguageModel(tokenizer.get_vocab_size())
    lm_model.load_state_dict(torch.load(lm_model_path, map_location="cpu"))
    lm_model.eval()
    
    log_mel = process_audio(audio_path)
    log_mel = torch.tensor(log_mel).unsqueeze(0).float().to("cpu")
    
    with torch.no_grad():
        logits = stt_model(log_mel).squeeze(0)
    
    result = ctc_decode(logits, tokenizer)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Transcribe Taglish audio.")
    parser.add_argument("--audio_path", default="Dataset/Audio/abc.wav", help="Path to input audio file")
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    result = transcribe(args.audio_path)
    
    output_path = "output/output.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Transcription saved to {output_path}")
    for segment in result["segments"]:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}]: {segment['text']}")

if __name__ == "__main__":
    main()