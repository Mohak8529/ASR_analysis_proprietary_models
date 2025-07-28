import torch
from src.model import CollectionScoringModel
from src.utils import load_config, load_audio, compute_mel_spectrogram, tokenize_text, build_vocabulary
from src.data import CollectionScoringDataset
import json
from pathlib import Path

def infer(audio_path: str, transcription_path: str, translation_path: str):
    config = load_config('config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocabulary
    dataset = CollectionScoringDataset(config['data']['dataset_path'], {}, config)
    vocab = build_vocabulary(dataset.transcription_files + dataset.translation_files, config['model']['vocab_size'])

    # Load model
    model = CollectionScoringModel(config).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model.eval()

    # Process inputs
    waveform = load_audio(audio_path, config['data']['audio_sample_rate'])
    spec = compute_mel_spectrogram(waveform, config['data']['audio_sample_rate'], config['data']['mel_bands'])
    with open(transcription_path, 'r') as f:
        transcription = f.read()
    with open(translation_path, 'r') as f:
        translation = f.read()
    text = f"{transcription} [SEP] {translation} [SEP] Check criteria"
    tokens = tokenize_text(text, vocab, config['model']['max_seq_len'])

    # Inference
    with torch.no_grad():
        spec = spec.unsqueeze(0).to(device)  # [1, mel, time]
        tokens = tokens.unsqueeze(0).to(device)  # [1, seq_len]
        logits = model(spec, tokens)  # [1, 22]
        preds = (logits > 0.5).float().cpu().numpy()[0]

    # Create JSON output
    criteria = [
        'call_closed_properly', 'call_open_timely_manner', 'standard_opening_spiel',
        'verification_of_account_security', 'friendly_confident_tone', 'attentive_listening',
        'call_control_efficiency', 'follow_policies_procedure', 'service_reminder',
        'customer_alternate_number', 'call_record_clause', 'pid_process', 'udcp_process',
        'call_avoidance', 'misleading_information', 'data_manipulation', 'call_recap',
        'ask_additional_number', 'probing_questions_effectiveness', 'payment_resolution_actions',
        'payment_delay_consequences', 'properly_document_the_call'
    ]
    output = {crit: bool(pred) for crit, pred in zip(criteria, preds)}
    
    with open('prediction.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("Prediction saved to prediction.json")
    return output

if __name__ == '__main__':
    # Example usage
    infer(
        'Dataset/audio/ac.wav',
        'Dataset/transcription/ac_transcription.txt',
        'Dataset/translation/ac_translation.txt'
    )