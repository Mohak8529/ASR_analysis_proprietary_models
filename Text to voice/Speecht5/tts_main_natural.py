import os
import torch
import soundfile as sf
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

MODEL_DIR = "speecht5_model"
SPEAKER_EMB_FILE = "speaker.npy"
TEXT = "Hello, this is a test using SpeechT5 with speaker embedding."
OUTPUT_WAV = "output.wav"

def load_model(model_dir=MODEL_DIR):
    proc = SpeechT5Processor.from_pretrained(model_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_dir)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return proc, model, vocoder

def load_speaker_embedding(path=SPEAKER_EMB_FILE):
    emb = np.load(path)
    tensor = torch.tensor(emb, dtype=torch.float32)
    
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2 and tensor.size(0) == 1:
        pass
    else:
        tensor = tensor.view(1, -1)
    
    print(f"Speaker embedding shape: {tensor.shape}")
    return tensor

def preprocess_text(text):
    """
    Preprocess text to make it more natural for TTS.
    Add pauses and normalize punctuation.
    """
    # Add natural pauses
    text = text.replace(',', ', ')  # Add space after commas
    text = text.replace('.', '. ')  # Add space after periods
    text = text.replace('!', '! ')  # Add space after exclamations
    text = text.replace('?', '? ')  # Add space after questions
    
    # Remove multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def text_to_speech_improved(proc, model, vocoder, text, speaker_emb, output_wav, 
                          temperature=0.7, do_sample=True, max_length=4000):
    """
    Improved TTS with better generation parameters
    """
    # Preprocess text
    text = preprocess_text(text)
    print(f"Processed text: '{text}'")
    
    inputs = proc(text=text, return_tensors="pt")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Speaker embedding shape: {speaker_emb.shape}")
    
    # Set model to eval mode for better quality
    model.eval()
    
    with torch.no_grad():
        # Use improved generation parameters
        speech = model.generate_speech(
            inputs["input_ids"], 
            speaker_emb, 
            vocoder=vocoder,
            # Add these parameters for better quality
            threshold=0.5,        # Threshold for stop token
            minlenratio=0.0,      # Minimum length ratio
            maxlenratio=20.0,     # Maximum length ratio
        )
    
    # Post-process the audio
    wav = speech.cpu().numpy().squeeze()
    
    # Apply some audio processing to reduce roboticness
    wav = apply_audio_enhancement(wav)
    
    # Save with higher quality
    sf.write(output_wav, wav, samplerate=16000, subtype='PCM_16')
    print(f"Saved enhanced speech to '{output_wav}'")
    print(f"Audio duration: {len(wav)/16000:.2f} seconds")

def apply_audio_enhancement(wav, sample_rate=16000):
    """
    Apply basic audio enhancement to reduce roboticness
    """
    # Normalize audio
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav)) * 0.95
    
    # Apply subtle smoothing to reduce harsh transitions
    from scipy import signal
    # Very light low-pass filter to smooth harsh edges
    nyquist = sample_rate / 2
    cutoff = 7500  # Cut frequencies above 7.5kHz
    b, a = signal.butter(2, cutoff / nyquist, btype='low')
    wav = signal.filtfilt(b, a, wav)
    
    return wav

def create_better_speaker_embedding(reference_audio_path, output_path="speaker_enhanced.npy"):
    """
    Create a potentially better speaker embedding with some preprocessing
    """
    import torchaudio
    from speechbrain.inference import EncoderClassifier
    
    EMBED_MODEL = "speechbrain/spkrec-xvect-voxceleb"
    CACHE_DIR = os.path.join("speecht5_model", "spkrec-xvect-voxceleb")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(
        source=EMBED_MODEL,
        savedir=CACHE_DIR,
        run_opts={"device": device}
    )
    
    # Load and preprocess audio
    signal, sr = torchaudio.load(reference_audio_path)
    
    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        signal = resampler(signal)
    
    # Normalize audio
    signal = signal / torch.max(torch.abs(signal))
    
    # Extract embedding
    embedding = classifier.encode_batch(signal)
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    embedding = embedding.squeeze(1).cpu().numpy()
    
    # Save enhanced embedding
    np.save(output_path, embedding, allow_pickle=False)
    print(f"Created enhanced speaker embedding: '{output_path}'")
    
    return output_path

def main():
    if not os.path.isdir(MODEL_DIR):
        print("Run download_model.py first.")
        return
    if not os.path.isfile(SPEAKER_EMB_FILE):
        print(f"Run speaker_embed.py to create '{SPEAKER_EMB_FILE}' before using this.")
        return
        
    proc, model, vocoder = load_model()
    speaker_emb = load_speaker_embedding()
    
    # Try with improved settings
    text_to_speech_improved(proc, model, vocoder, TEXT, speaker_emb, OUTPUT_WAV)
    
    # Optional: Create an enhanced version with better speaker embedding
    if os.path.exists("reference_audio_male.wav"):
        print("\nCreating enhanced speaker embedding...")
        enhanced_emb_path = create_better_speaker_embedding("reference_audio_male.wav")
        enhanced_speaker_emb = load_speaker_embedding(enhanced_emb_path)
        
        output_enhanced = "output_enhanced.wav"
        text_to_speech_improved(proc, model, vocoder, TEXT, enhanced_speaker_emb, output_enhanced)
        print(f"Also created enhanced version: '{output_enhanced}'")

if __name__ == "__main__":
    # Install scipy if not already installed
    try:
        from scipy import signal
    except ImportError:
        print("Installing scipy for audio enhancement...")
        os.system("pip install scipy")
        from scipy import signal
    
    main()