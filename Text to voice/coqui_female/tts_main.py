import os
import re
from TTS.api import TTS
from pydub import AudioSegment

# Set TTS_HOME to the storage location of the downloaded model
tts_home = "/mnt/ssd1/coqui_female/model"
os.environ["TTS_HOME"] = tts_home

# Import the TTS class
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=True, gpu=False)

# Define the input paragraph as a single continuous string
input_paragraph = "Markets opened flat today amid global uncertainty, with investors closely watching US Fed rate cues and crude oil prices. The Nifty hovered near 22,000 while the Sensex remained range-bound. The rupee weakened slightly against the dollar, trading at 83.10, influenced by foreign fund outflows and rising US yields. India’s manufacturing PMI rose to 57.9 in July, signaling sustained expansion, supported by strong domestic demand, robust output growth, and rising new orders from both local and international markets. Adani Group stocks saw a mild rebound after recent declines, as investor confidence steadied following SEBI’s update on the ongoing probe. Gold prices held steady at ₹61,200 per 10 grams in Delhi as traders awaited the US payroll data for market cues. Crypto markets surged overnight, with Bitcoin crossing $64,000 on optimism around a potential spot ETF approval. Government bond yields edged higher in response to fresh supply expectations and hawkish commentary from the RBI. LIC shares rose 2.3% after reports of increased stake in PSU banks. Overall market sentiment remains cautious ahead of key global earnings reports and crucial inflation data expected next week."

# Choose an output file path
output_path = "female_output.wav"

# Remove existing output file to prevent overwriting issues
if os.path.exists(output_path):
    try:
        os.remove(output_path)
        print(f"Removed existing output file: {output_path}")
    except Exception as e:
        print(f"Error removing existing output file {output_path}: {e}")

# Split paragraph into sentences, avoiding decimal splits
def split_sentences(paragraph):
    # Use regex to split at periods, but exclude decimals (e.g., 83.10)
    sentence_end = re.compile(r'(?<![0-9])\.(?=\s|$|[A-Z])')
    sentences = [s.strip() for s in sentence_end.split(paragraph) if s.strip()]
    return sentences

sentences = split_sentences(input_paragraph)
print(f"Split into {len(sentences)} sentences:")
for i, s in enumerate(sentences, 1):
    print(f"Sentence {i}: {s}")

# Synthesize each sentence and save to individual WAV files
temp_files = []
for i, sentence in enumerate(sentences):
    print(f"Processing sentence {i+1}/{len(sentences)}: {sentence}")
    
    # Save to a persistent WAV file
    temp_path = f"sentence_{i+1}.wav"
    try:
        # Synthesize the sentence
        tts.tts_to_file(text=sentence, file_path=temp_path)
        
        # Load and verify the audio
        temp_audio = AudioSegment.from_wav(temp_path)
        duration_ms = len(temp_audio)
        sample_rate = temp_audio.frame_rate
        channels = temp_audio.channels
        print(f"Sentence {i+1} audio duration: {duration_ms/1000:.2f} seconds, "
              f"sample rate: {sample_rate} Hz, channels: {channels}")
        
        # Store the file path for concatenation
        temp_files.append(temp_path)
    except Exception as e:
        print(f"Error processing sentence {i+1}: {e}")
        continue

# Concatenate all sentence WAV files
try:
    if not temp_files:
        raise ValueError("No audio files were generated for concatenation.")
    
    # Initialize combined audio with the first file
    combined_audio = AudioSegment.from_wav(temp_files[0])
    print(f"Added {temp_files[0]} to combined audio, duration: {len(combined_audio)/1000:.2f} seconds")
    
    # Append remaining audio files with a 0.5s pause between sentences
    for temp_path in temp_files[1:]:
        next_audio = AudioSegment.from_wav(temp_path)
        # Add a 500ms pause
        combined_audio += AudioSegment.silent(duration=500)
        combined_audio += next_audio
        print(f"Added {temp_path} to combined audio, current duration: {len(combined_audio)/1000:.2f} seconds")
    
    # Verify combined audio duration
    combined_duration_ms = len(combined_audio)
    print(f"Combined audio duration before export: {combined_duration_ms/1000:.2f} seconds")
    
    # Export the combined audio with error handling
    try:
        combined_audio.export(output_path, format="wav")
    except Exception as e:
        print(f"Error exporting to {output_path}: {e}")
        raise
    
    # Verify the final output file
    final_audio = AudioSegment.from_wav(output_path)
    final_duration_ms = len(final_audio)
    print(f"Final output audio duration: {final_duration_ms/1000:.2f} seconds")
    
    print(f"Coqui TTS: Speech saved as '{output_path}'.")
except Exception as e:
    print(f"Error during concatenation or export: {e}")

# Clean up temporary files
try:
    for temp_file in temp_files:
        os.remove(temp_file)
        print(f"Removed temporary file: {temp_file}")
except Exception as e:
    print(f"Error cleaning up temporary files: {e}")