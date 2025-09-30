import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import time

# Configuration
SAMPLE_RATE = 48000  # Hz
MIC1_DEVICE_INDEX = 11  # AB13X USB Audio
MIC2_DEVICE_INDEX = 12  # JBL TUNE 305C USB-C Audio
RECORD_SECONDS = 5  # Duration to record for testing
OUTPUT_DIR = "test_audio"
BLOCKSIZE = 16384

# Ensure output directory exists
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_audio(device_index, device_name, output_file):
    """Record audio from the specified device and save to WAV."""
    print(f"\nTesting {device_name} (Device {device_index})...")
    print(f"Recording for {RECORD_SECONDS} seconds. Please speak into the earphone now.")
    
    try:
        # Query device info to get channel count
        device_info = sd.query_devices(device_index)
        channels = max(1, min(device_info['max_input_channels'], 2))
        print(f"Using {channels} channel(s) for {device_name}")
        
        # Initialize recording array
        recording = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            recording.append(indata.copy())
        
        # Create and start input stream
        with sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            callback=callback
        ):
            # Record for specified duration
            time.sleep(RECORD_SECONDS)
        
        # Convert recorded data to numpy array
        if recording:
            audio = np.concatenate(recording, axis=0)
            # Convert to mono if stereo
            if channels > 1:
                audio = np.mean(audio, axis=1, keepdims=True)
            audio = audio.flatten()
            # Normalize to int16
            audio = (audio / (np.max(np.abs(audio)) + 1e-8) * 32767).astype(np.int16)
            # Save to WAV
            scipy.io.wavfile.write(output_file, SAMPLE_RATE, audio)
            print(f"Saved recording to {output_file}")
        else:
            print(f"No audio recorded for {device_name}")
            
    except Exception as e:
        print(f"Error recording from {device_name}: {e}")

def main():
    # List available devices
    print("Available audio devices:")
    print(sd.query_devices())
    
    # Test Mic 1
    record_audio(MIC1_DEVICE_INDEX, "AB13X USB Audio", os.path.join(OUTPUT_DIR, "test_mic1.wav"))
    
    # Test Mic 2
    record_audio(MIC2_DEVICE_INDEX, "JBL TUNE 305C USB-C Audio", os.path.join(OUTPUT_DIR, "test_mic2.wav"))

if __name__ == "__main__":
    main()