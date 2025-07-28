import pyaudio
import numpy as np
import scipy.io.wavfile
import time

# Microphone settings
MIC_A_INDEX = 11  # Your USB-C mic (Device 11: AB13X USB Audio)
MIC_B_INDEX = 12  # Friend’s USB-C mic (Device 12: JBL TUNE 305C USB-C)
MIC_A_CHANNELS = 1  # Mono
MIC_B_CHANNELS = 1  # Mono
SAMPLE_RATE = 48000  # Native rate
CHUNK_DURATION = 5
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

p = pyaudio.PyAudio()

try:
    stream_a = p.open(format=pyaudio.paInt16,
                      channels=MIC_A_CHANNELS,
                      rate=SAMPLE_RATE,
                      input=True,
                      input_device_index=MIC_A_INDEX,
                      frames_per_buffer=SAMPLES_PER_CHUNK)

    stream_b = p.open(format=pyaudio.paInt16,
                      channels=MIC_B_CHANNELS,
                      rate=SAMPLE_RATE,
                      input=True,
                      input_device_index=MIC_B_INDEX,
                      frames_per_buffer=SAMPLES_PER_CHUNK)

    print("Recording for 5 seconds... Speak into both USB-C microphones.")
    data_a = stream_a.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)
    data_b = stream_b.read(SAMPLES_PER_CHUNK, exception_on_overflow=False)

    audio_a = np.frombuffer(data_a, dtype=np.int16)
    audio_b = np.frombuffer(data_b, dtype=np.int16)

    scipy.io.wavfile.write("test_my_usb_c_mic.wav", SAMPLE_RATE, audio_a.reshape(-1, MIC_A_CHANNELS))
    scipy.io.wavfile.write("test_friend_usb_c_mic.wav", SAMPLE_RATE, audio_b.reshape(-1, MIC_B_CHANNELS))

    print("Saved recordings: test_my_usb_c_mic.wav, test_friend_usb_c_mic.wav")

except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        stream_a.stop_stream()
        stream_a.close()
        stream_b.stop_stream()
        stream_b.close()
    except:
        pass
    p.terminate()