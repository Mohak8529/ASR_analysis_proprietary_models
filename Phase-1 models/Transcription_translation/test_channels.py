import pyaudio

p = pyaudio.PyAudio()
print("Available audio input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:  # Only show input devices
        print(f"Device {i}: {dev['name']}")
        print(f"  Max Input Channels: {dev['maxInputChannels']} (Stereo if >= 2, Mono if 1)")
        print(f"  Default Sample Rate: {dev['defaultSampleRate']}")
        print()
p.terminate()