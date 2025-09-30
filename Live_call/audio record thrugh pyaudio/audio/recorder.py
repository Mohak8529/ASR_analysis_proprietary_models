import pyaudio
import threading
import queue
from config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE

class AudioRecorder:
    def __init__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.frames_queue = queue.Queue()  # queue of raw byte chunks for transcription
        self.all_raw_frames = []  # full session raw audio storage
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        self.thread = threading.Thread(target=self.record_loop, daemon=True)
        self.thread.start()

    def record_loop(self):
        while self.running:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.frames_queue.put(data)  # enqueue for transcription
                self.all_raw_frames.append(data)  # store entire session's raw audio
            except Exception as e:
                print(f"[Recorder Error] {e}")
                break

    def read(self):
        try:
            return self.frames_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()  # Wait for the recording thread to finish
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio_interface is not None:
            self.audio_interface.terminate()

    def get_all_recorded_frames(self):
        return b''.join(self.all_raw_frames)
