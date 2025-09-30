import numpy as np
from config import SAMPLE_RATE, BUFFER_MAX_SIZE

class BufferManager:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)

    def add(self, samples):
        self.buffer = np.concatenate([self.buffer, samples])
        max_samples = BUFFER_MAX_SIZE * SAMPLE_RATE
        if len(self.buffer) > max_samples:
            self.buffer = self.buffer[-max_samples:]

    def get_all(self):
        return self.buffer.copy()

    def clear(self):
        self.buffer = np.array([], dtype=np.float32)
