import asyncio
import websockets
import pyaudio
import signal
import sys

from .config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, WS_SERVER_URI

running = True

async def send_audio():
    global running
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    try:
        async with websockets.connect(WS_SERVER_URI) as websocket:
            print(f"[Client] Connected to {WS_SERVER_URI}")
            while running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                await websocket.send(data)
    except Exception as e:
        print(f"[Client] Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def signal_handler(sig, frame):
    global running
    print("\n[Client] Stopping...")
    running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(send_audio())
    print("[Client] Done.")
