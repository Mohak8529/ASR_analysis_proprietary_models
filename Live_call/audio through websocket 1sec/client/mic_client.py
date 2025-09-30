import asyncio
import websockets
import pyaudio
import signal
import sys

from .config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, WS_SERVER_URI

running = True
chunk_counter = 0

async def send_audio():
    global running, chunk_counter
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    try:
        async with websockets.connect(WS_SERVER_URI) as ws:
            print(f"[Client] Connected to {WS_SERVER_URI}")
            while running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk_counter += 1
                await ws.send(data)
                # Log each chunk sent
                print(f"[Client] Sent chunk #{chunk_counter} – {len(data)} bytes")
    except Exception as e:
        print(f"[Client] Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        # Final summary
        print(f"[Client] Total chunks sent: {chunk_counter}")

def signal_handler(sig, frame):
    global running
    print("\n[Client] Stopping...")
    running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(send_audio())
    print("[Client] Done.")
