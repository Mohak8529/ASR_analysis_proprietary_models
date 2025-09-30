import os
import sys
import signal
import asyncio
import websockets
import pyaudio
from .config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE, WS_SERVER_URI

running = True
chunk_counter = 0

def signal_handler(sig, frame):
    global running
    print("\n[Client] Stopping...")
    running = False

async def send_audio():
    global running, chunk_counter
    # Read channel from environment
    channel = os.getenv("CHANNEL", "").upper()
    if channel not in ("A", "B"):
        print("Error: Set CHANNEL=A or CHANNEL=B")
        sys.exit(1)

    # Build URI with channel query
    uri = f"{WS_SERVER_URI}?channel={channel}"
    print(f"[Client {channel}] Connecting to {uri}")

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    try:
        async with websockets.connect(uri) as ws:
            print(f"[Client {channel}] Connected")
            while running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                chunk_counter += 1
                # Prepend 1-byte channel tag
                await ws.send(channel.encode() + data)
                print(f"[Client {channel}] Sent chunk #{chunk_counter}")
    except Exception as e:
        print(f"[Client {channel}] Error: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f"[Client {channel}] Done. Total chunks sent: {chunk_counter}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(send_audio())
    print("[Client] Exited.")
