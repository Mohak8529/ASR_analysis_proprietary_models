import asyncio
import websockets
from .config import WS_HOST, WS_PORT

class WebSocketAudioServer:
    def __init__(self, audio_queue, all_raw_frames):
        self.audio_queue = audio_queue
        self.all_raw_frames = all_raw_frames
        self.chunk_counter = 0

    async def connection_handler(self, websocket):
        print(f"[WS] Connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)) and len(message) > 1:
                    # Get channel tag from first byte of message
                    channel = chr(message[0])
                    pcm_bytes = message[1:]
                    self.chunk_counter += 1
                    self.audio_queue.put_nowait((channel, pcm_bytes))    # You may add timestamp here if desired
                    self.all_raw_frames.append((channel, pcm_bytes))
                    print(f"[WS] Received chunk #{self.chunk_counter} – Channel {channel} – {len(pcm_bytes)} bytes")
                else:
                    print("[WS] Non-binary or too-short message ignored.")
        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Connection closed for {websocket.remote_address}")

    async def run(self):
        print(f"[WS] Listening on ws://{WS_HOST}:{WS_PORT}")
        async def handler(websocket):
            await self.connection_handler(websocket)
        async with websockets.serve(handler, WS_HOST, WS_PORT):
            await asyncio.Future()
