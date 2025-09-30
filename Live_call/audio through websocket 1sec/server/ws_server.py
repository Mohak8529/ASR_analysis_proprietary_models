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
                if isinstance(message, (bytes, bytearray)):
                    self.chunk_counter += 1
                    self.audio_queue.put_nowait(message)
                    self.all_raw_frames.append(message)
                    # Log each chunk received
                    print(f"[WS] Received chunk #{self.chunk_counter} – {len(message)} bytes")
                else:
                    print("[WS] Non-binary message ignored.")
        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Connection closed for {websocket.remote_address}")

    async def run(self):
        print(f"[WS] Listening on ws://{WS_HOST}:{WS_PORT}")
        async def handler(websocket):
            await self.connection_handler(websocket)
        async with websockets.serve(handler, WS_HOST, WS_PORT):
            await asyncio.Future()
