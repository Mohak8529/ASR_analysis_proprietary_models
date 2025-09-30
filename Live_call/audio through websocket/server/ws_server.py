import asyncio
import websockets
from .config import WS_HOST, WS_PORT

class WebSocketAudioServer:
    def __init__(self, audio_queue, all_raw_frames):
        self.audio_queue = audio_queue
        self.all_raw_frames = all_raw_frames

    # NEW SIGNATURE - only websocket parameter, no path
    async def connection_handler(self, websocket):
        print(f"[WS] Connection from {websocket.remote_address}")
        try:
            async for message in websocket:
                if isinstance(message, (bytes, bytearray)):
                    self.audio_queue.put_nowait(message)
                    self.all_raw_frames.append(message)
                else:
                    print("[WS] Non-binary message ignored.")
        except websockets.exceptions.ConnectionClosed:
            print(f"[WS] Connection closed for {websocket.remote_address}")

    async def run(self):
        print(f"[WS] Listening on ws://{WS_HOST}:{WS_PORT}")
        
        # Create wrapper function with correct signature
        async def handler(websocket):
            await self.connection_handler(websocket)
        
        async with websockets.serve(handler, WS_HOST, WS_PORT):
            await asyncio.Future()  # Keep server running
