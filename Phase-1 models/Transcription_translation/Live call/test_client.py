import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import base64
import signal
import sys
import time

# Configuration
SAMPLE_RATE = 48000
MIC1_DEVICE_INDEX = 11  # AB13X USB Audio
MIC2_DEVICE_INDEX = 12  # JBL TUNE 305C USB-C Audio
BLOCKSIZE = 65536  # Increased for stability
SERVER_URL = "ws://localhost:8765"
RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY = 2.0

# Global variables
is_recording = False
websocket_connection = None
mic1_overflows = 0
mic2_overflows = 0
event_loop = None
audio_queue = asyncio.Queue()
streams_active = False
stop_event = asyncio.Event()
chunk_count_mic1 = 0
chunk_count_mic2 = 0
chunk_sent_mic1 = 0
chunk_sent_mic2 = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C to stop recording."""
    global is_recording
    if is_recording:
        print("\nStopping recording...")
        is_recording = False
        stop_event.set()
        asyncio.run_coroutine_threadsafe(wait_and_close(), event_loop)
    else:
        asyncio.run_coroutine_threadsafe(clean_exit(), event_loop)

async def clean_exit():
    """Cleanly exit the program."""
    global websocket_connection
    try:
        if websocket_connection:
            try:
                await websocket_connection.close()
            except websockets.exceptions.ConnectionClosed:
                pass
        print("Program exiting...")
    except Exception as e:
        print(f"Error during clean exit: {e}")

async def wait_and_close():
    """Wait for audio queue to be empty and all chunks sent, then close connection."""
    global websocket_connection, streams_active, chunk_count_mic1, chunk_count_mic2, chunk_sent_mic1, chunk_sent_mic2
    try:
        streams_active = False
        print(f"Waiting for audio queue (mic1 queued: {chunk_count_mic1}, mic2 queued: {chunk_count_mic2}, mic1 sent: {chunk_sent_mic1}, mic2 sent: {chunk_sent_mic2})...")
        
        start_time = time.time()
        while (not audio_queue.empty() or chunk_sent_mic1 < chunk_count_mic1 or chunk_sent_mic2 < chunk_count_mic2) and time.time() - start_time < 60.0:
            if not audio_queue.empty():
                print(f"Processing remaining chunks (queue size: {audio_queue.qsize()})...")
            await asyncio.sleep(0.1)
        
        if websocket_connection:
            try:
                await websocket_connection.send(json.dumps({"type": "stop_recording"}))
                print("Stop recording message sent")
                await asyncio.sleep(2.0)
            except websockets.exceptions.ConnectionClosed:
                print("Connection already closed")
        
        if websocket_connection:
            try:
                await websocket_connection.close()
                print("WebSocket connection closed")
            except websockets.exceptions.ConnectionClosed:
                pass
        print(f"Total chunks: mic1 queued={chunk_count_mic1}, sent={chunk_sent_mic1}; mic2 queued={chunk_count_mic2}, sent={chunk_sent_mic2}")
    except Exception as e:
        print(f"Error in wait_and_close: {e}")
    finally:
        await clean_exit()

def callback_mic1(indata, frames, time, status):
    """Callback for microphone 1."""
    global mic1_overflows, is_recording, event_loop, chunk_count_mic1
    print(f"Mic1 callback triggered, status: {status}, frames: {frames}")
    
    if status:
        if str(status) == "input_overflow":
            mic1_overflows += 1
            print(f"Mic1 overflow detected (total: {mic1_overflows})")
    
    if is_recording and event_loop and streams_active:
        if indata.shape[1] > 1:
            mono_data = np.mean(indata, axis=1, keepdims=True)
        else:
            mono_data = indata.copy()
        
        try:
            chunk_count_mic1 += 1
            print(f"Queuing mic1 chunk {chunk_count_mic1} ({len(mono_data)/SAMPLE_RATE:.2f}s)")
            event_loop.call_soon_threadsafe(
                lambda: audio_queue.put_nowait((mono_data, "mic1", bool(status)))
            )
        except Exception as e:
            print(f"Error queuing mic1 audio: {e}")

def callback_mic2(indata, frames, time, status):
    """Callback for microphone 2."""
    global mic2_overflows, is_recording, event_loop, chunk_count_mic2
    print(f"Mic2 callback triggered, status: {status}, frames: {frames}")
    
    if status:
        if str(status) == "input_overflow":
            mic2_overflows += 1
            print(f"Mic2 overflow detected (total: {mic2_overflows})")
    
    if is_recording and event_loop and streams_active:
        if indata.shape[1] > 1:
            mono_data = np.mean(indata, axis=1, keepdims=True)
        else:
            mono_data = indata.copy()
        
        try:
            chunk_count_mic2 += 1
            print(f"Queuing mic2 chunk {chunk_count_mic2} ({len(mono_data)/SAMPLE_RATE:.2f}s)")
            event_loop.call_soon_threadsafe(
                lambda: audio_queue.put_nowait((mono_data, "mic2", bool(status)))
            )
        except Exception as e:
            print(f"Error queuing mic2 audio: {e}")

async def process_audio_queue():
    """Process audio data from queue and send to server."""
    global websocket_connection, chunk_sent_mic1, chunk_sent_mic2
    
    while True:
        try:
            audio_data, speaker, overflow = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            chunk_id = chunk_count_mic1 + 1 if speaker == "mic1" else chunk_count_mic2 + 1
            
            for attempt in range(RECONNECT_ATTEMPTS):
                try:
                    if not websocket_connection:
                        await attempt_reconnect()
                    
                    await send_audio_data(audio_data, speaker, overflow, chunk_id)
                    print(f"Sent {speaker} chunk {chunk_id}")
                    if speaker == "mic1":
                        chunk_sent_mic1 += 1
                    else:
                        chunk_sent_mic2 += 1
                    break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed for {speaker} chunk {chunk_id}, attempt {attempt + 1}/{RECONNECT_ATTEMPTS}: {e}")
                    if attempt < RECONNECT_ATTEMPTS - 1:
                        await attempt_reconnect()
                        await asyncio.sleep(RECONNECT_DELAY)
                    else:
                        print(f"Failed to send {speaker} chunk {chunk_id} after {RECONNECT_ATTEMPTS} attempts")
                except Exception as e:
                    print(f"Error sending {speaker} chunk {chunk_id}, attempt {attempt + 1}/{RECONNECT_ATTEMPTS}: {e}")
                    if attempt < RECONNECT_ATTEMPTS - 1:
                        await asyncio.sleep(RECONNECT_DELAY)
                    else:
                        print(f"Failed to send {speaker} chunk {chunk_id} after {RECONNECT_ATTEMPTS} attempts")
                
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            print(f"Error processing audio queue: {e}")
            await asyncio.sleep(0.01)

async def send_audio_data(audio_data, speaker, overflow, chunk_id):
    """Send audio data to server via WebSocket."""
    global websocket_connection
    try:
        audio_bytes = audio_data.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            "type": "audio_data",
            "speaker": speaker,
            "audio": audio_b64,
            "overflow": overflow,
            "chunk_id": chunk_id
        }
        
        await websocket_connection.send(json.dumps(message))
    except Exception as e:
        print(f"Error sending audio data: {e}")
        raise

async def attempt_reconnect():
    """Attempt to reconnect to the server."""
    global websocket_connection
    print("Attempting to reconnect...")
    for attempt in range(RECONNECT_ATTEMPTS):
        try:
            websocket_connection = await websockets.connect(
                SERVER_URL,
                ping_interval=60,
                ping_timeout=30,
                close_timeout=60
            )
            print("Reconnected successfully")
            await websocket_connection.send(json.dumps({"type": "start_recording"}))
            return
        except Exception as e:
            print(f"Reconnect attempt {attempt + 1}/{RECONNECT_ATTEMPTS} failed: {e}")
            if attempt < RECONNECT_ATTEMPTS - 1:
                await asyncio.sleep(RECONNECT_DELAY)
    print("Failed to reconnect after max attempts")

async def handle_server_messages(websocket):
    """Handle messages from server."""
    global is_recording
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if data["type"] == "ready":
                    print(f"Server message: {data['message']}")
                    print("Sending start request...")
                    await websocket.send(json.dumps({"type": "start_recording"}))
                
                elif data["type"] == "recording_started":
                    print(f"Server: {data['message']}")
                    print("🎤 Recording started! You can now speak into your earphones.")
                    print("Press Ctrl+C when done speaking to stop.")
                    is_recording = True
                
                elif data["type"] == "transcription":
                    trans_data = data["data"]
                    print(f"\n[{trans_data['start']:.2f} - {trans_data['end']:.2f}] ({trans_data['speaker']}): {trans_data['text']}")
                    print(f"Translated: {trans_data['translated_text']}")
                
                elif data["type"] == "recording_stopped":
                    print(f"Server message: {data['message']}")
                    print("Recording session completed")
                    is_recording = False
                
                elif data["type"] == "ack":
                    print(f"Received ACK for {data['speaker']} chunk {data['chunk_id']}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Error handling server message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("Connection to server closed")
        is_recording = False

async def start_audio_streams():
    """Start audio input streams."""
    global is_recording, websocket_connection, event_loop, streams_active, chunk_count_mic1, chunk_count_mic2, chunk_sent_mic1, chunk_sent_mic2
    
    chunk_count_mic1 = 0
    chunk_count_mic2 = 0
    chunk_sent_mic1 = 0
    chunk_sent_mic2 = 0
    
    event_loop = asyncio.get_running_loop()
    
    sd.default.blocksize = BLOCKSIZE
    sd.default.latency = 'high'
    
    print("Available audio devices:")
    print(sd.query_devices())
    print(f"Mic1 device index: {MIC1_DEVICE_INDEX} (AB13X USB Audio)")
    print(f"Mic2 device index: {MIC2_DEVICE_INDEX} (JBL TUNE 305C USB-C Audio)")
    
    try:
        mic1_info = sd.query_devices(MIC1_DEVICE_INDEX)
        mic2_info = sd.query_devices(MIC2_DEVICE_INDEX)
        channels_mic1 = max(1, min(mic1_info['max_input_channels'], 2))
        channels_mic2 = max(1, min(mic2_info['max_input_channels'], 2))
        print(f"Detected Mic1 channels: {channels_mic1} (max: {mic1_info['max_input_channels']})")
        print(f"Detected Mic2 channels: {channels_mic2} (max: {mic2_info['max_input_channels']})")
    except ValueError as e:
        print(f"Error querying devices: {e}")
        print("Ensure device indices are valid and devices are connected.")
        return False
    
    try:
        stream1 = sd.InputStream(
            device=MIC1_DEVICE_INDEX,
            channels=channels_mic1,
            samplerate=SAMPLE_RATE,
            callback=callback_mic1,
            blocksize=BLOCKSIZE,
            latency='high'
        )
        stream2 = sd.InputStream(
            device=MIC2_DEVICE_INDEX,
            channels=channels_mic2,
            samplerate=SAMPLE_RATE,
            callback=callback_mic2,
            blocksize=BLOCKSIZE,
            latency='high'
        )
        
        print("Audio streams initialized successfully")
        
        queue_task = asyncio.create_task(process_audio_queue())
        
        with stream1, stream2:
            print("Audio streams started. Waiting for server instructions...")
            streams_active = True
            try:
                await stop_event.wait()
            except asyncio.CancelledError:
                print("Audio streams cancelled")
            finally:
                streams_active = False
                print("Finalizing audio queue...")
                start_time = time.time()
                while not audio_queue.empty() and time.time() - start_time < 60.0:
                    print(f"Processing remaining chunks (queue size: {audio_queue.qsize()})...")
                    await asyncio.sleep(0.1)
                
    except Exception as e:
        print(f"Error setting up audio streams: {e}")
        return False
    
    return True

async def main():
    global websocket_connection
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=== Live Call Transcription Client ===")
    print(f"Connecting to {SERVER_URL}...")
    
    for attempt in range(RECONNECT_ATTEMPTS):
        try:
            async with websockets.connect(
                SERVER_URL,
                ping_interval=60,
                ping_timeout=30,
                close_timeout=60
            ) as client:
                print("Connected to server successfully!")
                websocket_connection = client
                
                audio_task = asyncio.create_task(start_audio_streams())
                message_task = asyncio.create_task(handle_server_messages(client))
                
                await asyncio.gather(audio_task, message_task, return_exceptions=True)
            break
        except Exception as e:
            print(f"Connection failed, attempt {attempt + 1}/{RECONNECT_ATTEMPTS}: {e}")
            if attempt < RECONNECT_ATTEMPTS - 1:
                await asyncio.sleep(RECONNECT_DELAY)
    else:
        print("Failed to connect to server after max attempts")
    
    print("Client disconnected.")

if __name__ == "__main__":
    asyncio.run(main())