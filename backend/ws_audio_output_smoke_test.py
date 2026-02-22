import asyncio
import json
import os

import websockets


WS_URL = os.getenv(
    "AERIVON_WS_URL_AUDIO",
    "ws://127.0.0.1:8080/ws/live?output=audio",
)


async def main() -> None:
    print("WS_URL:", WS_URL)

    audio_chunks = 0
    transcript_chars = 0
    interrupted_events = 0

    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
        # status
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=12.0))
            if msg.get("type") == "status":
                print("STATUS:", msg)
                break

        await ws.send(json.dumps({"type": "text", "text": "Say the word HELLO."}))

        # Collect until turn_complete
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20.0))
            t = msg.get("type")
            if t == "audio":
                audio_chunks += 1
            elif t == "transcript":
                transcript_chars += len(msg.get("text") or "")
            elif t == "interrupted":
                interrupted_events += 1
            elif t == "turn_complete":
                break

    print("AUDIO_CHUNKS:", audio_chunks)
    print("TRANSCRIPT_CHARS:", transcript_chars)
    print("INTERRUPTED_EVENTS:", interrupted_events)


if __name__ == "__main__":
    asyncio.run(main())
