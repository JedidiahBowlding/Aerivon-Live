import asyncio
import base64
import json
import os

import websockets


WS_URL = os.getenv(
    "AERIVON_WS_URL_VISION",
    "ws://127.0.0.1:8080/ws/live?output=text",
)

# 32x32 RGB red square PNG (base64)
PROBE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NMQEAAAjDMOZf"
    "9G9gYQkqkJmZmZmZmZmZmZmZ2wG6GwE4i8jXUQAAAABJRU5ErkJggg=="
)


async def main() -> None:
    print("WS_URL:", WS_URL)

    async with websockets.connect(WS_URL, max_size=8 * 1024 * 1024) as ws:
        # status
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=12.0))
            if msg.get("type") == "status":
                print("STATUS:", msg)
                break

        await ws.send(
            json.dumps(
                {
                    "type": "text",
                    "text": "What color is the square in the provided image? Reply with exactly one word: RED or GREEN.",
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data_b64": PROBE_PNG_B64,
                }
            )
        )

        out = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
            if msg.get("type") == "text":
                out += msg.get("text") or ""
            if msg.get("type") == "turn_complete":
                break
            if msg.get("type") == "error":
                print("ERROR:", msg)
                break

        print("OUT:", out.strip())


if __name__ == "__main__":
    asyncio.run(main())
