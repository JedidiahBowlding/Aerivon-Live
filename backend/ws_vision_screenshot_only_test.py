import asyncio
import base64
import json
from pathlib import Path

import websockets


WS_URL = "ws://127.0.0.1:8080/ws/live?output=text"
SCREENSHOT_PATH = "/tmp/aerivon_artifacts/screenshot_1771738708.png"


async def main() -> None:
    img = Path(SCREENSHOT_PATH).read_bytes()

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
                    "text": "What domain is shown in the screenshot? Answer with the domain only.",
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "data_b64": base64.b64encode(img).decode("ascii"),
                }
            )
        )

        out = ""
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60.0))
            if msg.get("type") == "text":
                out += msg.get("text") or ""
            if msg.get("type") == "error":
                print("ERROR:", msg)
                break
            if msg.get("type") == "turn_complete":
                break

        print("OUT:", out.strip())


if __name__ == "__main__":
    asyncio.run(main())
