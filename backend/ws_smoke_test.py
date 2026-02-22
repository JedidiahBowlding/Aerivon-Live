import asyncio
import base64
import json
import os
from pathlib import Path

import websockets


WS_URL = os.getenv(
    "AERIVON_WS_URL",
    "ws://127.0.0.1:8080/ws/live?output=text",
)

SCREENSHOT_PATH = os.getenv(
    "AERIVON_WS_SCREENSHOT",
    "/tmp/aerivon_artifacts/screenshot_1771738708.png",
)


async def recv_until(ws, *, predicate, timeout_s: float = 12.0):
    async def _inner():
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if predicate(msg):
                return msg

    return await asyncio.wait_for(_inner(), timeout=timeout_s)


async def main() -> None:
    print("WS_URL:", WS_URL)

    async with websockets.connect(WS_URL, max_size=4 * 1024 * 1024) as ws:
        status = await recv_until(ws, predicate=lambda m: m.get("type") == "status")
        print("STATUS:", status)

        # Text turn
        await ws.send(json.dumps({"type": "text", "text": "Say OK."}))
        got_ok = False
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=12.0)
            msg = json.loads(raw)
            if msg.get("type") == "text" and "ok" in (msg.get("text") or "").lower():
                got_ok = True
            if msg.get("type") == "turn_complete":
                break
        print("TEXT_OK:", got_ok)

        # Interrupt (forces server to restart session) so the next vision test runs in a clean session.
        await ws.send(json.dumps({"type": "interrupt"}))
        restarting = await recv_until(
            ws,
            predicate=lambda m: m.get("type") == "status" and m.get("status") == "restarting",
            timeout_s=8.0,
        )
        print("RESTARTING:", restarting)

        status2 = await recv_until(ws, predicate=lambda m: m.get("type") == "status" and m.get("status") == "connected", timeout_s=20.0)
        print("STATUS2:", status2)

        # Vision turn: send prompt then image, then wait for completion.
        img = Path(SCREENSHOT_PATH).read_bytes()
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

        domain = ""
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=45.0)
            msg = json.loads(raw)
            if msg.get("type") == "text":
                domain += msg.get("text") or ""
            elif msg.get("type") in {"error", "status", "interrupted"}:
                print("VISION_EVT:", msg)
                if msg.get("type") == "error":
                    break
            if msg.get("type") == "turn_complete":
                break
        domain = domain.strip().split()[0]
        print("VISION_DOMAIN:", domain)

    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
