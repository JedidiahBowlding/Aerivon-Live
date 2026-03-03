import asyncio
import json
import sys

import websockets

URL = "wss://aerivon-live-agent-63160614207.us-central1.run.app/ws/aerivon"
DRIFT_MARKERS = ("what would you like", "how can i help", "is there anything else")


async def main() -> int:
    async with websockets.connect(URL, max_size=2**22) as ws:
        for _ in range(3):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))
            print("INIT", msg.get("type"), msg.get("state") or msg.get("message") or (msg.get("text") or "")[:80])

        await ws.send(json.dumps({"type": "text", "text": "Create an illustrated story about a girl in an attic."}))

        saw_intent = False
        saw_text = False
        image_count = 0

        for _ in range(60):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            msg_type = msg.get("type")

            if msg_type == "intent":
                saw_intent = True
                print("INTENT", msg.get("intent"))
            elif msg_type == "text":
                text = (msg.get("text") or "").strip()
                if text:
                    saw_text = True
                lowered = text.lower()
                if any(marker in lowered for marker in DRIFT_MARKERS):
                    print("DRIFT_MARKER_DETECTED", lowered[:160])
                    return 2
            elif msg_type == "image":
                image_count += 1
                print("IMAGE", image_count)
            elif msg_type == "error":
                print("ERROR", msg)
                return 3
            elif msg_type == "done":
                break

        print("SUMMARY", {"intent": saw_intent, "text": saw_text, "images": image_count})

        if not saw_intent:
            print("FAIL: no intent event")
            return 4
        if image_count < 2:
            print("FAIL: expected at least 2 images")
            return 5

        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        print("EXCEPTION", type(exc).__name__, str(exc))
        raise
