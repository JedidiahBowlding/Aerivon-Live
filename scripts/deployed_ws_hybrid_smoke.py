import asyncio
import json

import websockets

URL = "wss://aerivon-live-agent-63160614207.us-central1.run.app/ws/aerivon"
DRIFT_MARKERS = ("what would you like", "how can i help", "is there anything else")


async def main() -> int:
    async with websockets.connect(URL, max_size=2**22) as ws:
        for _ in range(3):
            _ = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))

        await ws.send(json.dumps({
            "type": "text",
            "text": "Open https://example.com and turn the headline into an illustrated epic story."
        }))

        saw_intent = False
        saw_hybrid_intent = False
        saw_screenshot = False
        saw_image = 0

        for _ in range(90):
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
            msg_type = msg.get("type")

            if msg_type == "intent":
                saw_intent = True
                intent = msg.get("intent")
                if intent == "hybrid":
                    saw_hybrid_intent = True
                print("INTENT", intent)
            elif msg_type == "screenshot":
                saw_screenshot = True
                print("SCREENSHOT")
            elif msg_type == "image":
                saw_image += 1
                print("IMAGE", saw_image)
            elif msg_type == "text":
                text = (msg.get("text") or "").strip().lower()
                if any(marker in text for marker in DRIFT_MARKERS):
                    print("DRIFT_MARKER_DETECTED", text[:160])
                    return 2
            elif msg_type == "error":
                print("ERROR", msg)
                return 3
            elif msg_type == "done":
                break

        print("SUMMARY", {
            "intent_seen": saw_intent,
            "hybrid_intent": saw_hybrid_intent,
            "screenshot": saw_screenshot,
            "images": saw_image,
        })

        if not saw_hybrid_intent:
            return 4
        if not saw_screenshot:
            return 5
        if saw_image < 1:
            return 6

        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
