import asyncio
import json
import time
import websockets

URL = "wss://aerivon-live-agent-63160614207.us-central1.run.app/ws/aerivon"


async def main():
    async with websockets.connect(URL, max_size=2**22) as ws:
        for _ in range(3):
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            print("INIT", m)

        await ws.send(json.dumps({"type": "text", "text": "Create an illustrated story about a storm tower."}))
        await asyncio.sleep(0.35)
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "interrupt"}))
        print("SENT INTERRUPT")

        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            print("MSG", m)
            if m.get("type") == "interrupted":
                print("ACK_LATENCY", round(time.perf_counter() - t0, 3))
                break

asyncio.run(main())
