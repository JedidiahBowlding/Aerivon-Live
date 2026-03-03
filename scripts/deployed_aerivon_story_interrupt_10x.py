import asyncio
import json
import statistics
import time

import websockets

URL = "wss://aerivon-live-agent-63160614207.us-central1.run.app/ws/aerivon"


async def recv_json(ws, timeout=30):
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(raw)


async def one(iteration: int):
    async with websockets.connect(URL, max_size=2**22) as ws:
        for _ in range(3):
            await recv_json(ws, 25)

        await ws.send(json.dumps({"type": "text", "text": f"Iteration {iteration}: create an illustrated story about a tower in a storm."}))
        await asyncio.sleep(0.35)

        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "interrupt"}))

        ack = None
        for _ in range(40):
            msg = await recv_json(ws, 25)
            if msg.get("type") == "interrupted":
                ack = time.perf_counter() - t0
                break
            if msg.get("type") == "error":
                return False, None, f"error:{msg.get('error')}"

        if ack is None:
            return False, None, "no_interrupt_ack"

        await ws.send(json.dumps({"type": "text", "text": "Create a new illustrated story about a clockmaker immediately after interrupt."}))
        saw_done = False
        saw_story_intent = False
        images = 0
        for _ in range(120):
            msg = await recv_json(ws, 30)
            t = msg.get("type")
            if t == "intent" and msg.get("intent") == "story":
                saw_story_intent = True
            elif t == "image":
                images += 1
            elif t == "error":
                return False, ack, f"post_interrupt_error:{msg.get('error')}"
            elif t == "done":
                saw_done = True
                break

        if not saw_done:
            return False, ack, "post_interrupt_no_done"
        if not saw_story_intent:
            return False, ack, "post_interrupt_wrong_intent"
        if images < 1:
            return False, ack, "post_interrupt_no_images"
        return True, ack, f"images={images}"


async def main() -> int:
    oks = []
    acks = []
    errors = []

    for i in range(1, 11):
        try:
            ok, ack, detail = await one(i)
        except Exception as exc:
            ok, ack, detail = False, None, f"exception:{type(exc).__name__}:{exc}"

        oks.append(ok)
        if ack is not None:
            acks.append(ack)
        if not ok:
            errors.append({"iteration": i, "detail": detail})

        print(json.dumps({"iteration": i, "ok": ok, "ack_s": None if ack is None else round(ack, 3), "detail": detail}))

    summary = {
        "passed": sum(1 for ok in oks if ok),
        "failed": sum(1 for ok in oks if not ok),
        "avg_ack_s": round(statistics.mean(acks), 3) if acks else None,
        "p95_ack_s": round(sorted(acks)[max(0, int(len(acks) * 0.95) - 1)], 3) if acks else None,
        "errors": errors,
    }
    print(json.dumps({"summary": summary}, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
