import asyncio
import json
import statistics
import time

import websockets

URL = "wss://aerivon-live-agent-63160614207.us-central1.run.app/ws/story"


async def one(iteration: int):
    async with websockets.connect(URL, max_size=2**22) as ws:
        try:
            await asyncio.wait_for(ws.recv(), timeout=20)
        except TimeoutError:
            return False, None, "connect_timeout"

        await ws.send(
            json.dumps(
                {
                    "type": "prompt",
                    "text": f"Iteration {iteration}: create a vivid illustrated story about storms and old towers.",
                }
            )
        )
        await asyncio.sleep(0.35)

        interrupt_start = time.perf_counter()
        await ws.send(json.dumps({"type": "interrupt"}))

        ack = None
        for _ in range(30):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            except TimeoutError:
                return False, None, "interrupt_wait_timeout"
            if msg.get("type") == "status" and msg.get("status") == "interrupted":
                ack = time.perf_counter() - interrupt_start
                break
            if msg.get("type") == "error":
                return False, None, f"error:{msg.get('error')}"

        if ack is None:
            return False, None, "no_interrupt_ack"

        await ws.send(
            json.dumps(
                {
                    "type": "prompt",
                    "text": "new story immediately after interrupt about a clockmaker in moonlight",
                }
            )
        )

        saw_done = False
        images = 0
        for _ in range(80):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            except TimeoutError:
                return False, ack, "post_interrupt_timeout"
            if msg.get("type") == "image":
                images += 1
            if msg.get("type") == "done":
                saw_done = True
                break
            if msg.get("type") == "error":
                return False, ack, f"post_interrupt_error:{msg.get('error')}"

        if not saw_done:
            return False, ack, "post_interrupt_no_done"
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
        print(
            json.dumps(
                {
                    "iteration": i,
                    "ok": ok,
                    "ack_s": None if ack is None else round(ack, 3),
                    "detail": detail,
                }
            )
        )

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
