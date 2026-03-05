import asyncio
import json
import statistics
import time

import websockets

URL = "wss://aerivon-live-agent-63160614207.us-central1.run.app/ws/story"
TURN_TIMEOUT_STORY = 60.0
OPEN_TIMEOUT = 30.0
MAX_RECONNECT_ATTEMPTS = 3


def _is_resource_exhausted(detail: str) -> bool:
    low = (detail or "").lower()
    return "429" in low and "resource_exhausted" in low


async def _consume_story_until_result(ws, timeout_s: float) -> tuple[bool, str, dict | None]:
    deadline = time.perf_counter() + timeout_s
    saw_progress = False
    text_count = 0
    image_count = 0
    audio_count = 0

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if saw_progress:
                return True, f"progress:text={text_count},image={image_count},audio={audio_count}", None
            return False, "post_interrupt_timeout", None

        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=min(15, max(1, remaining))))
        except TimeoutError:
            if saw_progress:
                return True, f"progress:text={text_count},image={image_count},audio={audio_count}", None
            return False, "post_interrupt_timeout", None
        msg_type = msg.get("type")

        if msg_type == "text":
            text_count += 1
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "image":
            image_count += 1
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "audio":
            audio_count += 1
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "debug":
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "status" and msg.get("status") == "generating":
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "error":
            return False, f"post_interrupt_error:{msg.get('error')}", msg
        elif msg_type == "done":
            if saw_progress or text_count > 0 or image_count > 0 or audio_count > 0:
                return True, f"done:text={text_count},image={image_count},audio={audio_count}", None
            return True, "done", None


async def one(iteration: int):
    last_error = "connect_failed"
    for connect_attempt in range(MAX_RECONNECT_ATTEMPTS):
        try:
            async with websockets.connect(URL, max_size=None, open_timeout=OPEN_TIMEOUT, close_timeout=5.0) as ws:
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
                saw_reset = False
                for _ in range(45):
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                    except TimeoutError:
                        return False, None, "interrupt_wait_timeout"
                    if msg.get("type") == "interrupted":
                        interrupt_ack = True
                        _ = interrupt_ack
                        ack = time.perf_counter() - interrupt_start
                        continue
                    if (
                        ack is not None
                        and msg.get("type") == "status"
                        and (msg.get("status") == "connected" or msg.get("state") == "IDLE")
                    ):
                        saw_reset = True
                        break
                    if msg.get("type") == "error":
                        return False, None, f"error:{msg.get('error')}"

                if ack is None:
                    return False, None, "no_interrupt_ack"
                if not saw_reset:
                    return False, ack, "no_reset_after_interrupt"

                await asyncio.sleep(1)

                prompt = "new story immediately after interrupt about a clockmaker in moonlight"
                for attempt in range(3):
                    await ws.send(json.dumps({"type": "prompt", "text": prompt}))
                    ok, detail, err_msg = await _consume_story_until_result(ws, TURN_TIMEOUT_STORY)
                    if ok:
                        return True, ack, detail
                    if err_msg and _is_resource_exhausted(str(err_msg.get("error") or detail)) and attempt < 2:
                        await asyncio.sleep(2**attempt)
                        continue
                    return False, ack, detail

                return False, ack, "post_interrupt_retry_exhausted"
        except TimeoutError as exc:
            last_error = f"exception:TimeoutError:{exc}"
            if connect_attempt < MAX_RECONNECT_ATTEMPTS - 1:
                await asyncio.sleep(2**connect_attempt)
                continue
            return False, None, last_error
        except Exception as exc:
            last_error = f"exception:{type(exc).__name__}:{exc}"
            if connect_attempt < MAX_RECONNECT_ATTEMPTS - 1:
                await asyncio.sleep(2**connect_attempt)
                continue
            return False, None, last_error

    return False, None, last_error


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
        await asyncio.sleep(2)

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
