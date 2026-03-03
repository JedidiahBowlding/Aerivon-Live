import asyncio
import json
from dataclasses import dataclass

import websockets

BASE = "aerivon-live-agent-63160614207.us-central1.run.app"
WS_AERIVON = f"wss://{BASE}/ws/aerivon"
WS_UI = f"wss://{BASE}/ws/ui"
WS_STORY = f"wss://{BASE}/ws/story"

DRIFT_MARKERS = ("what would you like", "how can i help", "is there anything else")


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


async def test_live_agent() -> TestResult:
    try:
        async with websockets.connect(WS_AERIVON, max_size=2**22) as ws:
            for _ in range(3):
                await asyncio.wait_for(ws.recv(), timeout=25)

            await ws.send(json.dumps({"type": "text", "text": "What is 2+2? Reply in one short sentence."}))

            saw_intent_live = False
            saw_text = False
            for _ in range(40):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
                msg_type = msg.get("type")
                if msg_type == "intent" and msg.get("intent") == "live":
                    saw_intent_live = True
                elif msg_type == "text":
                    txt = (msg.get("text") or "").strip().lower()
                    if txt:
                        saw_text = True
                    if any(m in txt for m in DRIFT_MARKERS):
                        return TestResult("live_agent", False, "drift marker in live response")
                elif msg_type == "error":
                    return TestResult("live_agent", False, f"error: {msg.get('error')}")
                elif msg_type == "done":
                    break

            if not saw_intent_live:
                return TestResult("live_agent", False, "missing live intent")
            if not saw_text:
                return TestResult("live_agent", False, "no text response")
            return TestResult("live_agent", True, "intent=live and response received")
    except Exception as exc:
        return TestResult("live_agent", False, f"exception: {type(exc).__name__}: {exc}")


async def test_ui_navigation() -> TestResult:
    try:
        async with websockets.connect(WS_UI, max_size=2**22) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))
            if first.get("type") != "status":
                return TestResult("ui_navigation", False, "did not receive initial status")

            await ws.send(json.dumps({"type": "open", "url": "https://example.com"}))

            saw_ready = False
            saw_screenshot = False
            for _ in range(40):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))
                msg_type = msg.get("type")
                if msg_type == "screenshot":
                    saw_screenshot = True
                elif msg_type == "status" and msg.get("status") == "ready":
                    saw_ready = True
                    break
                elif msg_type == "error":
                    return TestResult("ui_navigation", False, f"open error: {msg.get('error')}")

            if not saw_ready:
                return TestResult("ui_navigation", False, "open did not reach ready state")

            await ws.send(json.dumps({"type": "task", "text": "What is the main heading text on this page?"}))

            saw_done = False
            saw_actions = False
            for _ in range(80):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=50))
                msg_type = msg.get("type")
                if msg_type == "actions":
                    saw_actions = True
                elif msg_type == "status" and msg.get("status") == "done":
                    saw_done = True
                    break
                elif msg_type == "error":
                    return TestResult("ui_navigation", False, f"task error: {msg.get('error')}")

            if not saw_screenshot:
                return TestResult("ui_navigation", False, "no screenshot received")
            if not saw_actions:
                return TestResult("ui_navigation", False, "no planner actions emitted")
            if not saw_done:
                return TestResult("ui_navigation", False, "task did not complete")
            return TestResult("ui_navigation", True, "open+task completed with planner actions")
    except Exception as exc:
        return TestResult("ui_navigation", False, f"exception: {type(exc).__name__}: {exc}")


async def test_story_teller() -> TestResult:
    try:
        async with websockets.connect(WS_STORY, max_size=2**22) as ws:
            first = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))
            if first.get("type") != "status":
                return TestResult("story_teller", False, "did not receive initial status")

            await ws.send(json.dumps({"type": "prompt", "text": "Create an illustrated story about a girl in an attic."}))

            saw_text = False
            images = 0
            for _ in range(100):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                msg_type = msg.get("type")
                if msg_type == "text" and (msg.get("text") or "").strip():
                    saw_text = True
                    low = (msg.get("text") or "").lower()
                    if any(m in low for m in DRIFT_MARKERS):
                        return TestResult("story_teller", False, "drift marker in story response")
                elif msg_type == "image":
                    images += 1
                elif msg_type == "error":
                    return TestResult("story_teller", False, f"error: {msg.get('error')}")
                elif msg_type == "done":
                    break

            if not saw_text:
                return TestResult("story_teller", False, "no story text received")
            if images < 1:
                return TestResult("story_teller", False, "no story images received")
            return TestResult("story_teller", True, f"text received, images={images}")
    except Exception as exc:
        return TestResult("story_teller", False, f"exception: {type(exc).__name__}: {exc}")


async def main() -> int:
    tests = [test_live_agent, test_ui_navigation, test_story_teller]
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print(f"{result.name}: {'PASS' if result.passed else 'FAIL'} - {result.detail}")

    passed = sum(1 for r in results if r.passed)
    print("---")
    print(f"summary: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
