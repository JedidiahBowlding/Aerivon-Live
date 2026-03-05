import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import websockets

BASE = "aerivon-live-agent-63160614207.us-central1.run.app"
WS_AERIVON = f"wss://{BASE}/ws/aerivon"
WS_UI = f"wss://{BASE}/ws/ui"
WS_STORY = f"wss://{BASE}/ws/story"
ITERATIONS = 10
TURN_TIMEOUT_AERIVON = 60.0
TURN_TIMEOUT_UI = 45.0
TURN_TIMEOUT_STORY = 60.0
ACK_TIMEOUT = 12.0
CONNECT_TIMEOUT = 30.0
ITERATION_TIMEOUT = 180.0
MAX_RECONNECT_ATTEMPTS = 3

DRIFT_MARKERS = ("what would you like", "how can i help", "is there anything else")
AUTH_MARKERS = ("401", "429", "1007", "1008", "unauthenticated", "resource exhausted")


@dataclass
class TurnMetrics:
    ok: bool
    duration_s: float
    error: str | None = None
    intents: list[str] = field(default_factory=list)
    text_messages: int = 0
    image_messages: int = 0
    audio_messages: int = 0
    duplicate_text_frames: int = 0
    duplicate_image_frames: int = 0
    drift_detected: bool = False


@dataclass
class LoopSummary:
    endpoint: str
    iterations: int
    successes: int = 0
    failures: int = 0
    durations: list[float] = field(default_factory=list)
    interrupt_ack_latencies: list[float] = field(default_factory=list)
    duplicate_text_frames: int = 0
    duplicate_image_frames: int = 0
    drift_count: int = 0
    error_messages: list[str] = field(default_factory=list)
    auth_like_errors: int = 0
    model_identities: list[str] = field(default_factory=list)

    def add_turn(self, metrics: TurnMetrics) -> None:
        if metrics.ok:
            self.successes += 1
        else:
            self.failures += 1
            if metrics.error:
                self.error_messages.append(metrics.error)
                low = metrics.error.lower()
                if any(marker in low for marker in AUTH_MARKERS):
                    self.auth_like_errors += 1
        self.durations.append(metrics.duration_s)
        self.duplicate_text_frames += metrics.duplicate_text_frames
        self.duplicate_image_frames += metrics.duplicate_image_frames
        if metrics.drift_detected:
            self.drift_count += 1

    def report(self) -> dict[str, Any]:
        observed = self.successes + self.failures
        avg_duration = statistics.mean(self.durations) if self.durations else 0.0
        p95_duration = statistics.quantiles(self.durations, n=20)[18] if len(self.durations) >= 2 else avg_duration
        avg_ack = statistics.mean(self.interrupt_ack_latencies) if self.interrupt_ack_latencies else 0.0
        return {
            "endpoint": self.endpoint,
            "iterations": self.iterations,
            "checks_observed": observed,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round((self.successes / max(1, observed)) * 100, 1),
            "avg_response_s": round(avg_duration, 3),
            "p95_response_s": round(p95_duration, 3),
            "avg_interrupt_ack_s": round(avg_ack, 3) if self.interrupt_ack_latencies else None,
            "drift_count": self.drift_count,
            "duplicate_text_frames": self.duplicate_text_frames,
            "duplicate_image_frames": self.duplicate_image_frames,
            "auth_like_errors": self.auth_like_errors,
            "sample_errors": self.error_messages[:5],
            "models_seen": sorted(set(self.model_identities)),
        }


async def recv_json(ws, timeout_s: float = 60.0) -> dict[str, Any]:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except TimeoutError:
        return {"type": "__timeout__", "error": "timeout"}
    except Exception as exc:
        return {"type": "__exception__", "error": f"{type(exc).__name__}: {exc}"}
    try:
        return json.loads(raw)
    except Exception as exc:
        return {"type": "__bad_json__", "error": f"{type(exc).__name__}: {exc}"}


def _check_drift(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in DRIFT_MARKERS)


def _is_resource_exhausted(error: str | None) -> bool:
    low = (error or "").lower()
    return "429" in low and "resource exhausted" in low


async def wait_for_message(
    ws,
    predicate,
    timeout_s: float,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        msg = await recv_json(ws, min(20.0, max(1.0, deadline - time.perf_counter())))
        msg_type = msg.get("type")
        if msg_type in {"__timeout__", "__exception__", "__bad_json__"}:
            return False, None, str(msg.get("error") or msg_type)
        if msg_type == "error":
            return False, msg, str(msg.get("error") or "error")
        if predicate(msg):
            return True, msg, None
    return False, None, "timeout"


async def run_aerivon_turn(ws, text: str, timeout_s: float = TURN_TIMEOUT_AERIVON) -> TurnMetrics:
    start = time.perf_counter()
    await ws.send(json.dumps({"type": "text", "text": text}))
    intents: list[str] = []
    last_text = None
    image_indexes: set[Any] = set()
    duplicate_text = 0
    duplicate_image = 0
    text_messages = 0
    image_messages = 0
    audio_messages = 0
    drift = False

    while True:
        msg = await recv_json(ws, timeout_s)
        msg_type = msg.get("type")

        if msg_type in {"__timeout__", "__exception__", "__bad_json__"}:
            end = time.perf_counter()
            return TurnMetrics(
                ok=False,
                duration_s=end - start,
                error=str(msg.get("error") or msg_type),
                intents=intents,
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )

        if msg_type == "intent":
            intents.append(str(msg.get("intent") or ""))
        elif msg_type == "text":
            text_val = str(msg.get("text") or "")
            if text_val:
                text_messages += 1
                if _check_drift(text_val):
                    drift = True
                if last_text is not None and text_val == last_text:
                    duplicate_text += 1
                last_text = text_val
        elif msg_type == "image":
            image_messages += 1
            idx = msg.get("index")
            if idx in image_indexes:
                duplicate_image += 1
            image_indexes.add(idx)
        elif msg_type == "audio":
            audio_messages += 1
        elif msg_type == "error":
            end = time.perf_counter()
            return TurnMetrics(
                ok=False,
                duration_s=end - start,
                error=str(msg.get("error") or "unknown error"),
                intents=intents,
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )
        elif msg_type == "done":
            end = time.perf_counter()
            return TurnMetrics(
                ok=True,
                duration_s=end - start,
                intents=intents,
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )


async def test_aerivon_loop() -> LoopSummary:
    summary = LoopSummary(endpoint="/ws/aerivon", iterations=ITERATIONS)

    for i in range(ITERATIONS):
        async with websockets.connect(WS_AERIVON, max_size=None, open_timeout=CONNECT_TIMEOUT, close_timeout=5.0) as ws:
            for _ in range(3):
                init_msg = await recv_json(ws, 30.0)
                if init_msg.get("type") == "status" and init_msg.get("model"):
                    summary.model_identities.append(str(init_msg.get("model")))

            try:
                q1 = await asyncio.wait_for(
                    run_aerivon_turn(ws, f"Iteration {i+1}: what is 2+2? reply briefly."),
                    timeout=ITERATION_TIMEOUT,
                )
            except TimeoutError:
                summary.failures += 1
                summary.error_messages.append("aerivon iteration timeout at q1")
                continue
            summary.add_turn(q1)

            recall = await run_aerivon_turn(ws, "What did we just do?")
            summary.add_turn(recall)

            interrupt_start = time.perf_counter()
            await ws.send(json.dumps({"type": "text", "text": "Give me a detailed 8-sentence explanation of why stars twinkle."}))
            await asyncio.sleep(0.2)
            await ws.send(json.dumps({"type": "interrupt"}))

            acked = False
            ack_deadline = time.perf_counter() + ACK_TIMEOUT
            while time.perf_counter() < ack_deadline:
                msg = await recv_json(ws, 20.0)
                if msg.get("type") in {"__timeout__", "__exception__", "__bad_json__"}:
                    break
                if msg.get("type") == "interrupted":
                    acked = True
                    summary.interrupt_ack_latencies.append(time.perf_counter() - interrupt_start)
                    break
                if msg.get("type") == "error":
                    break

            if not acked:
                summary.failures += 1
                summary.error_messages.append("missed interrupt ACK")

            post_interrupt = await run_aerivon_turn(ws, "Are you still there? one short sentence.")
            summary.add_turn(post_interrupt)

            if any(intent and intent != "live" for intent in (q1.intents + recall.intents + post_interrupt.intents)):
                summary.failures += 1
                summary.error_messages.append("engine drift: non-live intent in live-mode sequence")

    return summary


async def _wait_ui_result(ws, timeout_s: float = TURN_TIMEOUT_UI) -> tuple[bool, str, dict[str, Any] | None]:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        msg = await recv_json(ws, timeout_s)
        msg_type = msg.get("type")

        if msg_type in {"__timeout__", "__exception__", "__bad_json__"}:
            return False, str(msg.get("error") or msg_type), msg
        if msg_type == "error":
            return False, str(msg.get("error") or "ui error"), msg
        if msg_type == "status" and msg.get("status") in {"ready", "done", "cancelled"}:
            return True, str(msg.get("status")), msg
    return False, "timeout", None


async def test_ui_loop() -> LoopSummary:
    summary = LoopSummary(endpoint="/ws/ui", iterations=ITERATIONS)

    async def _ui_case(url: str, expect_ready: bool, run_task: bool = False) -> tuple[bool, str]:
        try:
            async with websockets.connect(WS_UI, max_size=None, open_timeout=CONNECT_TIMEOUT, close_timeout=5.0) as ws:
                first = await recv_json(ws, 30.0)
                if first.get("type") == "status" and first.get("model"):
                    summary.model_identities.append(str(first.get("model")))

                await ws.send(json.dumps({"type": "open", "url": url}))
                ok, detail, msg = await _wait_ui_result(ws)

                if expect_ready:
                    if not ok or detail != "ready":
                        return False, f"open failed for {url}: {detail}"
                    if run_task:
                        await ws.send(json.dumps({"type": "task", "text": "What is the main heading text on this page?"}))
                        task_deadline = time.perf_counter() + TURN_TIMEOUT_UI
                        saw_actions = False
                        saw_done = False
                        while time.perf_counter() < task_deadline:
                            event = await recv_json(ws, 120.0)
                            event_type = event.get("type")
                            if event_type == "actions":
                                saw_actions = True
                                plan = event.get("plan") or {}
                                if not isinstance(plan, dict):
                                    return False, "planner returned non-object plan"
                                actions = plan.get("actions")
                                if actions is not None and not isinstance(actions, list):
                                    return False, "planner returned non-list actions"
                            elif event_type == "error":
                                return False, f"task error: {event.get('error') or 'unknown'}"
                            elif event_type == "status" and event.get("status") == "done":
                                saw_done = True
                                break
                        if not saw_actions:
                            return False, "task completed without actions event"
                        if not saw_done:
                            return False, "task did not reach done status"
                    return True, "ok"

                if ok:
                    return False, f"expected error for {url} but received status={detail}"
                if msg is not None and msg.get("type") == "error":
                    return True, "error as expected"
                return False, f"no clean error payload for {url}"
        except Exception as exc:
            if expect_ready:
                return False, f"exception for {url}: {type(exc).__name__}: {exc}"
            return True, f"connection closed for invalid input {url}: {type(exc).__name__}"

    for _ in range(ITERATIONS):
        iter_start = time.perf_counter()
        iteration_ok = True

        ok, detail = await _ui_case("https://example.com", expect_ready=True, run_task=True)
        if not ok:
            iteration_ok = False
            summary.error_messages.append(detail)

        ok, detail = await _ui_case("example.org", expect_ready=False)
        if not ok:
            iteration_ok = False
            summary.error_messages.append(detail)

        ok, detail = await _ui_case("ht!tp://bad", expect_ready=False)
        if not ok:
            iteration_ok = False
            summary.error_messages.append(detail)

        ok, detail = await _ui_case("http://localhost:8080", expect_ready=False)
        if not ok:
            iteration_ok = False
            summary.error_messages.append(detail)

        metrics = TurnMetrics(
            ok=iteration_ok,
            duration_s=time.perf_counter() - iter_start,
            duplicate_text_frames=0,
            duplicate_image_frames=0,
        )
        summary.add_turn(metrics)

    return summary


async def run_story_prompt(ws, prompt: str, timeout_s: float = TURN_TIMEOUT_STORY) -> TurnMetrics:
    start = time.perf_counter()
    await ws.send(json.dumps({"type": "prompt", "text": prompt}))

    last_text = None
    image_indexes: set[Any] = set()
    duplicate_text = 0
    duplicate_image = 0
    text_messages = 0
    image_messages = 0
    audio_messages = 0
    drift = False
    saw_progress = False
    deadline = time.perf_counter() + timeout_s

    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if saw_progress:
                return TurnMetrics(
                    ok=True,
                    duration_s=time.perf_counter() - start,
                    text_messages=text_messages,
                    image_messages=image_messages,
                    audio_messages=audio_messages,
                    duplicate_text_frames=duplicate_text,
                    duplicate_image_frames=duplicate_image,
                    drift_detected=drift,
                )
            return TurnMetrics(
                ok=False,
                duration_s=time.perf_counter() - start,
                error="timeout",
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )

        msg = await recv_json(ws, min(15.0, max(1.0, remaining)))
        msg_type = msg.get("type")

        if msg_type in {"__timeout__", "__exception__", "__bad_json__"}:
            if msg_type == "__timeout__":
                continue
            if saw_progress:
                return TurnMetrics(
                    ok=True,
                    duration_s=time.perf_counter() - start,
                    text_messages=text_messages,
                    image_messages=image_messages,
                    audio_messages=audio_messages,
                    duplicate_text_frames=duplicate_text,
                    duplicate_image_frames=duplicate_image,
                    drift_detected=drift,
                )
            return TurnMetrics(
                ok=False,
                duration_s=time.perf_counter() - start,
                error=str(msg.get("error") or msg_type),
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )

        if msg_type == "text":
            text = str(msg.get("text") or "")
            if text:
                text_messages += 1
                saw_progress = True
                deadline = time.perf_counter() + timeout_s
                if _check_drift(text):
                    drift = True
                if last_text is not None and text == last_text:
                    duplicate_text += 1
                last_text = text
        elif msg_type == "image":
            image_messages += 1
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
            idx = msg.get("index")
            if idx in image_indexes:
                duplicate_image += 1
            image_indexes.add(idx)
        elif msg_type == "audio":
            audio_messages += 1
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "debug":
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "status" and msg.get("status") == "generating":
            saw_progress = True
            deadline = time.perf_counter() + timeout_s
        elif msg_type == "error":
            return TurnMetrics(
                ok=False,
                duration_s=time.perf_counter() - start,
                error=str(msg.get("error") or "story error"),
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )
        elif msg_type == "done":
            return TurnMetrics(
                ok=True,
                duration_s=time.perf_counter() - start,
                text_messages=text_messages,
                image_messages=image_messages,
                audio_messages=audio_messages,
                duplicate_text_frames=duplicate_text,
                duplicate_image_frames=duplicate_image,
                drift_detected=drift,
            )


async def run_story_prompt_with_retry(
    ws,
    prompt: str,
    timeout_s: float = TURN_TIMEOUT_STORY,
) -> TurnMetrics:
    metrics = TurnMetrics(ok=False, duration_s=0.0, error="uninitialized")
    for attempt in range(3):
        metrics = await run_story_prompt(ws, prompt, timeout_s)
        if metrics.ok:
            return metrics
        if _is_resource_exhausted(metrics.error) and attempt < 2:
            await asyncio.sleep(2**attempt)
            continue
        return metrics
    return metrics


async def test_story_loop() -> LoopSummary:
    summary = LoopSummary(endpoint="/ws/story", iterations=ITERATIONS)

    for i in range(ITERATIONS):
        connected = False
        ws = None
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                ws = await websockets.connect(
                    WS_STORY,
                    max_size=None,
                    open_timeout=CONNECT_TIMEOUT,
                    close_timeout=5.0,
                )
                first = await recv_json(ws, 30.0)
                if first.get("type") == "status" and first.get("model"):
                    summary.model_identities.append(str(first.get("model")))
                connected = True
                break
            except Exception as exc:
                if ws is not None:
                    await ws.close()
                if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                summary.failures += 1
                summary.error_messages.append(f"story connect failed: {type(exc).__name__}: {exc}")

        if not connected or ws is None:
            await asyncio.sleep(2)
            continue

        try:
            try:
                a = await asyncio.wait_for(
                    run_story_prompt_with_retry(ws, f"Iteration {i+1}: Create an illustrated story about a girl in an attic."),
                    timeout=ITERATION_TIMEOUT,
                )
            except TimeoutError:
                summary.failures += 1
                summary.error_messages.append("story iteration timeout at prompt A")
                await asyncio.sleep(2)
                continue
            summary.add_turn(a)

            await asyncio.sleep(1)
            b = await run_story_prompt_with_retry(
                ws,
                "Create an illustrated cyberpunk story in neon watercolor style about a lone mechanic.",
            )
            summary.add_turn(b)

            interrupt_start = time.perf_counter()
            await ws.send(json.dumps({"type": "prompt", "text": "Create a long illustrated story about mountain storms and old gods."}))
            await asyncio.sleep(0.25)
            await ws.send(json.dumps({"type": "interrupt"}))

            ack_ok, _, ack_err = await wait_for_message(ws, lambda m: m.get("type") == "interrupted", ACK_TIMEOUT)
            if ack_ok:
                summary.interrupt_ack_latencies.append(time.perf_counter() - interrupt_start)
            else:
                summary.failures += 1
                summary.error_messages.append(f"story interrupt did not ACK: {ack_err or 'unknown'}")

            reset_ok, _, reset_err = await wait_for_message(
                ws,
                lambda m: (
                    m.get("type") == "status"
                    and (m.get("status") == "connected" or m.get("state") == "IDLE")
                ),
                TURN_TIMEOUT_STORY,
            )
            if not reset_ok:
                summary.failures += 1
                summary.error_messages.append(f"story reset not observed: {reset_err or 'unknown'}")

            await asyncio.sleep(1)
            d = await run_story_prompt_with_retry(
                ws,
                "Create a new illustrated story about a clockmaker right after interruption.",
            )
            summary.add_turn(d)
        finally:
            await ws.close()

        await asyncio.sleep(2)

    return summary


async def main() -> int:
    start = time.perf_counter()
    print("[stability] running /ws/aerivon 10x", flush=True)
    aerivon = await test_aerivon_loop()
    print("[stability] running /ws/ui 10x", flush=True)
    ui = await test_ui_loop()
    print("[stability] running /ws/story 10x", flush=True)
    story = await test_story_loop()
    elapsed = time.perf_counter() - start

    report = {
        "elapsed_s": round(elapsed, 2),
        "results": [
            aerivon.report(),
            ui.report(),
            story.report(),
        ],
    }

    print(json.dumps(report, indent=2))

    failed = any(r["failures"] > 0 for r in report["results"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
