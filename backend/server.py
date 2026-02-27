from __future__ import annotations

import asyncio
import base64
from datetime import datetime
import json
import os
import random
import re
import sys
import time
import hashlib
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.cloud import storage

from playwright.async_api import async_playwright

from agent import AerivonLiveAgent
from gemini_client import check_live_model_availability, resolve_fallback_model


def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 32.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Retry a function with exponential backoff for transient errors.
    
    Handles 429 RESOURCE_EXHAUSTED and other transient API errors.
    """
    def wrapper(*args, **kwargs):
        retries = 0
        delay = initial_delay
        
        while retries <= max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if this is a retryable error
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                is_transient = any(x in error_str for x in [
                    "503", "500", "UNAVAILABLE", "DEADLINE_EXCEEDED"
                ])
                
                if not (is_rate_limit or is_transient):
                    # Not a retryable error, raise immediately
                    raise
                
                if retries >= max_retries:
                    print(f"[RETRY] Max retries ({max_retries}) exceeded. Giving up.", file=sys.stderr)
                    raise
                
                # Calculate delay with jitter
                wait_time = min(delay, max_delay)
                if jitter:
                    wait_time = wait_time * (0.5 + random.random())  # 50-150% of delay
                
                retries += 1
                print(
                    f"[RETRY] Attempt {retries}/{max_retries} failed with {type(e).__name__}. "
                    f"Retrying in {wait_time:.1f}s...",
                    file=sys.stderr
                )
                time.sleep(wait_time)
                delay *= exponential_base
        
        raise Exception(f"Failed after {max_retries} retries")
    
    return wrapper


app = FastAPI(title="Aerivon Live Agent API")

# CORS: allow the demo frontend (served on a different port) to call the backend.
_cors_origins_env = (os.getenv("AERIVON_CORS_ORIGINS") or "").strip()
if _cors_origins_env:
    cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    cors_origins = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug middleware to log WebSocket upgrade requests
@app.middleware("http")
async def log_ws_requests(request: Request, call_next):
    if request.url.path.startswith("/ws/"):
        print(f"DEBUG: Path={request.url.path} Method={request.method} Headers={dict(request.headers)}", flush=True)
    response = await call_next(request)
    return response

agent: AerivonLiveAgent | None = None
SESSION_TOOL_RESULTS: dict[str, list[dict[str, Any]]] = {}
LAST_REQUEST_TIME: dict[str, float] = {}

# Active SSE streams per user (used for server-side interruption on new request).
ACTIVE_SSE_CANCEL: dict[str, asyncio.Event] = {}

MAX_MESSAGE_LENGTH = 4000
MAX_SESSION_RESULTS = 100
RATE_LIMIT_SECONDS = 1.0
MAX_RESULT_SIZE = 20000
MAX_WS_MESSAGE_BYTES = 256 * 1024
DEFAULT_LIVE_AUDIO_SAMPLE_RATE = int(os.getenv("AERIVON_LIVE_AUDIO_SAMPLE_RATE", "24000"))

API_KEY_ENV_VARS = ("GOOGLE_CLOUD_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _get_api_key() -> str | None:
    for name in API_KEY_ENV_VARS:
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return None


def _make_genai_client(*, prefer_vertex: bool, project: str | None, location: str) -> genai.Client:
    http_options = HttpOptions(api_version="v1beta1")
    if prefer_vertex and project:
        return genai.Client(vertexai=True, project=project, location=location, http_options=http_options)

    api_key = _get_api_key()
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_options)

    # No credentials configured.
    raise ValueError(
        "Missing credentials. Set GOOGLE_GENAI_USE_VERTEXAI=true + GOOGLE_CLOUD_PROJECT (and ADC), "
        "or set an API key env var (GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_CLOUD_API_KEY)."
    )

# Live generation tuning. Live sessions can default to relatively small output budgets unless specified.
# Bump the default so audio replies are less likely to stop mid-sentence.
AERIVON_LIVE_MAX_OUTPUT_TOKENS = int(os.getenv("AERIVON_LIVE_MAX_OUTPUT_TOKENS", "2048"))
AERIVON_LIVE_TEMPERATURE = float(os.getenv("AERIVON_LIVE_TEMPERATURE", "0.7"))

# Persistent memory (optional): store one JSON per user in GCS.
AERIVON_MEMORY_BUCKET = os.getenv("AERIVON_MEMORY_BUCKET", "").strip()
AERIVON_MEMORY_PREFIX = os.getenv("AERIVON_MEMORY_PREFIX", "memory/").strip() or "memory/"
AERIVON_MEMORY_MAX_EXCHANGES = int(os.getenv("AERIVON_MEMORY_MAX_EXCHANGES", "6"))

# Persistent memory (optional): Firestore document per user.
# If set, Firestore takes precedence over GCS for memory I/O.
AERIVON_FIRESTORE_COLLECTION = os.getenv("AERIVON_FIRESTORE_COLLECTION", "").strip()

UI_MAX_STEPS = int(os.getenv("AERIVON_UI_MAX_STEPS", "6"))
UI_MODEL = os.getenv("AERIVON_UI_MODEL", "gemini-2.0-flash").strip()
INJECTION_PATTERNS = (
    "ignore previous instructions",
    "reveal system prompt",
    "export secrets",
    "exfiltrate",
)
BLOCKED_HOST_PATTERNS = (
    "metadata.google.internal",
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
)
PRIVATE_IP_REGEX = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})\b"
)


def _get_agent() -> AerivonLiveAgent:
    global agent
    if agent is None:
        agent = AerivonLiveAgent()
    return agent


def _sanitize_user_id(raw: str | None) -> str:
    raw = (raw or "").strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", raw or ""):
        return raw
    if not raw:
        return "default"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _memory_user_key(*, user_id: str, scope: str | None) -> str:
    scope_raw = (scope or "").strip()
    if not scope_raw:
        return user_id
    scope_id = _sanitize_user_id(scope_raw)
    if scope_id == "default":
        return user_id
    composite = f"{user_id}__{scope_id}"
    return _sanitize_user_id(composite)


def _memory_blob_name(user_id: str) -> str:
    prefix = AERIVON_MEMORY_PREFIX
    if not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{user_id}.json"


async def _load_user_memory(*, user_id: str) -> dict[str, Any] | None:
    if AERIVON_FIRESTORE_COLLECTION:
        def _load_fs() -> dict[str, Any] | None:
            try:
                from google.cloud import firestore  # type: ignore
            except Exception:
                return None

            client = firestore.Client()
            doc = client.collection(AERIVON_FIRESTORE_COLLECTION).document(user_id).get()
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            return data if isinstance(data, dict) else None

        try:
            return await asyncio.to_thread(_load_fs)
        except Exception:
            return None

    if not AERIVON_MEMORY_BUCKET:
        return None

    blob_name = _memory_blob_name(user_id)

    def _load_gcs() -> dict[str, Any] | None:
        client = storage.Client()
        bucket = client.bucket(AERIVON_MEMORY_BUCKET)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            return None
        text = blob.download_as_text(encoding="utf-8")
        data = json.loads(text) if text else {}
        if not isinstance(data, dict):
            return None
        return data

    try:
        return await asyncio.to_thread(_load_gcs)
    except Exception:
        return None


async def _save_user_memory(*, user_id: str, memory: dict[str, Any]) -> None:
    if AERIVON_FIRESTORE_COLLECTION:
        def _save_fs() -> None:
            try:
                from google.cloud import firestore  # type: ignore
            except Exception:
                return

            client = firestore.Client()
            client.collection(AERIVON_FIRESTORE_COLLECTION).document(user_id).set(
                memory,
                merge=True,
            )

        try:
            await asyncio.to_thread(_save_fs)
        except Exception:
            return
        return

    if not AERIVON_MEMORY_BUCKET:
        return

    blob_name = _memory_blob_name(user_id)

    def _save_gcs() -> None:
        client = storage.Client()
        bucket = client.bucket(AERIVON_MEMORY_BUCKET)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(memory, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

    try:
        await asyncio.to_thread(_save_gcs)
    except Exception:
        return


async def _append_exchange_to_memory(*, user_id: str, user_text: str, model_text: str) -> None:
    if not (AERIVON_FIRESTORE_COLLECTION or AERIVON_MEMORY_BUCKET):
        return

    mem = await _load_user_memory(user_id=user_id) or {}
    exchanges = mem.get("exchanges")
    if not isinstance(exchanges, list):
        exchanges = []

    exchanges.append(
        {
            "t": int(time.time()),
            "user": (user_text or "").strip()[:2000],
            "model": (model_text or "").strip()[:4000],
        }
    )
    exchanges = exchanges[-AERIVON_MEMORY_MAX_EXCHANGES :]
    mem["user_id"] = user_id
    mem["updated_at"] = int(time.time())
    mem["exchanges"] = exchanges
    joined = " ".join(
        [f"U:{ex.get('user','')} M:{ex.get('model','')}" for ex in exchanges if isinstance(ex, dict)]
    )
    mem["summary"] = joined[:1200]
    await _save_user_memory(user_id=user_id, memory=mem)


def _memory_to_prompt(memory: dict[str, Any] | None) -> str:
    if not memory or not isinstance(memory, dict):
        return ""
    exchanges = memory.get("exchanges")
    if not isinstance(exchanges, list) or not exchanges:
        return ""

    # Keep it short: last few exchanges only.
    lines: list[str] = []
    lines.append("Persistent user memory (from previous sessions):")
    summary = memory.get("summary")
    if isinstance(summary, str) and summary.strip():
        lines.append(f"Summary: {summary.strip()}")

    lines.append("Recent context:")
    for ex in exchanges[-AERIVON_MEMORY_MAX_EXCHANGES:]:
        if not isinstance(ex, dict):
            continue
        u = str(ex.get("user") or "").strip()
        m = str(ex.get("model") or "").strip()
        if u:
            lines.append(f"- User: {u[:400]}")
        if m:
            lines.append(f"- Model: {m[:600]}")

    return "\n".join(lines).strip()


async def _check_live_model_availability_fast(project: str | None, location: str) -> dict[str, Any]:
    """Bound the Live availability probe so endpoints don't hang.

    Some environments/network paths can cause models.list() to stall.
    This helper runs the synchronous probe in a thread and applies a short timeout.
    """

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(check_live_model_availability, project, location),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        return {"live_models_available": False, "error": "probe_timeout"}
    except Exception as exc:
        return {"live_models_available": False, "error": str(exc)}


def _contains_unsafe_target(message: str) -> bool:
    lowered = message.lower()
    if any(host in lowered for host in BLOCKED_HOST_PATTERNS):
        return True

    if PRIVATE_IP_REGEX.search(lowered):
        return True

    for candidate in re.findall(r"https?://[^\s\"']+", lowered):
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
        if host in BLOCKED_HOST_PATTERNS:
            return True
    return False


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of a JSON object from model text."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")

    # Strip markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()

    # Fast path.
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Scan for first balanced {...}.
    start = text.find("{")
    if start < 0:
        raise ValueError("no json object found")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated json object")


async def _annotate_screenshot(page, png_bytes: bytes) -> bytes:
    """Draw red bounding boxes around all clickable elements so Gemini can click precisely."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        rects = await page.evaluate("""() => {
            const els = document.querySelectorAll('a, button, input, select, textarea, [onclick], [role=button], [role=link], [tabindex]');
            return Array.from(els).slice(0, 80).map(el => {
                const r = el.getBoundingClientRect();
                return {
                    x: Math.round(r.left),
                    y: Math.round(r.top),
                    w: Math.round(r.width),
                    h: Math.round(r.height),
                    text: el.textContent.trim().slice(0, 30),
                    tag: el.tagName.toLowerCase()
                };
            }).filter(r => r.w > 2 && r.h > 2 && r.x >= 0 && r.y >= 0);
        }""")
        
        print(f"[UI NAV DEBUG] Found {len(rects)} elements before annotation: {rects}", file=__import__("sys").stderr, flush=True)

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        for r in rects:
            x1, y1, x2, y2 = r['x'], r['y'], r['x'] + r['w'], r['y'] + r['h']
            cx, cy = r['x'] + r['w'] // 2, r['y'] + r['h'] // 2
            
            # VERY prominent red box (thick border)
            for thickness in range(4):
                draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], outline='red', width=1)
            
            # LARGE red dot at center  
            dot_radius = 8
            draw.ellipse([cx-dot_radius, cy-dot_radius, cx+dot_radius, cy+dot_radius], fill='red', outline='yellow', width=2)
            
            # Draw coordinate text AT the center point
            coord_text = f"({cx},{cy})"
            draw.text((cx + dot_radius + 2, cy - 10), coord_text, fill='yellow')
            
            # Label with element text above the box
            label = r.get('text') or r.get('tag', '')
            if label:
                draw.text((x1 + 2, y1 - 15), label[:20], fill='yellow')

        out = io.BytesIO()
        img.save(out, format='PNG')
        
        # Save annotated screenshot for debugging
        try:
            img.save('/tmp/aerivon_annotated_debug.png')
            print(f"[UI NAV DEBUG] Saved annotated screenshot to /tmp/aerivon_annotated_debug.png", file=__import__("sys").stderr, flush=True)
        except Exception:
            pass
        
        print(f"[UI NAV DEBUG] Annotated screenshot with {len(rects)} clickable elements", file=__import__("sys").stderr, flush=True)
        return out.getvalue()

    except ImportError:
        print("[UI NAV DEBUG] Pillow not installed, skipping annotation", file=__import__("sys").stderr, flush=True)
        return png_bytes
    except Exception as e:
        print(f"[UI NAV DEBUG] Annotation failed: {e}", file=__import__("sys").stderr, flush=True)
        return png_bytes


def _ui_action_prompt(task: str, memory: list[str], clickable_elements: list[dict] = None) -> str:
    allowed = [
        "goto (url)",
        "click (element_index)",
        "type (text)",
        "press (key)",
        "scroll (delta_y)",
        "wait (ms)",
    ]
    context = "\n".join(memory[-12:]).strip()
    
    # Build clickable elements list for prompt
    elements_text = ""
    if clickable_elements:
        elements_text = "\n\n**Clickable Elements (use element_index to click):**\n"
        for idx, el in enumerate(clickable_elements):
            cx = el['x'] + el['w'] // 2
            cy = el['y'] + el['h'] // 2
            text = el.get('text', '')[:30]
            tag = el.get('tag', 'unknown')
            elements_text += f"  [{idx}] {tag}: \"{text}\" at center ({cx},{cy})\n"

    return (
        "You are Aerivon UI Navigator - an AI agent that DIRECTLY CONTROLS a real browser.\n\n"
        "CRITICAL: You ARE NOT just a language model. You HAVE a browser automation backend (Playwright).\n"
        "When you return JSON actions, they ARE IMMEDIATELY EXECUTED in a real Chromium browser.\n"
        "The browser DOES navigate, click, type, and take screenshots based on your JSON output.\n\n"
        "NEVER say things like:\n"
        "  - 'I cannot directly interact'\n"
        "  - 'I am just a language model'\n"
        "  - 'I cannot click buttons'\n"
        "These are FALSE. You CAN and DO control the browser through your JSON action output.\n\n"
        "Your workflow:\n"
        "1. You receive a screenshot of the current browser state\n"
        "2. You return JSON actions (click, type, scroll, etc.)\n"
        "3. The backend EXECUTES them in the real browser\n"
        "4. You receive the next screenshot showing the results\n\n"
        "Return ONLY valid JSON (no markdown) matching this schema:\n"
        "{\n"
        "  \"actions\": [\n"
        "    {\"type\": \"goto\", \"url\": \"https://...\"} |\n"
        "    {\"type\": \"click\", \"element_index\": 0} |\n"
        "    {\"type\": \"type\", \"text\": \"...\"} |\n"
        "    {\"type\": \"press\", \"key\": \"Enter\"} |\n"
        "    {\"type\": \"scroll\", \"delta_y\": 500} |\n"
        "    {\"type\": \"wait\", \"ms\": 1000}\n"
        "  ],\n"
        "  \"done\": true|false,\n"
        "  \"note\": \"Describe the action being executed (e.g., 'Clicked the login button', 'Navigated to homepage')\"\n"
        "}\n\n"
        "CRITICAL 'note' field rules:\n"
        "  ✅ GOOD: \"Clicked the 'Learn more' link\", \"Typed search query\", \"Scrolling to footer\"\n"
        "  ❌ BAD: \"I cannot interact\", \"I'm unable to\", \"As an AI I can't\"\n"
        "  Remember: Your actions ARE being executed in a real browser. Describe what you're doing, not what you can't do.\n\n"
        f"Allowed action types: {', '.join(allowed)}.\n"
        "CRITICAL: For clicks, use 'element_index' (NOT x,y coordinates). Choose the index from the list below.\n"
        f"{elements_text}\n"
        "Rules: do not invent URLs; do not access localhost/private IPs/metadata.\n"
        "If the target element isn't visible, prefer scroll then another step.\n\n"
        "Context (what has happened so far):\n"
        f"{context}\n\n"
        f"User intent (do not change this goal): {task}\n"
    )


async def _ui_plan_actions(*, client: genai.Client, screenshot_png: bytes, task: str, memory: list[str], page=None) -> dict[str, Any]:
    # Extract clickable elements list
    clickable_elements = []
    if page is not None:
        clickable_elements = await page.evaluate("""() => {
            const els = document.querySelectorAll('a, button, input, select, textarea, [onclick], [role=button], [role=link], [tabindex]');
            return Array.from(els).slice(0, 80).map(el => {
                const r = el.getBoundingClientRect();
                return {
                    x: Math.round(r.left),
                    y: Math.round(r.top),
                    w: Math.round(r.width),
                    h: Math.round(r.height),
                    text: el.textContent.trim().slice(0, 30),
                    tag: el.tagName.toLowerCase()
                };
            }).filter(r => r.w > 2 && r.h > 2 && r.x >= 0 && r.y >= 0);
        }""")
        
        # Also annotate screenshot with red boxes for visual confirmation
        screenshot_png = await _annotate_screenshot(page, screenshot_png)
    
    cfg = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=_ui_action_prompt(task, memory, clickable_elements)),
                types.Part.from_bytes(data=screenshot_png, mime_type="image/png"),
            ],
        )
    ]

    # Run sync Gemini call in thread to avoid blocking async event loop
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model=UI_MODEL,
        contents=contents,
        config=cfg,
    )
    
    if not resp.candidates:
        return {"error": "Gemini returned no candidates"}
    
    parts = resp.candidates[0].content.parts
    text = "".join(p.text for p in parts if getattr(p, "text", None))
    result = _extract_json_object(text)
    
    # Store clickable elements in result for click handler to use
    result['_clickable_elements'] = clickable_elements
    return result


async def _ui_screenshot_b64(page) -> tuple[str, bytes]:
    png = await page.screenshot(full_page=False, type="png")
    return base64.b64encode(png).decode("ascii"), png


@app.websocket("/ws/ui")
async def ws_ui(websocket: WebSocket) -> None:
    """UI Navigator WS: Gemini multimodal plans JSON actions, backend executes via Playwright."""

    await websocket.accept()

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not use_vertex or not project:
        await websocket.send_json({"type": "error", "error": "Vertex not enabled"})
        await websocket.close(code=1011)
        return

    gen_client = genai.Client(http_options=HttpOptions(api_version="v1beta1"))

    session_id = int(time.time())
    await websocket.send_json({"type": "status", "status": "connected", "session_id": session_id, "model": UI_MODEL})

    cancel_flag = False
    memory: list[str] = []
    current_task: str | None = None

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1366, "height": 768},
                device_scale_factor=1,  # Prevent coordinate scaling issues
                java_script_enabled=True,
                ignore_https_errors=True,
            )
            page = await context.new_page()

            async def send(payload: dict[str, Any]) -> None:
                payload.setdefault("session_id", session_id)
                await websocket.send_json(payload)

            async def safe_goto(url: str) -> None:
                from tools import is_safe_url

                if not is_safe_url(url):
                    raise ValueError("Blocked unsafe URL")
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                await page.wait_for_timeout(800)

            async def execute_action(action: dict[str, Any], clickable_elements: list[dict] = None) -> dict[str, Any]:
                nonlocal cancel_flag, page
                if cancel_flag:
                    return {"ok": False, "skipped": True, "reason": "cancelled"}

                t = str(action.get("type") or "").strip().lower()
                if t == "goto":
                    url = str(action.get("url") or "")
                    await safe_goto(url)
                    return {"ok": True, "type": "goto", "url": url}
                if t == "click":
                    # Support element_index (new) or x,y coordinates (fallback)
                    if "element_index" in action:
                        idx = int(action.get("element_index"))
                        if not clickable_elements or idx < 0 or idx >= len(clickable_elements):
                            return {"ok": False, "error": f"Invalid element_index: {idx}", "max_index": len(clickable_elements or []) - 1}
                        
                        el = clickable_elements[idx]
                        x = el['x'] + el['w'] // 2
                        y = el['y'] + el['h'] // 2
                        print(f"[UI NAV DEBUG] Using element_index {idx}: {el.get('tag')} \"{el.get('text', '')}\" at ({x},{y})", file=__import__("sys").stderr, flush=True)
                    else:
                        # Fallback to x,y coordinates
                        x = int(action.get("x"))
                        y = int(action.get("y"))
                        print(f"[UI NAV DEBUG] Using x,y coordinates: ({x},{y})", file=__import__("sys").stderr, flush=True)
                    
                    # Debug: Identify the element at the click coordinates
                    target_info = await page.evaluate(
                        """([x, y]) => {
                            const el = document.elementFromPoint(x, y);
                            if (!el) return null;
                            return {
                                tag: el.tagName,
                                text: (el.innerText || el.textContent || '').trim().slice(0, 50),
                                href: el.href || el.closest('a')?.href || null,
                                id: el.id || null,
                                class: el.className || null
                            };
                        }""",
                        [x, y],
                    )
                    
                    print(f"[UI NAV DEBUG] Click target at ({x},{y}): {target_info}", file=__import__("sys").stderr, flush=True)
                    
                    if not target_info:
                        return {"ok": False, "error": "No element at coordinates", "x": x, "y": y}
                    
                    # Track state before click
                    url_before = page.url
                    
                    # Click the actual DOM element (more reliable than mouse.click)
                    try:
                        element_handle = await page.evaluate_handle(
                            """([x, y]) => document.elementFromPoint(x, y)""",
                            [x, y],
                        )
                        if element_handle:
                            await element_handle.as_element().click()
                        else:
                            # Fallback to mouse click
                            await page.mouse.click(x, y)
                    except Exception as e:
                        print(f"[UI NAV DEBUG] Element click failed, using mouse: {e}", file=__import__("sys").stderr, flush=True)
                        await page.mouse.click(x, y)
                    
                    # Wait for navigation or content update
                    try:
                        await page.wait_for_load_state("domcontentloaded", timeout=2000)
                    except Exception:
                        await page.wait_for_timeout(500)
                    
                    # Handle new tabs or popups
                    if len(context.pages) > 1:
                        page = context.pages[-1]
                        await page.wait_for_load_state("domcontentloaded", timeout=2000)
                    
                    url_after = page.url
                    
                    # Verify something actually happened
                    if url_after == url_before and not target_info.get('href'):
                        # Might be SPA or button click - that's OK
                        return {
                            "ok": True,
                            "type": "click",
                            "x": x,
                            "y": y,
                            "url_after": url_after,
                            "target": target_info,
                            "navigation": False
                        }
                    
                    return {
                        "ok": True,
                        "type": "click",
                        "x": x,
                        "y": y,
                        "url_after": url_after,
                        "target": target_info,
                        "navigation": url_after != url_before
                    }
                if t == "type":
                    text = str(action.get("text") or "")
                    await page.keyboard.type(text)
                    return {"ok": True, "type": "type", "text": text}
                if t == "press":
                    key = str(action.get("key") or "Enter")
                    await page.keyboard.press(key)
                    await page.wait_for_timeout(600)
                    return {"ok": True, "type": "press", "key": key}
                if t == "scroll":
                    dy = int(action.get("delta_y") or 0)
                    await page.mouse.wheel(0, dy)
                    await page.wait_for_timeout(300)
                    return {"ok": True, "type": "scroll", "delta_y": dy}
                if t == "wait":
                    ms = int(action.get("ms") or 0)
                    await page.wait_for_timeout(max(0, ms))
                    return {"ok": True, "type": "wait", "ms": ms}

                return {"ok": False, "error": f"unknown action type: {t}"}

            try:
                while True:
                    msg = await websocket.receive_json()
                    if not isinstance(msg, dict):
                        continue

                    msg_type = str(msg.get("type") or "").strip().lower()

                    if msg_type == "interrupt":
                        cancel_flag = True
                        await send({"type": "interrupted", "source": "client"})
                        continue

                    if msg_type == "open":
                        cancel_flag = False
                        # Starting a new navigation flow resets intent/memory.
                        memory.clear()
                        current_task = None
                        url = str(msg.get("url") or "")
                        await send({"type": "status", "status": "navigating", "url": url})
                        await safe_goto(url)
                        try:
                            title = await page.title()
                        except Exception:
                            title = ""
                        memory.append(f"Opened URL: {page.url} Title: {title}")
                        b64, _png = await _ui_screenshot_b64(page)
                        await send({"type": "screenshot", "mime_type": "image/png", "data_b64": b64, "url": page.url})
                        await send({"type": "status", "status": "ready", "url": page.url})
                        continue

                    if msg_type == "task":
                        cancel_flag = False
                        task = str(msg.get("text") or "")
                        if not task:
                            await send({"type": "error", "error": "missing text"})
                            continue

                        # Persist the original intent across steps.
                        if current_task is None:
                            current_task = task
                            memory.append(f"User intent: {current_task}")
                        else:
                            # Treat subsequent task messages as clarifications.
                            memory.append(f"User clarification: {task}")

                        task = current_task

                        # Prevent planning against an empty page.
                        if (page.url or "").startswith("about:blank"):
                            await send({"type": "error", "error": "No page loaded yet. Click Open URL first."})
                            continue

                        await send({"type": "status", "status": "planning", "task": task})

                        for step in range(UI_MAX_STEPS):
                            if cancel_flag:
                                await send({"type": "status", "status": "cancelled"})
                                break

                            b64, png = await _ui_screenshot_b64(page)
                            try:
                                title = await page.title()
                            except Exception:
                                title = ""
                            memory.append(f"Step {step+1} URL: {page.url} Title: {title}")

                            plan = await _ui_plan_actions(client=gen_client, screenshot_png=png, task=task, memory=memory, page=page)
                            await send({"type": "actions", "step": step + 1, "plan": plan})

                            memory.append(f"Planned: {json.dumps(plan, ensure_ascii=False)[:1500]}")

                            # Extract clickable elements for click handler
                            clickable_elements = plan.get("_clickable_elements") or []

                            actions = plan.get("actions") or []
                            if not isinstance(actions, list):
                                await send({"type": "error", "error": "plan.actions must be a list"})
                                break

                            for idx, action in enumerate(actions):
                                if cancel_flag:
                                    await send({"type": "status", "status": "cancelled"})
                                    break
                                if not isinstance(action, dict):
                                    await send({"type": "error", "error": "action must be an object"})
                                    break
                                res = await execute_action(action, clickable_elements)
                                await send({"type": "action_result", "index": idx, "result": res})
                                memory.append(f"Executed action {idx}: {json.dumps(action, ensure_ascii=False)} => {json.dumps(res, ensure_ascii=False)[:800]}")

                            b64_after, _png_after = await _ui_screenshot_b64(page)
                            await send(
                                {
                                    "type": "screenshot",
                                    "mime_type": "image/png",
                                    "data_b64": b64_after,
                                    "url": page.url,
                                }
                            )

                            if bool(plan.get("done")) is True:
                                await send({"type": "status", "status": "done", "note": plan.get("note") or ""})
                                memory.append(f"Done: {plan.get('note') or ''}")
                                break

                        continue

            except WebSocketDisconnect:
                return
            finally:
                try:
                    await context.close()
                except Exception:
                    pass
                try:
                    await browser.close()
                except Exception:
                    pass
    except Exception as e:
        import sys
        import traceback
        print(f"[WS/UI ERROR] Playwright/WebSocket error: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        try:
            await websocket.send_json({"type": "error", "error": f"Server error: {str(e)}"})
            await websocket.close(code=1011)
        except Exception:
            pass


@app.websocket("/ws/story")
async def ws_story(websocket: WebSocket) -> None:
    """Interactive Storybook WS: Gemini interleaved text+image output with TTS narration.

    Client messages:
    - {"type": "prompt", "text": "Once upon a time..."}
    - {"type": "interrupt"}

    Server messages:
    - {"type": "status", "status": "connected"|"generating"|"done"}
    - {"type": "text", "text": "...", "index": N}          <- narration chunk
    - {"type": "image", "data_b64": "...", "mime_type": "image/png", "index": N}
    - {"type": "audio", "data_b64": "...", "index": N}     <- TTS for preceding text
    - {"type": "error", "error": "..."}
    - {"type": "done"}
    """

    await websocket.accept()

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not use_vertex or not project:
        await websocket.send_json({"type": "error", "error": "Vertex AI not configured"})
        await websocket.close(code=1011)
        return

    session_id = int(time.time())
    cancel_flag = False

    async def send(payload: dict) -> None:
        payload.setdefault("session_id", session_id)
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    # Live Audio Narration — synthesize text to PCM16 audio via Gemini Live API
    async def live_narrate(text: str) -> bytes | None:
        """Use Gemini Live API to narrate text with high-quality voice."""
        text = (text or "").strip()
        if not text or len(text) < 3:
            return None
        try:
            client = _make_genai_client(prefer_vertex=True, project=project, location=location)
            
            # Use async Live API with system instruction to just read text aloud
            async with client.aio.live.connect(
                model="gemini-2.0-flash-live-preview-04-09",
                config=types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    system_instruction="You are a professional narrator. Your only job is to read the provided text aloud exactly as written, with expression and emotion. Do not respond, comment, or add anything. Just narrate the text word-for-word.",
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Aoede"  # Warm, storytelling voice
                            )
                        )
                    ),
                ),
            ) as session:
                # Send text for narration with explicit instruction
                narration_prompt = f"Please read this text aloud:\n\n{text}"
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=narration_prompt)]
                    ),
                    turn_complete=True
                )
                
                # Collect audio chunks
                audio_chunks = []
                async for response in session.receive():
                    if response.data:
                        audio_chunks.append(response.data)
                    if response.server_content and response.server_content.turn_complete:
                        break
                
                if audio_chunks:
                    # Concatenate all PCM16 chunks
                    return b"".join(audio_chunks)
                return None
                
        except Exception as e:
            print(f"[STORY NARRATION] Error: {e}", file=__import__("sys").stderr)
            import traceback
            traceback.print_exc()
            return None

    # Image generation helper using Imagen 3 via Vertex AI
    async def generate_image(prompt: str) -> bytes | None:
        if not prompt or len(prompt) < 3:
            return None
        try:
            import vertexai  # type: ignore
            from vertexai.preview.vision_models import ImageGenerationModel  # type: ignore

            def _generate() -> bytes | None:
                # Initialize Vertex AI
                vertexai.init(project=project, location="global")
                
                model = ImageGenerationModel.from_pretrained("imagegeneration@006")
                images = model.generate_images(
                    prompt=f"{prompt}, storybook illustration art style, vibrant colors, whimsical",
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="block_some",
                    person_generation="allow_adult",
                )
                if images and len(images.images) > 0:
                    return images.images[0]._image_bytes
                return None

            return await asyncio.to_thread(_generate)
        except Exception as e:
            print(f"[STORY IMAGE] Error generating image: {e}", file=sys.stderr)
            return None

    await send({"type": "status", "status": "connected", "model": "gemini-2.5-flash-image-preview"})

    try:
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict):
                continue

            msg_type = str(msg.get("type") or "").strip().lower()

            if msg_type == "interrupt":
                cancel_flag = True
                await send({"type": "status", "status": "interrupted"})
                cancel_flag = False
                continue

            if msg_type != "prompt":
                continue

            prompt = str(msg.get("text") or "").strip()
            if not prompt:
                await send({"type": "error", "error": "missing prompt text"})
                continue

            cancel_flag = False
            await send({"type": "status", "status": "generating"})

            # Build the story prompt that instructs Gemini to interleave text + images
            story_prompt = (
                "You are a creative storyteller and visual artist. "
                "Create an immersive, illustrated story based on the user's prompt. "
                "Structure it as 2 scenes only. For each scene:\n"
                "1. Write 2-3 sentences of vivid narration\n"
                "2. Generate an illustration for that scene in a beautiful storybook art style\n"
                "Alternate between narration and images throughout. "
                "Make the story engaging, emotional, and visually rich.\n\n"
                f"Story prompt: {prompt}"
            )

            try:
                gen_client = _make_genai_client(
                    prefer_vertex=True, project=project, location="global"
                )

                @retry_with_exponential_backoff
                def _run_story() -> list[dict]:
                    """Run generate_content in thread, return list of parts as dicts."""
                    print(f"[STORY DEBUG] Calling Gemini with model: gemini-2.5-flash-image-preview", file=sys.stderr)
                    try:
                        resp = gen_client.models.generate_content(
                            model="gemini-2.5-flash-image-preview",
                            contents=story_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["TEXT", "IMAGE"],
                                temperature=0.9,
                                max_output_tokens=4096,
                            ),
                        )
                        print(f"[STORY DEBUG] Gemini response received successfully", file=sys.stderr)
                    except Exception as gemini_error:
                        print(f"[STORY ERROR] Gemini API call failed: {type(gemini_error).__name__}: {gemini_error}", file=sys.stderr)
                        raise
                    
                    # Extract parts from multimodal response
                    parts = []
                    for candidate in resp.candidates:
                        for part in candidate.content.parts:
                            if part.text:
                                parts.append({"kind": "text", "text": part.text})
                            elif part.inline_data:
                                parts.append({
                                    "kind": "image",
                                    "data": base64.b64encode(part.inline_data.data).decode("ascii"),
                                    "mime_type": part.inline_data.mime_type,
                                })
                    
                    print(f"[STORY DEBUG] Extracted {len(parts)} parts from response", file=sys.stderr)
                    return parts

                parts = await asyncio.to_thread(_run_story)

                # Stream parts to client with TTS for text chunks
                pending_text = ""
                for idx, part in enumerate(parts):
                    if cancel_flag:
                        break

                    if part["kind"] == "text":
                        text = part["text"]
                        pending_text += text
                        await send({"type": "text", "text": text, "index": idx})

                    elif part["kind"] == "image":
                        # Narrate accumulated text before this image using Gemini Live
                        if pending_text.strip():
                            audio_bytes = await live_narrate(pending_text)
                            if audio_bytes:
                                await send({
                                    "type": "audio",
                                    "data_b64": base64.b64encode(audio_bytes).decode("ascii"),
                                    "mime_type": "audio/pcm;rate=24000",
                                    "sample_rate": 24000,
                                    "index": idx,
                                })
                            pending_text = ""

                        await send({
                            "type": "image",
                            "data_b64": part["data"],
                            "mime_type": part["mime_type"],
                            "index": idx,
                        })

                # Narrate any trailing text after last image
                if pending_text.strip() and not cancel_flag:
                    audio_bytes = await live_narrate(pending_text)
                    if audio_bytes:
                        await send({
                            "type": "audio",
                            "data_b64": base64.b64encode(audio_bytes).decode("ascii"),
                            "mime_type": "audio/pcm;rate=24000",
                            "sample_rate": 24000,
                            "index": 9999,
                        })

                await send({"type": "status", "status": "done"})
                await send({"type": "done"})

            except Exception as e:
                await send({"type": "error", "error": str(e)})

    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await send({"type": "error", "error": str(exc)})
        except Exception:
            pass


class AgentMessageRequest(BaseModel):
    user_id: str | None = None
    message: str = Field(min_length=1)


class AgentMessageResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]]


class ToolResultRequest(BaseModel):
    session_id: str
    tool_name: str
    tool_call_id: str | None = None
    result: dict[str, Any]


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    lang: str | None = None
    voice: str | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    status = await _check_live_model_availability_fast(project, location)

    return {
        "status": "ok" if status["live_models_available"] else "live_model_unavailable",
        "project": project,
        "location": location,
    }


@app.post("/story/save")
async def save_story(request: Request) -> dict[str, Any]:
    """Save a generated story to GCS."""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        scenes = data.get("scenes", [])
        created = data.get("created", "")
        
        if not prompt or not scenes:
            raise HTTPException(status_code=400, detail="Missing prompt or scenes")
        
        # Generate unique ID from timestamp + prompt hash
        story_id = f"{int(time.time())}_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}"
        
        # Get GCS bucket (use same bucket as agent memory)
        bucket_name = os.getenv("AERIVON_MEMORY_BUCKET", "aerivon-live-agent-memory-1771792693")
        
        # Prepare story data
        story_data = {
            "id": story_id,
            "prompt": prompt,
            "scenes": scenes,
            "created": created,
            "saved_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Save to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"stories/{story_id}.json")
        
        blob.upload_from_string(
            json.dumps(story_data, indent=2),
            content_type="application/json"
        )
        
        print(f"[STORY SAVE] Saved story {story_id} to gs://{bucket_name}/stories/{story_id}.json", file=sys.stderr)
        
        return {
            "success": True,
            "story_id": story_id,
            "url": f"gs://{bucket_name}/stories/{story_id}.json"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[STORY SAVE ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to save story: {str(e)}")


@app.get("/story/list")
async def list_stories() -> dict[str, Any]:
    """List all saved stories from GCS."""
    try:
        bucket_name = os.getenv("AERIVON_MEMORY_BUCKET", "aerivon-live-agent-memory-1771792693")
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs in stories/ prefix
        blobs = bucket.list_blobs(prefix="stories/")
        
        stories = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                # Get story metadata
                story_id = blob.name.replace("stories/", "").replace(".json", "")
                stories.append({
                    "story_id": story_id,
                    "url": f"gs://{bucket_name}/{blob.name}",
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "size": blob.size
                })
        
        # Sort by story_id (which starts with timestamp) descending
        stories.sort(key=lambda s: s["story_id"], reverse=True)
        
        return {
            "success": True,
            "count": len(stories),
            "stories": stories
        }
        
    except Exception as e:
        print(f"[STORY LIST ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to list stories: {str(e)}")


@app.get("/agent/startup-check")
async def startup_check() -> dict[str, Any]:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    status = await _check_live_model_availability_fast(project, location)

    return {
        "project": project,
        "location": location,
        "live_models_available": status["live_models_available"],
        "live_models": status.get("live_models", []),
        "status": "ok" if status["live_models_available"] else "unavailable",
    }


@app.get("/agent/security-check")
async def security_check() -> dict[str, Any]:
    from agent import ALLOWED_TOOLS

    return {
        "status": "ok",
        "limits": {
            "MAX_MESSAGE_LENGTH": MAX_MESSAGE_LENGTH,
            "MAX_SESSION_RESULTS": MAX_SESSION_RESULTS,
            "RATE_LIMIT_SECONDS": RATE_LIMIT_SECONDS,
            "MAX_RESULT_SIZE": MAX_RESULT_SIZE,
        },
        "ssrf": {
            "blocked_hosts": list(BLOCKED_HOST_PATTERNS),
            "private_ip_regex": PRIVATE_IP_REGEX.pattern,
        },
        "prompt_injection": {
            "blocked_phrases": list(INJECTION_PATTERNS),
        },
        "tools": {
            "allowlist": sorted(ALLOWED_TOOLS),
            "max_tool_calls_per_turn": 6,
            "timeout_seconds": 30,
            "tool_output_wrapped_as_untrusted": True,
        },
    }


@app.post("/agent/speak")
async def post_agent_speak(payload: SpeakRequest) -> StreamingResponse:
    """Synthesize speech using Gemini Live API.

    Returns PCM16 audio at 24kHz for natural voice synthesis.
    """
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not use_vertex or not project:
        raise HTTPException(status_code=500, detail="Vertex AI not configured")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    def _narrate_live() -> bytes:
        """Use Gemini Live API to generate speech."""
        try:
            client = _make_genai_client(prefer_vertex=True, project=project, location=location)
            
            # Create Live session for speech synthesis
            session = client.live.connect(
                model="gemini-2.0-flash-exp",
                config=types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck"  # Natural conversational voice
                            )
                        )
                    ),
                ),
            )
            
            # Send text for narration
            session.send(text, end_of_turn=True)
            
            # Collect audio chunks
            audio_chunks = []
            for response in session.receive():
                if response.data:
                    audio_chunks.append(response.data)
                if response.server_content and response.server_content.turn_complete:
                    break
            
            if audio_chunks:
                return b"".join(audio_chunks)
            return b""
            
        except Exception as e:
            print(f"[SPEAK] Gemini Live error: {e}", file=sys.stderr)
            raise

    try:
        audio = await asyncio.to_thread(_narrate_live)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not audio:
        raise HTTPException(status_code=500, detail="No audio generated")

    async def body():
        yield audio

    return StreamingResponse(body(), media_type="audio/pcm;rate=24000")


@app.get("/agent/architecture")
async def architecture() -> dict[str, Any]:
    return {
        "agent": "Aerivon Live",
        "entrypoints": [
            {"method": "POST", "path": "/agent/message"},
            {"method": "POST", "path": "/agent/message-stream"},
            {"method": "POST", "path": "/agent/tool-result"},
            {"method": "WS", "path": "/ws/live"},
        ],
        "diagnostics": [
            {"method": "GET", "path": "/health"},
            {"method": "GET", "path": "/agent/startup-check"},
            {"method": "GET", "path": "/agent/security-check"},
            {"method": "GET", "path": "/agent/self-test"},
        ],
        "flow": [
            "Client POSTs /agent/message",
            "Agent uses Gemini Live if available; otherwise falls back to standard Gemini",
            "Model issues tool calls; agent validates allowlist + args + relevance",
            "Tools execute; results are wrapped as untrusted_data and sent back to model",
            "Final response returned to client",
        ],
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Realtime WS interface for Gemini Live.

    This endpoint is intentionally minimal and demo-focused.

    Client messages:
    - {"type":"audio","mime_type":"audio/pcm","data_b64":"..."}
    - {"type":"audio_end"}
    - {"type":"interrupt"}
    - {"type":"text","text":"..."}  (optional conditioning)

    Server messages:
    - {"type":"status","status":"connected"|"restarting",...,"session_id":N}
    - {"type":"audio_config","sample_rate":24000,"format":"pcm_s16le",...,"session_id":N}
    - {"type":"audio","mime_type":"audio/pcm","data_b64":"...","session_id":N}
    - {"type":"transcript","text":"...","finished":false|true,"session_id":N}
    - {"type":"interrupted","source":"client"|"upstream","session_id":N}
    - {"type":"turn_complete","session_id":N}
    - {"type":"error","error":"..."}
    """

    await websocket.accept()

    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    vertex_live_enabled = bool(use_vertex and project)
    # If Vertex Live isn't configured, this WS will fall back to standard generation.

    # Do NOT pre-probe models.list() here; it can hang. I'll attempt a Live connect and if that
    # fails it will fall back to standard generation.
    live_available: bool | None = None

    mode = (websocket.query_params.get("mode") or "agent").strip().lower()
    if mode not in {"agent", "stt"}:
        mode = "agent"

    output_mode = (websocket.query_params.get("output") or os.getenv("AERIVON_WS_OUTPUT") or "audio").strip().lower()
    if output_mode not in {"audio", "text"}:
        output_mode = "audio"

    # STT mode forces TEXT output.
    if mode == "stt":
        output_mode = "text"

    # This WS endpoint supports image messages; surface that explicitly so the UI
    # doesn't show "vision=undefined".
    vision_enabled = True

    # Persistent per-user memory (optional).
    user_id = _sanitize_user_id(websocket.query_params.get("user_id"))
    memory_scope = (websocket.query_params.get("memory_scope") or "").strip()
    memory_user_id = _memory_user_key(user_id=user_id, scope=memory_scope)
    user_memory: dict[str, Any] | None = None
    memory_prompt = ""
    if mode != "stt":
        user_memory = await _load_user_memory(user_id=memory_user_id)
        memory_prompt = _memory_to_prompt(user_memory)

    model = (os.getenv("AERIVON_LIVE_MODEL") or "gemini-2.0-flash-live-preview-04-09").strip()

    try:
        client = _make_genai_client(prefer_vertex=vertex_live_enabled, project=project, location=location)
    except Exception as exc:
        await websocket.send_json({"type": "error", "error": str(exc)})
        await websocket.close(code=1011)
        return

    voice_name = (websocket.query_params.get("voice") or os.getenv("AERIVON_LIVE_VOICE") or "").strip()
    voice_lang = (websocket.query_params.get("lang") or os.getenv("AERIVON_LIVE_VOICE_LANG") or "en-US").strip()
    speech_config: types.SpeechConfig | None = None
    if voice_name:
        speech_config = types.SpeechConfig(
            language_code=voice_lang,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            ),
        )

    if mode == "stt":
        system_instruction = (
            "You are a real-time speech-to-text transcriber. "
            "Transcribe ONLY what the user says. "
            "Output ONLY the transcript text. No commentary, no punctuation requirements. "
            "If you are unsure, output your best guess."
        )
    else:
        system_instruction = "You are Aerivon Live. Be concise and helpful."
        if memory_prompt:
            system_instruction = f"{system_instruction}\n\n{memory_prompt}"

    def _build_live_config(response_modalities: list[types.Modality]) -> types.LiveConnectConfig:
        """Best-effort include generation_config (max tokens, temperature) for Live.

        Some google-genai versions may not expose generation_config on LiveConnectConfig.
        In that case, fall back to the minimal config rather than crashing.
        """

        gen_cfg = types.GenerationConfig(
            max_output_tokens=AERIVON_LIVE_MAX_OUTPUT_TOKENS,
            temperature=AERIVON_LIVE_TEMPERATURE,
        )

        base_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_modalities": response_modalities,
        }
        if response_modalities == [types.Modality.AUDIO]:
            base_kwargs["speech_config"] = speech_config
            base_kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()

        # Try with generation_config; if unsupported, retry without it.
        try:
            return types.LiveConnectConfig(**base_kwargs, generation_config=gen_cfg)
        except TypeError:
            return types.LiveConnectConfig(**base_kwargs)

    if output_mode == "audio":
        session_config = _build_live_config([types.Modality.AUDIO])
    else:
        session_config = _build_live_config([types.Modality.TEXT])

    try:
        from websockets.exceptions import ConnectionClosed  # type: ignore
    except Exception:  # pragma: no cover
        ConnectionClosed = ()  # type: ignore

    session_seq = 0

    def _is_ws_closed_error(exc: Exception) -> bool:
        if isinstance(exc, WebSocketDisconnect):
            return True
        msg = str(exc)
        return (
            "Cannot call \"send\" once a close message has been sent." in msg
            or "Unexpected ASGI message 'websocket.send'" in msg
            or "Unexpected ASGI message 'websocket.close'" in msg
            or "after sending 'websocket.close'" in msg
            or "response already completed" in msg
        )

    def _pcm_s16le_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
        import struct

        # Minimal RIFF/WAVE header for PCM s16le.
        byte_rate = sample_rate * channels * 2
        block_align = channels * 2
        data_size = len(pcm)
        riff_size = 36 + data_size
        return b"".join(
            [
                b"RIFF",
                struct.pack("<I", riff_size),
                b"WAVE",
                b"fmt ",
                struct.pack("<I", 16),  # PCM fmt chunk size
                struct.pack("<H", 1),  # audio format = PCM
                struct.pack("<H", channels),
                struct.pack("<I", sample_rate),
                struct.pack("<I", byte_rate),
                struct.pack("<H", block_align),
                struct.pack("<H", 16),  # bits per sample
                b"data",
                struct.pack("<I", data_size),
                pcm,
            ]
        )

    async def run_fallback_session() -> None:
        """Non-Live fallback over the same WS shape (text responses only)."""

        nonlocal session_seq
        session_seq += 1
        session_id = session_seq

        async def ws_send(payload: dict[str, Any]) -> None:
            payload.setdefault("session_id", session_id)
            try:
                await websocket.send_json(payload)
            except Exception as exc:
                if _is_ws_closed_error(exc):
                    raise WebSocketDisconnect(code=1006)
                raise

        # Pick a standard model. If there's no Vertex project, skip model listing.
        preferred = os.getenv(
            "AERIVON_WS_FALLBACK_MODEL",
            os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash"),
        )
        fallback_model = resolve_fallback_model(project, location, preferred) if project else preferred

        await ws_send(
            {
                "type": "status",
                "status": "connected",
                "model": fallback_model,
                "vision": True,
                "output": "text",
                "mode": "fallback",
                "detail": "Gemini Live unavailable; using standard generate_content",
                "user_id": user_id,
                "memory_scope": memory_scope or None,
            }
        )

        # State for mic buffering.
        audio_pcm = bytearray()
        last_text_prompt = ""

        def gen_cfg() -> types.GenerateContentConfig:
            return types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=AERIVON_LIVE_MAX_OUTPUT_TOKENS,
                temperature=AERIVON_LIVE_TEMPERATURE,
            )

        async def _persist_exchange(*, user_text: str, model_text: str) -> None:
            nonlocal user_memory
            if not AERIVON_MEMORY_BUCKET:
                return
            if mode == "stt":
                return

            mem = user_memory if isinstance(user_memory, dict) else {}
            exchanges = mem.get("exchanges")
            if not isinstance(exchanges, list):
                exchanges = []

            exchanges.append(
                {
                    "t": int(time.time()),
                    "user": (user_text or "").strip()[:2000],
                    "model": (model_text or "").strip()[:4000],
                }
            )
            exchanges = exchanges[-AERIVON_MEMORY_MAX_EXCHANGES :]
            mem["user_id"] = user_id
            mem["memory_user_id"] = memory_user_id
            mem["updated_at"] = int(time.time())
            mem["exchanges"] = exchanges

            # Cheap summary (no extra model call): truncate concatenation.
            joined = " ".join(
                [f"U:{ex.get('user','')} M:{ex.get('model','')}" for ex in exchanges if isinstance(ex, dict)]
            )
            mem["summary"] = joined[:1200]

            user_memory = mem
            await _save_user_memory(user_id=memory_user_id, memory=mem)

        async def generate_and_send(parts: list[types.Part], *, user_text_for_memory: str) -> None:
            # Run sync generate_content in a thread so the event loop stays responsive.
            def _run() -> str:
                resp = client.models.generate_content(
                    model=fallback_model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=gen_cfg(),
                )
                cand = resp.candidates[0] if resp.candidates else None
                out_parts = cand.content.parts if cand and cand.content else []
                return "".join([p.text for p in out_parts if getattr(p, "text", None)])

            text = ""
            try:
                text = await asyncio.to_thread(_run)
                if text:
                    await ws_send({"type": "text", "text": text})
            except Exception as exc:
                await ws_send({"type": "error", "error": str(exc)})
            finally:
                if text:
                    await _persist_exchange(user_text=user_text_for_memory, model_text=text)
                await ws_send({"type": "turn_complete"})

        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                continue

            msg_type = str(data.get("type") or "").strip().lower()

            if msg_type == "interrupt":
                audio_pcm.clear()
                await ws_send({"type": "interrupted", "source": "client"})
                await ws_send({"type": "turn_complete"})
                await ws_send(
                    {
                        "type": "status",
                        "status": "connected",
                        "model": fallback_model,
                        "vision": True,
                        "output": "text",
                        "mode": "fallback",
                        "detail": "ready_after_interrupt",
                        "user_id": user_id,
                        "memory_scope": memory_scope or None,
                    }
                )
                continue

            if msg_type == "text":
                last_text_prompt = str(data.get("text") or "")
                # Generate immediately for text-only turns.
                if last_text_prompt.strip():
                    await generate_and_send(
                        [types.Part.from_text(text=last_text_prompt)],
                        user_text_for_memory=last_text_prompt,
                    )
                continue

            if msg_type == "image":
                mime_type = str(data.get("mime_type") or "image/png")
                b64 = str(data.get("data_b64") or "")
                if not b64:
                    await ws_send({"type": "error", "error": "missing data_b64"})
                    continue
                try:
                    img = base64.b64decode(b64, validate=True)
                except Exception:
                    await ws_send({"type": "error", "error": "invalid base64"})
                    continue

                prompt = str(data.get("text") or "").strip() or last_text_prompt.strip() or "Describe the image."
                await generate_and_send(
                    [
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=img, mime_type=mime_type),
                    ],
                    user_text_for_memory=prompt,
                )
                continue

            if msg_type == "audio":
                # Buffer PCM until audio_end.
                b64 = str(data.get("data_b64") or "")
                if not b64:
                    continue
                try:
                    chunk = base64.b64decode(b64, validate=True)
                except Exception:
                    continue
                # Keep a simple cap (~6s of 16kHz mono s16 = 192KB/sec). Allow ~2MB.
                if len(audio_pcm) + len(chunk) <= 2 * 1024 * 1024:
                    audio_pcm.extend(chunk)
                continue

            if msg_type == "audio_end":
                if not audio_pcm:
                    await ws_send({"type": "turn_complete"})
                    continue
                wav = _pcm_s16le_to_wav(bytes(audio_pcm), sample_rate=16000, channels=1)
                audio_pcm.clear()
                prompt = last_text_prompt.strip() or "Transcribe and respond to the user's audio."
                await generate_and_send(
                    [
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                    ],
                    user_text_for_memory="(voice message)",
                )
                continue

            # Ignore unknown message types.


    async def run_one_session() -> bool:
        """Return True to restart (interrupt/upstream drop), False to stop."""
        nonlocal session_seq, user_memory
        session_seq += 1
        session_id = session_seq

        # Reload memory from GCS before starting each session so restarts pick up saved context.
        if mode != "stt":
            user_memory = await _load_user_memory(user_id=memory_user_id)
            memory_prompt = _memory_to_prompt(user_memory)
            sys_instr = "You are Aerivon Live. Be concise and helpful."
            if memory_prompt:
                sys_instr = (
                    f"{sys_instr}\n\n{memory_prompt}\n\n"
                    "IMPORTANT: This is a continuing conversation. The user's voice input was transcribed as '(voice message)' above, "
                    "but your PREVIOUS responses reveal what they asked. Use your past responses to infer the conversation context. "
                    "If asked what they said before, reconstruct it from your own previous answers."
                )
            
            # Rebuild session config with fresh memory.
            def _build_config(response_modalities: list[types.Modality]) -> types.LiveConnectConfig:
                gen_cfg = types.GenerationConfig(
                    max_output_tokens=AERIVON_LIVE_MAX_OUTPUT_TOKENS,
                    temperature=AERIVON_LIVE_TEMPERATURE,
                )
                base_kwargs: dict[str, Any] = {
                    "system_instruction": sys_instr,
                    "response_modalities": response_modalities,
                }
                if response_modalities == [types.Modality.AUDIO]:
                    base_kwargs["speech_config"] = speech_config
                    base_kwargs["output_audio_transcription"] = types.AudioTranscriptionConfig()
                try:
                    return types.LiveConnectConfig(**base_kwargs, generation_config=gen_cfg)
                except TypeError:
                    return types.LiveConnectConfig(**base_kwargs)
            
            current_session_config = _build_config([types.Modality.AUDIO] if output_mode == "audio" else [types.Modality.TEXT])
        else:
            current_session_config = session_config

        async def ws_send(payload: dict[str, Any]) -> None:
            payload.setdefault("session_id", session_id)
            try:
                await websocket.send_json(payload)
            except Exception as exc:
                if _is_ws_closed_error(exc):
                    raise WebSocketDisconnect(code=1006)
                raise

        # Turn buffers for persistent memory.
        last_user_for_memory: str = ""
        model_text_parts: list[str] = []

        async def persist_exchange_if_any() -> None:
            nonlocal user_memory
            if not AERIVON_MEMORY_BUCKET:
                return
            if mode == "stt":
                return
            u = (last_user_for_memory or "").strip()
            m = "".join(model_text_parts).strip()
            if not u and not m:
                return

            mem = user_memory if isinstance(user_memory, dict) else {}
            exchanges = mem.get("exchanges")
            if not isinstance(exchanges, list):
                exchanges = []
            exchanges.append({"t": int(time.time()), "user": u[:2000], "model": m[:4000]})
            exchanges = exchanges[-AERIVON_MEMORY_MAX_EXCHANGES :]
            mem["user_id"] = user_id
            mem["memory_user_id"] = memory_user_id
            mem["updated_at"] = int(time.time())
            mem["exchanges"] = exchanges
            joined = " ".join(
                [f"U:{ex.get('user','')} M:{ex.get('model','')}" for ex in exchanges if isinstance(ex, dict)]
            )
            mem["summary"] = joined[:1200]
            user_memory = mem
            await _save_user_memory(user_id=memory_user_id, memory=mem)

        async with client.aio.live.connect(model=model, config=current_session_config) as stream:
            await ws_send(
                {
                    "type": "status",
                    "status": "connected",
                    "model": model,
                    "vision": vision_enabled,
                    "output": output_mode,
                    "mode": mode,
                    "user_id": user_id,
                    "memory_scope": memory_scope or None,
                    "audio_config": (
                        {
                            "sample_rate": DEFAULT_LIVE_AUDIO_SAMPLE_RATE,
                            "format": "pcm_s16le",
                            "channels": 1,
                            "mime_type": "audio/pcm",
                        }
                        if output_mode == "audio"
                        else None
                    ),
                }
            )
            if output_mode == "audio":
                await ws_send(
                    {
                        "type": "audio_config",
                        "sample_rate": DEFAULT_LIVE_AUDIO_SAMPLE_RATE,
                        "format": "pcm_s16le",
                        "channels": 1,
                        "mime_type": "audio/pcm",
                    }
                )

            async def recv_loop() -> None:
                async for msg in stream.receive():
                    data_bytes = getattr(msg, "data", None)
                    if isinstance(data_bytes, (bytes, bytearray)) and data_bytes:
                        await ws_send(
                            {
                                "type": "audio",
                                "mime_type": "audio/pcm",
                                "data_b64": base64.b64encode(bytes(data_bytes)).decode("ascii"),
                            }
                        )

                    if output_mode == "text" and getattr(msg, "text", None):
                        if mode != "stt" and msg.text:
                            model_text_parts.append(str(msg.text))
                        await ws_send({"type": "text", "text": msg.text})
                        continue

                    sc = getattr(msg, "server_content", None)
                    if sc is not None and getattr(sc, "interrupted", None) is True:
                        await ws_send({"type": "interrupted", "source": "upstream"})

                    if sc is not None:
                        # Some responses (notably vision) deliver text via model_turn parts.
                        model_turn = getattr(sc, "model_turn", None)
                        parts = getattr(model_turn, "parts", None) if model_turn is not None else None
                        if parts:
                            for part in parts:
                                part_text = getattr(part, "text", None)
                                if part_text:
                                    if mode != "stt":
                                        model_text_parts.append(str(part_text))
                                    await ws_send({"type": "text", "text": part_text})

                        otx = getattr(sc, "output_transcription", None)
                        if otx is not None:
                            tx_text = getattr(otx, "text", None)
                            tx_finished = getattr(otx, "finished", None)
                            if tx_text is not None or tx_finished is not None:
                                if mode != "stt" and tx_text:
                                    model_text_parts.append(str(tx_text))
                                await ws_send(
                                    {
                                        "type": "transcript",
                                        "text": tx_text,
                                        "finished": bool(tx_finished) if tx_finished is not None else False,
                                    }
                                )

                    if sc is not None and getattr(sc, "turn_complete", None) is True:
                        await persist_exchange_if_any()
                        model_text_parts.clear()
                        last_user_for_memory = ""
                        await ws_send({"type": "turn_complete"})

            recv_task = asyncio.create_task(recv_loop())

            async def restart(reason: str, detail: str = "") -> bool:
                # Save any partial exchange BEFORE restarting to preserve memory across upstream disconnects.
                if reason != "client_interrupt":
                    try:
                        await persist_exchange_if_any()
                    except Exception:
                        pass  # Don't block restart on memory save failure
                
                await ws_send(
                    {
                        "type": "status",
                        "status": "restarting",
                        "reason": reason,
                        "detail": detail,
                        "model": model,
                        "vision": vision_enabled,
                        "output": output_mode,
                        "mode": mode,
                    }
                )
                return True

            async def safe_send_realtime_input(**kwargs: Any) -> bool:
                try:
                    await stream.send_realtime_input(**kwargs)
                    return False
                except ConnectionClosed as exc:  # type: ignore[misc]
                    return await restart("upstream_disconnected", str(exc))

            async def safe_send_client_content(*, turns: types.Content, turn_complete: bool = True) -> bool:
                try:
                    await stream.send_client_content(turns=turns, turn_complete=turn_complete)
                    return False
                except ConnectionClosed as exc:  # type: ignore[misc]
                    return await restart("upstream_disconnected", str(exc))

            try:
                while True:
                    # Wake periodically so upstream disconnects trigger a restart.
                    try:
                        data = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if recv_task.done():
                            exc: Exception | None
                            try:
                                exc = recv_task.exception()
                            except Exception:
                                exc = None
                            return await restart("upstream_disconnected", str(exc) if exc else "")
                        continue

                    if not isinstance(data, dict):
                        continue

                    msg_type = str(data.get("type") or "").strip().lower()
                    if msg_type == "interrupt":
                        # Deterministic: notify client immediately, stop forwarding old audio, then reconnect.
                        try:
                            await ws_send({"type": "interrupted", "source": "client"})
                        except Exception:
                            pass
                        recv_task.cancel()
                        return await restart("client_interrupt")

                    if msg_type == "text":
                        text = str(data.get("text") or "")
                        if len(text) > MAX_MESSAGE_LENGTH:
                            await ws_send({"type": "error", "error": "text too long"})
                            continue
                        # In audio output mode, client_content turns are the most reliable way
                        # to trigger generation (send_realtime_input(text=...) may not yield audio).
                        if output_mode == "audio" and mode != "stt":
                            parts = [types.Part.from_text(text=text)]
                            last_user_for_memory = text
                            if await safe_send_client_content(
                                turns=types.Content(role="user", parts=parts),
                                turn_complete=True,
                            ):
                                return True
                        else:
                            last_user_for_memory = text
                            if await safe_send_realtime_input(text=text):
                                return True
                        continue

                    if msg_type == "audio":
                        mime_type = str(data.get("mime_type") or "audio/pcm")
                        b64 = str(data.get("data_b64") or "")
                        if not b64:
                            await ws_send({"type": "error", "error": "missing data_b64"})
                            continue
                        if len(b64) > (MAX_WS_MESSAGE_BYTES * 2):
                            await ws_send({"type": "error", "error": "audio chunk too large"})
                            continue
                        try:
                            chunk = base64.b64decode(b64, validate=True)
                        except Exception:
                            await ws_send({"type": "error", "error": "invalid base64"})
                            continue
                        if len(chunk) > MAX_WS_MESSAGE_BYTES:
                            await ws_send({"type": "error", "error": "audio chunk too large"})
                            continue
                        if await safe_send_realtime_input(audio=types.Blob(data=chunk, mime_type=mime_type)):
                            return True
                        continue

                    if msg_type == "audio_end":
                        if not last_user_for_memory:
                            last_user_for_memory = "(voice message)"
                        if await safe_send_realtime_input(audio_stream_end=True):
                            return True
                        continue

                    if msg_type == "image":
                        mime_type = str(data.get("mime_type") or "image/png")
                        if not mime_type.lower().startswith("image/"):
                            await ws_send({"type": "error", "error": "invalid image mime_type"})
                            continue
                        b64 = str(data.get("data_b64") or "")
                        if not b64:
                            await ws_send({"type": "error", "error": "missing data_b64"})
                            continue
                        if len(b64) > (MAX_WS_MESSAGE_BYTES * 2):
                            await ws_send({"type": "error", "error": "image chunk too large"})
                            continue
                        try:
                            chunk = base64.b64decode(b64, validate=True)
                        except Exception:
                            await ws_send({"type": "error", "error": "invalid base64"})
                            continue
                        if len(chunk) > MAX_WS_MESSAGE_BYTES:
                            await ws_send({"type": "error", "error": "image chunk too large"})
                            continue

                        prompt = str(data.get("text") or "")
                        parts: list[types.Part] = []
                        if prompt:
                            parts.append(types.Part.from_text(text=prompt))
                            last_user_for_memory = prompt
                        parts.append(types.Part.from_bytes(data=chunk, mime_type=mime_type))

                        if not last_user_for_memory:
                            last_user_for_memory = "(image)"

                        if await safe_send_client_content(turns=types.Content(role="user", parts=parts)):
                            return True
                        continue

            finally:
                try:
                    await asyncio.wait_for(recv_task, timeout=1.0)
                except asyncio.CancelledError:
                    # Python 3.14: CancelledError may not be caught by Exception.
                    pass
                except Exception:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

    try:
        # Without Vertex Live, don't attempt live.connect (it will fail with API-key clients).
        if not vertex_live_enabled:
            await run_fallback_session()
            return

        while True:
            try:
                should_restart = await run_one_session()
            except WebSocketDisconnect:
                return
            except Exception:
                # Live connect/runtime failed; fall back to standard generation so the client
                # still gets a response.
                await run_fallback_session()
                return

            if not should_restart:
                break
    except WebSocketDisconnect:
        return
    except Exception as exc:
        # Best effort: report the error then close.
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.get("/agent/self-test")
async def self_test() -> dict[str, Any]:
    from agent import ALLOWED_TOOLS, validate_tool_args
    from tools import is_safe_url

    results: list[dict[str, Any]] = []

    def add_test(name: str, passed: bool, detail: str = "") -> None:
        results.append({"name": name, "passed": passed, "detail": detail})

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    status = check_live_model_availability(project, location)
    add_test(
        "startup_check_runs",
        isinstance(status.get("live_models_available"), bool),
        f"live_models_available={status.get('live_models_available')}",
    )

    required_tools = {
        "browse_url",
        "scrape_leads",
        "extract_page_content",
        "take_screenshot",
        "generate_outreach_message",
    }
    add_test(
        "tool_allowlist_present",
        required_tools.issubset(ALLOWED_TOOLS),
        f"missing={sorted(required_tools - set(ALLOWED_TOOLS))}",
    )

    add_test(
        "ssrf_block_metadata",
        is_safe_url("http://metadata.google.internal") is False,
        "is_safe_url(metadata.google.internal)",
    )
    add_test(
        "ssrf_block_localhost",
        is_safe_url("http://localhost:8080") is False,
        "is_safe_url(localhost)",
    )
    add_test(
        "agent_arg_validation",
        validate_tool_args("browse_url", {"url": "file:///etc/passwd"})[0] is False,
        "browse_url blocks file://",
    )

    add_test(
        "api_limits_configured",
        MAX_MESSAGE_LENGTH <= 4000 and MAX_SESSION_RESULTS <= 100 and MAX_RESULT_SIZE <= 20000,
        f"MAX_MESSAGE_LENGTH={MAX_MESSAGE_LENGTH} MAX_SESSION_RESULTS={MAX_SESSION_RESULTS} MAX_RESULT_SIZE={MAX_RESULT_SIZE}",
    )

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    return {
        "agent": "Aerivon Live",
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "tests": results,
    }


@app.post("/agent/message", response_model=AgentMessageResponse)
async def post_agent_message(payload: AgentMessageRequest, request: Request) -> AgentMessageResponse:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    last_seen = LAST_REQUEST_TIME.get(client_ip)
    if last_seen is not None and now - last_seen < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    LAST_REQUEST_TIME[client_ip] = now

    if len(payload.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=413, detail="Message too long")

    lowered = payload.message.lower()
    if any(pattern in lowered for pattern in INJECTION_PATTERNS):
        raise HTTPException(status_code=400, detail="Message rejected by security policy")

    if _contains_unsafe_target(payload.message):
        raise HTTPException(status_code=400, detail="Blocked unsafe host")

    user_id = _sanitize_user_id(payload.user_id or client_ip)
    session_id = user_id
    SESSION_TOOL_RESULTS.setdefault(session_id, [])

    user_memory = await _load_user_memory(user_id=user_id)
    memory_prompt = _memory_to_prompt(user_memory)
    message = payload.message
    if memory_prompt:
        message = f"{memory_prompt}\n\nUser: {payload.message}".strip()

    try:
        turn = await _get_agent().process_message(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    SESSION_TOOL_RESULTS[session_id].extend(turn.tool_calls)
    SESSION_TOOL_RESULTS[session_id] = SESSION_TOOL_RESULTS[session_id][-MAX_SESSION_RESULTS:]

    await _append_exchange_to_memory(user_id=user_id, user_text=payload.message, model_text=turn.response_text)

    return AgentMessageResponse(
        response=turn.response_text,
        tool_calls=turn.tool_calls,
    )


@app.post("/agent/message-stream")
async def post_agent_message_stream(payload: AgentMessageRequest, request: Request) -> StreamingResponse:
    """Stream a text response via Server-Sent Events (SSE).

    This is designed for hackathon demos where the client wants streaming text output
    and interruption via starting a new stream for the same user_id.
    """

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    last_seen = LAST_REQUEST_TIME.get(client_ip)
    if last_seen is not None and now - last_seen < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    LAST_REQUEST_TIME[client_ip] = now

    if len(payload.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=413, detail="Message too long")

    lowered = payload.message.lower()
    if any(pattern in lowered for pattern in INJECTION_PATTERNS):
        raise HTTPException(status_code=400, detail="Message rejected by security policy")

    if _contains_unsafe_target(payload.message):
        raise HTTPException(status_code=400, detail="Blocked unsafe host")

    user_id = _sanitize_user_id(payload.user_id or client_ip)

    # Server-side interruption: cancel any prior stream for this user.
    prev = ACTIVE_SSE_CANCEL.get(user_id)
    if prev is not None:
        prev.set()
    cancel_event = asyncio.Event()
    ACTIVE_SSE_CANCEL[user_id] = cancel_event

    async def event_iter():
        def sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        user_memory = await _load_user_memory(user_id=user_id)
        memory_prompt = _memory_to_prompt(user_memory)

        system_instruction = "You are Aerivon Live. Be concise and helpful."
        if memory_prompt:
            system_instruction = f"{system_instruction}\n\n{memory_prompt}"

        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        prefer_vertex = bool(use_vertex and project)

        # For SSE text streaming, use standard generate_content_stream (not Live).
        preferred_model = (
            os.getenv("AERIVON_SSE_MODEL")
            or os.getenv("GEMINI_FALLBACK_MODEL")
            or os.getenv("AERIVON_WS_FALLBACK_MODEL")
            or "gemini-2.5-flash"
        ).strip()
        model = resolve_fallback_model(project, location, preferred_model) if project else preferred_model

        try:
            client = _make_genai_client(prefer_vertex=prefer_vertex, project=project, location=location)
        except Exception as exc:
            yield sse("error", {"type": "error", "error": str(exc)})
            yield sse("done", {"type": "done"})
            return

        # Yield an initial status event so the client can update UI immediately.
        yield sse(
            "status",
            {
                "type": "status",
                "status": "connected",
                "user_id": user_id,
                "model": model,
            },
        )

        text_parts: list[str] = []
        interrupted = False

        import threading

        stop_flag = threading.Event()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()

        def _extract_text_from_response(resp: Any) -> str:
            try:
                cands = getattr(resp, "candidates", None) or []
                cand = cands[0] if cands else None
                content = getattr(cand, "content", None) if cand is not None else None
                parts = getattr(content, "parts", None) if content is not None else None
                if not parts:
                    return ""
                return "".join([p.text for p in parts if getattr(p, "text", None)])
            except Exception:
                return ""

        def _run_stream() -> None:
            try:
                stream_fn = getattr(client.models, "generate_content_stream", None)
                if stream_fn is None:
                    # No streaming API available; fall back to one-shot.
                    resp = client.models.generate_content(
                        model=model,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=payload.message)])],
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            max_output_tokens=AERIVON_LIVE_MAX_OUTPUT_TOKENS,
                            temperature=AERIVON_LIVE_TEMPERATURE,
                        ),
                    )
                    text = _extract_text_from_response(resp)
                    if text:
                        asyncio.run_coroutine_threadsafe(q.put(text), loop)
                    return

                for resp in stream_fn(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=payload.message)])],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=AERIVON_LIVE_MAX_OUTPUT_TOKENS,
                        temperature=AERIVON_LIVE_TEMPERATURE,
                    ),
                ):
                    if stop_flag.is_set():
                        break
                    text = _extract_text_from_response(resp)
                    if text:
                        asyncio.run_coroutine_threadsafe(q.put(text), loop)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)

        prod_task = asyncio.create_task(asyncio.to_thread(_run_stream))

        try:
            try:
                while True:
                    if cancel_event.is_set():
                        stop_flag.set()
                        interrupted = True
                        yield sse("interrupted", {"type": "interrupted", "source": "new_request"})
                        break

                    item = await q.get()
                    if item is None:
                        break
                    text_parts.append(item)
                    yield sse("text", {"type": "text", "text": item})
            finally:
                stop_flag.set()
                try:
                    await asyncio.wait_for(prod_task, timeout=1.0)
                except Exception:
                    pass
        except Exception as exc:
            yield sse("error", {"type": "error", "error": str(exc)})
        finally:
            # Only persist full exchange if it completed normally.
            if not interrupted:
                await _append_exchange_to_memory(
                    user_id=user_id,
                    user_text=payload.message,
                    model_text="".join(text_parts),
                )

            # Clean up cancel registry if this is still the active stream.
            current = ACTIVE_SSE_CANCEL.get(user_id)
            if current is cancel_event:
                ACTIVE_SSE_CANCEL.pop(user_id, None)

            yield sse("done", {"type": "done"})

    return StreamingResponse(event_iter(), media_type="text/event-stream")


@app.post("/agent/tool-result")
async def post_agent_tool_result(payload: ToolResultRequest) -> dict[str, Any]:
    encoded_result = json.dumps(payload.result, ensure_ascii=False)
    if len(encoded_result) > MAX_RESULT_SIZE:
        raise HTTPException(status_code=413, detail="Tool result exceeds max size")

    if payload.session_id not in SESSION_TOOL_RESULTS:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    stored = {
        "id": payload.tool_call_id,
        "name": payload.tool_name,
        "result": payload.result,
        "source": "external",
    }
    SESSION_TOOL_RESULTS[payload.session_id].append(stored)
    SESSION_TOOL_RESULTS[payload.session_id] = SESSION_TOOL_RESULTS[payload.session_id][
        -MAX_SESSION_RESULTS:
    ]
    return {
        "ok": True,
        "session_id": payload.session_id,
        "stored_tool_results": len(SESSION_TOOL_RESULTS[payload.session_id]),
    }
