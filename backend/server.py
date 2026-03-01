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
import inspect

# ============ STEP 0: BUILD FINGERPRINT (MODULE IMPORT TIME) ============
# This MUST appear in logs if this module is loaded
BUILD_MARKER = f"AERIVON_BUILD_MARKER::{time.time()}::{os.getpid()}"
print(f"\n{'='*80}", file=sys.stderr)
print(BUILD_MARKER, file=sys.stderr)
sys.stderr.flush()

# Also print the absolute file path & a hash of the source
THIS_FILE = os.path.abspath(__file__)
try:
    with open(__file__, "rb") as f:
        SRC = f.read()
    SRC_SHA = hashlib.sha256(SRC).hexdigest()[:12]
except Exception as e:
    SRC_SHA = f"ERR:{e}"
print(f"AERIVON_FILE::{THIS_FILE}::SHA::{SRC_SHA}", file=sys.stderr)
print(f"{'='*80}\n", file=sys.stderr)
sys.stderr.flush()


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
UI_MODEL = os.getenv("AERIVON_UI_MODEL", "gemini-3-flash-preview").strip()
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
            doc = client.collection(AERIVON_FIRESTORE_COLLECTION).document(user_id).get()  # Sync call in thread
            if not doc.exists:  # type: ignore - doc is synchronous DocumentSnapshot
                return None
            data = doc.to_dict() or {}  # type: ignore - doc is synchronous DocumentSnapshot
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


def _pcm_s16le_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Convert raw PCM s16le audio to WAV format."""
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
        from PIL import Image, ImageDraw
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
            print("[UI NAV DEBUG] Saved annotated screenshot to /tmp/aerivon_annotated_debug.png", file=__import__("sys").stderr, flush=True)
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


def _ui_action_prompt(task: str, memory: list[str], clickable_elements: list[dict] | None = None) -> str:
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
    
    # Detect if this is an analysis/extraction task vs interaction task
    task_lower = task.lower()
    analysis_keywords = [
        "find", "what is", "what's", "analyze", "identify", "read", 
        "show me", "get the", "extract", "tell me", "hero text", "heading",
        "largest text", "main text", "title"
    ]
    is_analysis = any(keyword in task_lower for keyword in analysis_keywords)
    
    if is_analysis:
        # Analysis mode: extract information and return in note field
        return (
            "You are Aerivon UI Navigator - an AI that analyzes webpages.\n\n"
            "The user wants you to ANALYZE the screenshot and EXTRACT specific information.\n\n"
            "Your task: Look at the screenshot and answer the user's question directly.\n\n"
            "Return ONLY valid JSON (no markdown) matching this schema:\n"
            "{\n"
            "  \"actions\": [],\n"
            "  \"done\": true,\n"
            "  \"note\": \"Your analysis result - the actual text/information the user requested\"\n"
            "}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- The 'note' field should contain your direct answer to the user's request\n"
            "- If looking for hero text/heading: identify the LARGEST, MOST PROMINENT text on the page\n"
            "- Quote the exact text you find, don't paraphrase\n"
            "- Be specific and concise\n"
            "- Set 'actions' to empty array []\n"
            "- Set 'done' to true\n\n"
            "Examples:\n"
            "  Good note: \"The hero text says: 'MOVE THE WORLD'\"\n"
            "  Good note: \"The main heading is 'Welcome to Nike'\"\n"
            "  Good note: \"The largest text at the top reads: 'Just Do It'\"\n"
            "  Bad note: \"I cannot see the text\" (you CAN see it in the screenshot)\n"
            "  Bad note: \"The page has loaded\" (tell them what you found, not just status)\n\n"
            f"Context:\n{context}\n\n"
            f"User's request: {task}\n\n"
            "Analyze the screenshot NOW and extract the requested information."
        )
    else:
        # Interactive mode: perform browser actions
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
        contents=contents,  # type: ignore - SDK accepts list[Content]
        config=cfg,
    )
    
    if not resp.candidates:
        return {"error": "Gemini returned no candidates"}
    
    candidate = resp.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return {"error": "Gemini returned no content"}
    
    parts = candidate.content.parts
    text = "".join(p.text for p in parts if p.text)
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
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not use_vertex or not project:
        await websocket.send_json({"type": "error", "error": "Vertex not enabled"})
        await websocket.close(code=1011)
        return

    gen_client = _make_genai_client(prefer_vertex=True, project=project, location=location)

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

            async def execute_action(action: dict[str, Any], clickable_elements: list[dict] | None = None) -> dict[str, Any]:
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
                        idx = int(action.get("element_index") or 0)
                        if not clickable_elements or idx < 0 or idx >= len(clickable_elements):
                            return {"ok": False, "error": f"Invalid element_index: {idx}", "max_index": len(clickable_elements or []) - 1}
                        
                        el = clickable_elements[idx]
                        x = el['x'] + el['w'] // 2
                        y = el['y'] + el['h'] // 2
                        print(f"[UI NAV DEBUG] Using element_index {idx}: {el.get('tag')} \"{el.get('text', '')}\" at ({x},{y})", file=__import__("sys").stderr, flush=True)
                    else:
                        # Fallback to x,y coordinates
                        x = int(action.get("x") or 0)
                        y = int(action.get("y") or 0)
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
                            element = element_handle.as_element()
                            if element:
                                await element.click()
                            else:
                                # Fallback to mouse click
                                await page.mouse.click(x, y)
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
                model="gemini-3-flash-preview",
                config=types.LiveConnectConfig(
                    response_modalities=[types.Modality.AUDIO],
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
                
                # Fallback to Imagen 3 for legacy image generation
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

    await send({"type": "status", "status": "connected", "model": "gemini-3-pro-image-preview"})

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
                "For each scene in the story:\n"
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
                    print("[STORY DEBUG] Calling Gemini with model: gemini-3-pro-image-preview", file=sys.stderr)
                    try:
                        resp = gen_client.models.generate_content(
                            model="gemini-3-pro-image-preview",
                            contents=story_prompt,
                            config=types.GenerateContentConfig(
                                response_modalities=["TEXT", "IMAGE"],
                                temperature=0.9,
                                max_output_tokens=4096,
                            ),
                        )
                        print("[STORY DEBUG] Gemini response received successfully", file=sys.stderr)
                    except Exception as gemini_error:
                        print(f"[STORY ERROR] Gemini API call failed: {type(gemini_error).__name__}: {gemini_error}", file=sys.stderr)
                        raise
                    
                    # Extract parts from multimodal response
                    parts = []
                    if resp.candidates:
                        for candidate in resp.candidates:
                            if candidate.content and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if part.text:
                                        parts.append({"kind": "text", "text": part.text})
                                    elif part.inline_data and part.inline_data.data:
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

    async def _narrate_live() -> bytes:
        """Use Gemini Live API to generate speech."""
        try:
            client = _make_genai_client(prefer_vertex=True, project=project, location=location)
            
            # Create Live session for speech synthesis (async only)
            async with client.aio.live.connect(
                model="gemini-3-flash-preview",
                config=types.LiveConnectConfig(
                    response_modalities=[types.Modality.AUDIO],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck"  # Natural conversational voice
                            )
                        )
                    ),
                ),
            ) as session:
                # Send text for narration
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=text)]
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
                    return b"".join(audio_chunks)
                return b""
            
        except Exception as e:
            print(f"[SPEAK] Gemini Live error: {e}", file=sys.stderr)
            raise

    try:
        audio = await _narrate_live()
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

    model = (os.getenv("AERIVON_LIVE_MODEL") or "gemini-3-flash-preview").strip()

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
            os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3-flash-preview"),
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
                out_parts = cand.content.parts if (cand and cand.content and cand.content.parts) else []
                return "".join([p.text for p in out_parts if p.text])

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


@app.websocket("/ws/aerivon")
async def ws_aerivon_unified(websocket: WebSocket) -> None:
    """Unified Aerivon Agent - Orchestrates Live, UI, and Story capabilities with intent detection."""
    
    # ============ ABSOLUTE FIRST ACTION: WRITE TO FILE ============
    import traceback as tb
    try:
        with open("/tmp/aerivon_ENTRY_IMMEDIATE.txt", "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ENTRY AT {time.time()}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write(f"File: {__file__}\n")
            for line in tb.format_stack():
                f.write(line)
            f.write(f"{'='*80}\n")
    except:
        pass
    
    # ============ STEP 1: CONFIRM HANDLER FUNCTION ============
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"AERIVON_WS_ENTRY {__file__} PID {os.getpid()} FUNC {inspect.currentframe().f_code.co_name}", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)
    sys.stderr.flush()
    
    # ============ STEP 4a: PROVE ENDPOINT REACHED ============
    from pathlib import Path
    try:
        Path("/tmp/aerivon_ws_reached.log").write_text(
            f"REACHED::{time.time()}::{os.getpid()}::{__file__}\n"
        )
        print("✅ Created /tmp/aerivon_ws_reached.log", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to create ws_reached log: {e}", file=sys.stderr)
    sys.stderr.flush()
    
    await websocket.accept()
    
    print("[AERIVON] 🎯 WebSocket ACCEPTED", file=sys.stderr)
    sys.stderr.flush()
    
    use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not use_vertex or not project:
        await websocket.send_json({"type": "error", "error": "Vertex AI not enabled"})
        await websocket.close(code=1011)
        return

    gen_client = _make_genai_client(prefer_vertex=True, project=project, location=location)
    session_id = int(time.time())
    
    # Session state
    context = {
        "user_id": f"ws_{session_id}",  # Default user_id from session
        "memory_scope": "aerivon_global",
        "cancel_flag": False,
        "state": "IDLE",
        "last_user_text": "",
        "conversation_history": [],
        "current_mode": None,
        "active_task": None,
        "audio_mode": False,  # Track if user is using voice input
        "pw": None,
        "browser": None,
        "browser_context": None,
        "page": None,
    }
    
    # Helper function for narration (shared across story and UI handlers)
    async def narrate_and_send(text: str) -> None:
        """Generate speech audio and send to client."""
        text = (text or "").strip()
        if not text or len(text) < 3:
            print(f"[AERIVON NARRATION] Skipping empty/short text: '{text}'", file=sys.stderr)
            return
        
        print(f"[AERIVON NARRATION] Starting narration for {len(text)} chars", file=sys.stderr)
        try:
            narrate_client = _make_genai_client(prefer_vertex=True, project=project, location=location)
            
            async with narrate_client.aio.live.connect(
                model="gemini-3-flash-preview",
                config=types.LiveConnectConfig(
                    response_modalities=[types.Modality.AUDIO],
                    system_instruction="You are a professional narrator. Read the provided text aloud exactly as written, with expression and emotion.",
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Aoede"
                            )
                        )
                    ),
                ),
            ) as session:
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"Please read this text aloud:\n\n{text}")]
                    ),
                    turn_complete=True
                )
                
                audio_chunks = []
                async for response in session.receive():
                    # Cooperative cancellation check
                    if context["cancel_flag"]:
                        print(f"[AERIVON NARRATION] Cancelled during audio streaming", file=sys.stderr)
                        raise asyncio.CancelledError()
                    
                    if response.data:
                        audio_chunks.append(response.data)
                        print(f"[AERIVON NARRATION] Received audio chunk: {len(response.data)} bytes", file=sys.stderr)
                    if response.server_content and response.server_content.turn_complete:
                        break
                
                if audio_chunks:
                    total_bytes = b"".join(audio_chunks)
                    print(f"[AERIVON NARRATION] ✅ Generated {len(total_bytes)} bytes of audio", file=sys.stderr)
                    
                    # Send audio to client
                    await websocket.send_json({
                        "type": "audio",
                        "data_b64": base64.b64encode(total_bytes).decode("ascii"),
                        "mime_type": "audio/pcm",
                        "sample_rate": 24000
                    })
                else:
                    print(f"[AERIVON NARRATION] ❌ No audio chunks received", file=sys.stderr)
                    
        except asyncio.CancelledError:
            print(f"[AERIVON NARRATION] Task cancelled cleanly", file=sys.stderr)
            raise  # Re-raise so run_with_cancel handler catches it
        except Exception as e:
            print(f"[AERIVON NARRATION] ❌ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Audio buffering for voice input (PCM s16le)
    audio_pcm = bytearray()
    
    await websocket.send_json({
        "type": "status",
        "state": "IDLE",
        "session_id": session_id,
        "message": "Aerivon Command Center ready"
    })
    
    await websocket.send_json({
        "type": "thinking",
        "text": "Aerivon ready. I can help with voice conversations, web navigation, or create illustrated stories."
    })
    
    async def detect_intent(message: dict) -> dict:
        """Detect user intent from message content."""
        msg_type = message.get("type", "")
        text = (message.get("text", "") or "").lower()
        
        # Hard rules for quick detection
        if msg_type in ["audio", "audio_chunk", "audio_end"]:
            return {
                "intent": "live",
                "confidence": 0.95,
                "reason": "Audio input detected",
                "requires_audio": True,
                "requires_browser": False,
                "requires_images": False
            }
        
        # PRIORITY: Check for memory recall questions BEFORE other keyword matching
        # Questions like "do you remember", "what did we", "did I" should use conversational mode to access memory
        memory_recall_patterns = ["remember", "recall", "did we", "did i", "what did", "what was", "earlier"]
        is_memory_question = any(p in text for p in memory_recall_patterns)
        
        if is_memory_question:
            return {
                "intent": "live",
                "confidence": 0.90,
                "reason": "Memory recall question detected",
                "requires_audio": False,
                "requires_browser": False,
                "requires_images": False
            }
        
        # Check for story-related keywords
        story_keywords = ["story", "tale", "narrate", "illustrate", "fantasy", "adventure"]
        # UI keywords: prioritize STRONG signals (actions + explicit URLs)
        # Removed weak keywords like "show me", "what is", "what's" that appear in normal questions
        ui_keywords = [
            "open", "navigate", "click", "go to", "visit",
            "scroll", "type in", "type into", "press", "search on",
            "button", "link", "element"
        ]
        
        # Check for explicit domain/URL patterns (strong UI signal)
        import re
        has_url_pattern = bool(re.search(r'\b\w+\.(com|org|net|io|gov|edu)\b', text))
        has_http_url = bool(re.search(r'https?://', text))
        
        has_story = any(k in text for k in story_keywords)
        has_ui = any(k in text for k in ui_keywords) or has_url_pattern or has_http_url
        
        # Check hybrid FIRST (before story/ui individually)
        if has_story and has_ui:
            return {
                "intent": "hybrid",
                "confidence": 0.90,
                "reason": "Both navigation and story keywords detected",
                "requires_audio": False,
                "requires_browser": True,
                "requires_images": True
            }
        
        if has_story:
            return {
                "intent": "story",
                "confidence": 0.85,
                "reason": "Story-related keywords detected",
                "requires_audio": False,
                "requires_browser": False,
                "requires_images": True
            }
        
        if has_ui:
            # Higher confidence if explicit URL/domain present
            confidence = 0.95 if (has_url_pattern or has_http_url) else 0.80
            return {
                "intent": "ui",
                "confidence": confidence,
                "reason": "Navigation keywords or URL detected",
                "requires_audio": False,
                "requires_browser": True,
                "requires_images": False
            }
        
        # Default to conversational
        return {
            "intent": "live",
            "confidence": 0.70,
            "reason": "Default conversational intent",
            "requires_audio": False,
            "requires_browser": False,
            "requires_images": False
        }
    
    async def run_with_cancel(coro):
        """Run a coroutine with cancellation support."""
        task = asyncio.create_task(coro)
        context["active_task"] = task
        try:
            return await task
        finally:
            if context.get("active_task") is task:
                context["active_task"] = None
    
    async def handle_text_message(text: str):
        """Handle text-based requests with intent routing."""
        # Only disable audio mode if it's not already enabled
        # (preserves audio_mode=True when called from handle_audio_input)
        if not context.get("audio_mode"):
            context["audio_mode"] = False
        
        context["last_user_text"] = text
        context["state"] = "THINKING"
        
        await websocket.send_json({"type": "status", "state": "THINKING"})
        await websocket.send_json({"type": "thinking", "text": "Understanding your request..."})
        
        # Detect intent
        intent_result = await detect_intent({"type": "text", "text": text})
        context["current_mode"] = intent_result["intent"]
        
        # Send intent to frontend
        await websocket.send_json({
            "type": "intent",
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "reason": intent_result["reason"]
        })
        
        # Route based on intent with cancellation support
        try:
            if intent_result["intent"] == "story":
                await run_with_cancel(handle_story_request(text))
            elif intent_result["intent"] == "ui":
                await run_with_cancel(handle_ui_request(text))
            elif intent_result["intent"] == "hybrid":
                await run_with_cancel(handle_hybrid_request(text))
            else:  # live/conversational
                await run_with_cancel(handle_conversational(text))
        except asyncio.CancelledError:
            await websocket.send_json({"type": "text", "text": "⚠️ Task cancelled"})
            context["state"] = "IDLE"
            await websocket.send_json({"type": "status", "state": "IDLE"})
            raise
    
    async def handle_story_request(prompt: str):
        """Handle story generation requests."""
        context["state"] = "GENERATING"
        await websocket.send_json({"type": "status", "state": "GENERATING"})
        await websocket.send_json({"type": "thinking", "text": "Creating an illustrated story..."})
        
        try:
            # Build story prompt with conversation context if available
            story_context = ""
            if context["conversation_history"]:
                # Include recent context for story continuity
                recent_msgs = []
                for ex in context["conversation_history"][-3:]:
                    user_msg = ex.get("user", "")
                    if user_msg:
                        recent_msgs.append(f"User mentioned: {user_msg}")
                if recent_msgs:
                    story_context = "\n\nContext from conversation:\n" + "\n".join(recent_msgs)
            
            story_prompt = f"""Create an illustrated fantasy story based on this prompt:
{prompt}{story_context}

For each scene in the story:
1. Write 2-3 sentences of vivid narration
2. Then request an image that visualizes that scene

Keep the story to exactly 2 scenes. Do not label scenes as "Scene 1" or "Scene 2".
Make it engaging and visual."""
            
            # Use Vertex AI client for story generation
            story_gen_client = _make_genai_client(
                prefer_vertex=True, project=project, location="global"
            )
            
            @retry_with_exponential_backoff
            def _run_story() -> list[dict]:
                """Run generate_content in thread, return list of parts as dicts."""
                resp = story_gen_client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=story_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        temperature=0.9,
                        max_output_tokens=2000,
                    ),
                )
                
                # Extract parts from multimodal response
                parts = []
                if resp.candidates:
                    for candidate in resp.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if part.text:
                                    parts.append({"kind": "text", "text": part.text})
                                elif part.inline_data and part.inline_data.data:
                                    parts.append({
                                        "kind": "image",
                                        "data": base64.b64encode(part.inline_data.data).decode("ascii"),
                                        "mime_type": part.inline_data.mime_type,
                                    })
                return parts
            
            # Run story generation in thread
            parts = await asyncio.to_thread(_run_story)
            
            # Helper to narrate text using Gemini Live API
            async def narrate_text(text: str) -> bytes | None:
                """Generate speech audio for story narration."""
                # Check cancel flag before narration
                if context["cancel_flag"]:
                    print(f"[AERIVON NARRATION] Cancelled before narration", file=sys.stderr)
                    return None
                
                text = (text or "").strip()
                if not text or len(text) < 3:
                    print(f"[AERIVON NARRATION] Skipping empty/short text: '{text}'", file=sys.stderr)
                    return None
                
                print(f"[AERIVON NARRATION] Starting narration for {len(text)} chars", file=sys.stderr)
                try:
                    narrate_client = _make_genai_client(prefer_vertex=True, project=project, location=location)
                    
                    async with narrate_client.aio.live.connect(
                        model="gemini-3-flash-preview",
                        config=types.LiveConnectConfig(
                            response_modalities=[types.Modality.AUDIO],
                            system_instruction="You are a professional narrator. Read the provided text aloud exactly as written, with expression and emotion.",
                            speech_config=types.SpeechConfig(
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name="Aoede"  # Warm, storytelling voice
                                    )
                                )
                            ),
                        ),
                    ) as session:
                        await session.send_client_content(
                            turns=types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=f"Please read this text aloud:\n\n{text}")]
                            ),
                            turn_complete=True
                        )
                        
                        audio_chunks = []
                        async for response in session.receive():
                            # Cooperative cancellation check
                            if context["cancel_flag"]:
                                print(f"[AERIVON NARRATION] Cancelled during audio streaming", file=sys.stderr)
                                raise asyncio.CancelledError()
                            
                            if response.data:
                                audio_chunks.append(response.data)
                                print(f"[AERIVON NARRATION] Received audio chunk: {len(response.data)} bytes", file=sys.stderr)
                            if response.server_content and response.server_content.turn_complete:
                                break
                        
                        if audio_chunks:
                            total_bytes = b"".join(audio_chunks)
                            print(f"[AERIVON NARRATION] ✅ Generated {len(total_bytes)} bytes of audio", file=sys.stderr)
                            return total_bytes
                        else:
                            print(f"[AERIVON NARRATION] ❌ No audio chunks received", file=sys.stderr)
                            return None
                        
                except asyncio.CancelledError:
                    print(f"[AERIVON NARRATION] Task cancelled cleanly", file=sys.stderr)
                    return None  # Return None instead of re-raising, story loop will handle cancel_flag
                except Exception as e:
                    print(f"[AERIVON NARRATION] ❌ Error: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    return None
            
            # Stream parts to client with narration
            pending_text = ""
            for idx, part in enumerate(parts):
                if context["cancel_flag"]:
                    break
                
                if part["kind"] == "text":
                    text = part["text"]
                    pending_text += text
                    await websocket.send_json({
                        "type": "text",
                        "text": text,
                        "index": idx
                    })
                elif part["kind"] == "image":
                    # Narrate accumulated text before showing image
                    if pending_text.strip():
                        print(f"[AERIVON NARRATION] About to narrate text before image: {len(pending_text)} chars", file=sys.stderr)
                        audio_bytes = await narrate_text(pending_text)
                        if audio_bytes:
                            b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                            print(f"[AERIVON NARRATION] Sending {len(audio_bytes)} bytes ({len(b64_audio)} b64 chars) to client", file=sys.stderr)
                            await websocket.send_json({
                                "type": "audio",
                                "data_b64": b64_audio,
                                "mime_type": "audio/pcm;rate=24000",
                                "sample_rate": 24000,
                                "index": idx
                            })
                        else:
                            print(f"[AERIVON NARRATION] No audio generated for text", file=sys.stderr)
                        pending_text = ""
                    
                    await websocket.send_json({
                        "type": "image",
                        "data_b64": part["data"],
                        "mime_type": part["mime_type"],
                        "index": idx
                    })
            
            # Narrate any trailing text
            if pending_text.strip() and not context["cancel_flag"]:
                print(f"[AERIVON NARRATION] About to narrate trailing text: {len(pending_text)} chars", file=sys.stderr)
                audio_bytes = await narrate_text(pending_text)
                if audio_bytes:
                    b64_audio = base64.b64encode(audio_bytes).decode("ascii")
                    print(f"[AERIVON NARRATION] Sending trailing {len(audio_bytes)} bytes ({len(b64_audio)} b64 chars) to client", file=sys.stderr)
                    await websocket.send_json({
                        "type": "audio",
                        "data_b64": b64_audio,
                        "mime_type": "audio/pcm;rate=24000",
                        "sample_rate": 24000,
                        "index": 9999
                    })
            
            # Save story generation to memory
            if context["user_id"] and AERIVON_MEMORY_BUCKET:
                memory_user_id = _memory_user_key(user_id=context["user_id"], scope=context["memory_scope"])
                story_summary = f"Generated illustrated story with {len(parts)} parts (text and images)"
                await _append_exchange_to_memory(
                    user_id=memory_user_id,
                    user_text=prompt,
                    model_text=story_summary
                )
                context["conversation_history"].append({
                    "t": int(time.time()),
                    "user": prompt,
                    "model": story_summary
                })
                print(f"[MEMORY] Saved story generation for user {context['user_id']}", file=sys.stderr)
            
            context["state"] = "DONE"
            await websocket.send_json({"type": "status", "state": "DONE"})
            await websocket.send_json({"type": "done"})
            
        except Exception as e:
            await websocket.send_json({"type": "error", "error": f"Story generation failed: {str(e)}"})
            context["state"] = "IDLE"
            await websocket.send_json({"type": "status", "state": "IDLE"})
    
    async def ensure_browser():
        """Ensure browser is running and ready (reuse across requests)."""
        if context["page"]:
            return
        context["pw"] = await async_playwright().start()
        context["browser"] = await context["pw"].chromium.launch(headless=True)
        context["browser_context"] = await context["browser"].new_context(
            viewport={"width": 1366, "height": 768},
            device_scale_factor=1,
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        context["page"] = await context["browser_context"].new_page()
    
    async def handle_ui_request(goal: str):
        """Handle UI navigation requests with Playwright browser automation."""
        context["state"] = "NAVIGATING"
        await websocket.send_json({"type": "status", "state": "NAVIGATING"})
        await websocket.send_json({"type": "thinking", "text": f"Launching browser for: {goal}"})
        
        try:
            # Ensure browser is ready (reuse if already open)
            await ensure_browser()
            page = context["page"]
            # Infer URL from goal
            goal_lower = goal.lower()
            url = ""
            
            # Check if this is an analysis task without a specific URL
            analysis_keywords = ["find", "analyze", "identify", "read", "get", "extract", "show", "what is", "what's", "hero text", "heading"]
            is_analysis_task = any(kw in goal_lower for kw in analysis_keywords)
            
            # Try to extract or infer URL
            # FIRST: Check for explicit URLs anywhere in the goal (e.g., "Open http://localhost")
            import re
            url_match = re.search(r'https?://[^\s]+', goal_lower)
            if url_match:
                url = url_match.group(0)
            elif "nike" in goal_lower:
                url = "https://www.nike.com"
            elif "amazon" in goal_lower:
                url = "https://www.amazon.com"
            elif "google" in goal_lower:
                url = "https://www.google.com"
            elif "youtube" in goal_lower:
                url = "https://www.youtube.com"
            elif "github" in goal_lower:
                url = "https://www.github.com"
            elif is_analysis_task:
                # Check conversation history for recent website mentions
                recent_url = None
                if context["conversation_history"]:
                    # Look at last 3 exchanges for URLs or website names
                    for ex in reversed(context["conversation_history"][-3:]):
                        user_msg = ex.get("user", "").lower()
                        model_msg = ex.get("model", "").lower()
                        
                        # Check for explicit URLs
                        import re
                        url_pattern = r'https?://[^\s]+'
                        user_urls = re.findall(url_pattern, user_msg)
                        model_urls = re.findall(url_pattern, model_msg)
                        if user_urls:
                            recent_url = user_urls[-1]
                            break
                        elif model_urls:
                            recent_url = model_urls[-1]
                            break
                        
                        # Check for website names
                        if "nike" in user_msg or "nike" in model_msg:
                            recent_url = "https://www.nike.com"
                            break
                        elif "amazon" in user_msg or "amazon" in model_msg:
                            recent_url = "https://www.amazon.com"
                            break
                
                if recent_url:
                    url = recent_url
                    await websocket.send_json({
                        "type": "text",
                        "text": f"📌 Using website from context: {url}"
                    })
                else:
                    # Analysis task without URL and no context - ask for clarification
                    await websocket.send_json({
                        "type": "text",
                        "text": f"I can help you {goal}, but I need to know which website to analyze.\n\nPlease specify a URL or website name. For example:\n• \"Find the hero text on Nike.com\"\n• \"Analyze the heading on https://example.com\"\n• Or first say \"Open Nike.com\" then ask me to analyze it."
                    })
                    context["state"] = "IDLE"
                    await websocket.send_json({"type": "status", "state": "IDLE"})
                    await websocket.send_json({"type": "done"})
                    return
            else:
                # Not an analysis task and no URL - do a Google search
                import urllib.parse
                search_query = urllib.parse.quote_plus(goal)
                url = f"https://www.google.com/search?q={search_query}"
            
            from tools import is_safe_url
            if not is_safe_url(url):
                await websocket.send_json({
                    "type": "error",
                    "error": f"🔒 Blocked unsafe URL: {url}"
                })
                context["state"] = "IDLE"
                await websocket.send_json({"type": "status", "state": "IDLE"})
                await websocket.send_json({"type": "done"})
                return
            
            # Navigate to URL (browser already ensured above)
            await websocket.send_json({
                "type": "text",
                "text": f"🌐 Opening {url}..."
            })
            
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(1000)
            
            page_title = await page.title()
            await websocket.send_json({
                "type": "text",
                "text": f"✅ Loaded: {page_title}"
            })
            
            # Take screenshot
            b64, png = await _ui_screenshot_b64(page)
            await websocket.send_json({
                "type": "screenshot",
                "mime_type": "image/png",
                "data_b64": b64,
                "url": page.url
            })
            
            # Determine if we need action planning or if this is just "open the page"
            analysis_keywords = [
                "find", "what is", "what's", "analyze", "identify", "read", 
                "show me", "get the", "extract", "tell me", "hero text", "heading",
                "largest text", "main text", "title", "click", "search", "type"
            ]
            simple_navigation_only = [
                "open", "go to", "visit", "navigate to", "load", "show me the website"
            ]
            
            # Check if this is an analysis/interaction task (not just opening a page)
            has_analysis_intent = any(kw in goal_lower for kw in analysis_keywords)
            is_simple_navigation = (
                any(nav in goal_lower for nav in simple_navigation_only) and 
                not has_analysis_intent
            )
            
            # Run action planning if there's more to do than just opening the page
            if not is_simple_navigation:
                await websocket.send_json({
                    "type": "text",
                    "text": f"🤖 Planning actions to: {goal}"
                })
                
                memory = [
                    f"User goal: {goal}",
                    f"Current URL: {page.url}",
                    f"Page title: {page_title}"
                ]
                
                # Run planning loop (limited to 3 steps for Command Center)
                for step in range(3):
                    if context["cancel_flag"]:
                        await websocket.send_json({"type": "text", "text": "⚠️ Cancelled by user"})
                        break
                    
                    # Get action plan from Gemini
                    plan = await _ui_plan_actions(
                        client=gen_client,
                        screenshot_png=png,
                        task=goal,
                        memory=memory,
                        page=page
                    )
                    
                    await websocket.send_json({
                        "type": "text",
                        "text": f"📋 Step {step + 1}: {plan.get('note', 'Executing actions...')}"
                    })
                    
                    # Execute actions
                    clickable_elements = plan.get("_clickable_elements") or []
                    actions = plan.get("actions") or []
                    
                    for action in actions:
                        if context["cancel_flag"]:
                            break
                        
                        action_type = str(action.get("type", "")).lower()
                        
                        # Validate goto URLs before navigation (SSRF protection)
                        if action_type == "goto":
                            goto_url = str(action.get("url", ""))
                            from tools import is_safe_url
                            if not is_safe_url(goto_url):
                                await websocket.send_json({
                                    "type": "error",
                                    "error": f"🔒 Blocked unsafe URL: {goto_url}"
                                })
                                continue
                            await page.goto(goto_url, wait_until="domcontentloaded", timeout=45000)
                            await page.wait_for_timeout(1000)
                        # Execute action (simplified version)
                        elif action_type == "click":
                            if "element_index" in action:
                                idx = int(action.get("element_index", 0))
                                if 0 <= idx < len(clickable_elements):
                                    el = clickable_elements[idx]
                                    x = el['x'] + el['w'] // 2
                                    y = el['y'] + el['h'] // 2
                                    await page.mouse.click(x, y)
                                    await page.wait_for_timeout(800)
                        elif action_type == "type":
                            text = str(action.get("text", ""))
                            await page.keyboard.type(text)
                            await page.wait_for_timeout(300)
                        elif action_type == "press":
                            key = str(action.get("key", "Enter"))
                            await page.keyboard.press(key)
                            await page.wait_for_timeout(600)
                        elif action_type == "scroll":
                            dy = int(action.get("delta_y", 0))
                            await page.mouse.wheel(0, dy)
                            await page.wait_for_timeout(300)
                    
                    # Screenshot after actions
                    b64, png = await _ui_screenshot_b64(page)
                    await websocket.send_json({
                        "type": "screenshot",
                        "mime_type": "image/png",
                        "data_b64": b64,
                        "url": page.url
                    })
                    
                    memory.append(f"Step {step + 1} completed. URL: {page.url}")
                    
                    # Check if done
                    if plan.get("done"):
                        completion_note = plan.get('note', 'Task finished!')
                        await websocket.send_json({
                            "type": "text",
                            "text": f"✨ Complete: {completion_note}"
                        })
                        
                        # Narrate the result
                        await narrate_and_send(completion_note)
                        break
            
            # Save UI interaction to memory
            if context["user_id"] and AERIVON_MEMORY_BUCKET:
                memory_user_id = _memory_user_key(user_id=context["user_id"], scope=context["memory_scope"])
                completion_summary = f"Navigated to {page.url}. {completion_note if 'completion_note' in locals() else 'Task completed.'}"
                await _append_exchange_to_memory(
                    user_id=memory_user_id,
                    user_text=goal,
                    model_text=completion_summary
                )
                context["conversation_history"].append({
                    "t": int(time.time()),
                    "user": goal,
                    "model": completion_summary
                })
                print(f"[MEMORY] Saved UI navigation for user {context['user_id']}", file=sys.stderr)
            
            context["state"] = "DONE"
            await websocket.send_json({"type": "status", "state": "DONE"})
            await websocket.send_json({"type": "done"})
            
        except Exception as e:
            import traceback
            error_msg = f"Navigation failed: {str(e)}"
            print(f"[AERIVON UI NAV ERROR] {error_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            
            await websocket.send_json({
                "type": "error",
                "error": error_msg
            })
            context["state"] = "IDLE"
            await websocket.send_json({"type": "status", "state": "IDLE"})
    
    async def handle_hybrid_request(request: str):
        """Handle hybrid requests that combine UI navigation and story generation."""
        await websocket.send_json({"type": "status", "state": "THINKING"})
        await websocket.send_json({"type": "thinking", "text": "Planning multi-step solution..."})
        
        # For now, acknowledge the hybrid intent
        await websocket.send_json({
            "type": "text",
            "text": "Hybrid mode detected! This would:\n1. Navigate the web to gather information\n2. Transform findings into an illustrated story\n\nFull hybrid orchestration coming soon."
        })
        
        context["state"] = "DONE"
        await websocket.send_json({"type": "status", "state": "DONE"})
        await websocket.send_json({"type": "done"})
    
    async def handle_conversational(text: str):
        """Handle general conversational requests."""
        context["state"] = "THINKING"
        await websocket.send_json({"type": "status", "state": "THINKING"})
        
        try:
            # Build conversation context from memory
            system_instruction = "You are Aerivon Live. Be concise and helpful."
            
            # Add conversation history if available
            if context["conversation_history"]:
                history_text = "\n\nPrevious conversation:\n"
                for ex in context["conversation_history"][-6:]:  # Last 6 exchanges
                    user_msg = ex.get("user", "")
                    model_msg = ex.get("model", "")
                    if user_msg:
                        history_text += f"User: {user_msg}\n"
                    if model_msg:
                        history_text += f"Assistant: {model_msg}\n"
                system_instruction += history_text
            
            # Use Gemini 3 Flash for conversational responses with memory
            # Wrap in asyncio.to_thread to avoid blocking event loop
            def _generate():
                return gen_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction + "\nIf the user asks for help debugging/building, be detailed and actionable.",
                        temperature=0.6,
                        max_output_tokens=3000  # Increased from 1800 for complete responses
                    )
                )
            response = await asyncio.to_thread(_generate)
            
            model_response_text = ""
            if response.text:
                model_response_text = response.text
                await websocket.send_json({
                    "type": "text",
                    "text": response.text
                })
                
                # If user spoke to us, speak back
                if context.get("audio_mode"):
                    await narrate_and_send(response.text)
            
            # Save exchange to memory
            if context["user_id"] and AERIVON_MEMORY_BUCKET:
                memory_user_id = _memory_user_key(user_id=context["user_id"], scope=context["memory_scope"])
                await _append_exchange_to_memory(
                    user_id=memory_user_id,
                    user_text=text,
                    model_text=model_response_text
                )
                # Update local context
                context["conversation_history"].append({
                    "t": int(time.time()),
                    "user": text,
                    "model": model_response_text
                })
                print(f"[MEMORY] Saved exchange for user {context['user_id']}", file=sys.stderr)
            
            context["state"] = "IDLE"
            await websocket.send_json({"type": "status", "state": "IDLE"})
            await websocket.send_json({"type": "done"})
            
        except Exception as e:
            await websocket.send_json({"type": "error", "error": f"Conversation failed: {str(e)}"})
            context["state"] = "IDLE"
            await websocket.send_json({"type": "status", "state": "IDLE"})
    
    async def handle_audio_input(wav_bytes: bytes):
        """Handle audio input - transcribe and respond."""
        try:
            # User is using voice input - enable audio mode for response
            context["audio_mode"] = True
            
            context["state"] = "THINKING"
            await websocket.send_json({"type": "status", "state": "THINKING"})
            await websocket.send_json({"type": "thinking", "text": "Listening to your voice..."})
            
            # Use Gemini to transcribe the audio (simple transcription, not verbose response)
            # Wrap in asyncio.to_thread to avoid blocking event loop
            def _transcribe():
                return gen_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[
                        types.Part.from_text(text="Transcribe this audio into text. Return ONLY the exact words spoken, nothing more. Do not add any commentary, explanation, or response. Just the transcription."),
                        types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,  # Lower temperature for accurate transcription
                        max_output_tokens=200  # Reduced from 500 - just transcription
                    )
                )
            response = await asyncio.to_thread(_transcribe)
            
            if response.text:
                # Extract the transcribed text for intent detection
                transcribed_text = response.text.strip()
                
                # Show what was heard
                await websocket.send_json({
                    "type": "text",
                    "text": f"🎤 {transcribed_text}"
                })
                
                # Route based on the transcribed content
                await handle_text_message(transcribed_text)
            else:
                await websocket.send_json({
                    "type": "text",
                    "text": "I couldn't understand the audio. Please try again."
                })
                context["state"] = "IDLE"
                await websocket.send_json({"type": "status", "state": "IDLE"})
                
        except Exception as e:
            await websocket.send_json({"type": "error", "error": f"Audio processing failed: {str(e)}"})
            context["state"] = "IDLE"
            await websocket.send_json({"type": "status", "state": "IDLE"})
    
    # ============ DUAL-TASK ARCHITECTURE ============
    # Fixes interrupt timing issue with concurrent message processing
    
    print("[AERIVON] ⚡️ REACHED DUAL-TASK SETUP! About to open debug log...", file=sys.stderr)
    sys.stderr.flush()
    
    # Debug log file
    debug_log = open("/tmp/aerivon_debug.log", "a")
    def log(msg):
        debug_log.write(f"{msg}\n")
        debug_log.flush()
    
    log("[AERIVON] Debug log file created successfully!")
    print("[AERIVON] ✅ Debug log opened and first message written", file=sys.stderr)
    sys.stderr.flush()
    
    # Message queue for all non-interrupt messages
    message_queue = asyncio.Queue()
    
    async def message_listener():
        """
        Dedicated listener task - always polls for incoming WebSocket messages.
        Processes interrupts IMMEDIATELY without queueing.
        Queues all other messages for the main processor.
        """
        log("[AERIVON] message_listener task started")
        try:
            while True:
                log("[AERIVON LISTENER] Waiting for next message...")
                data = await websocket.receive_json()
                msg_type = data.get("type", "")
                
                log(f"[AERIVON LISTENER] Received: {msg_type}")
                
                if msg_type == "interrupt":
                    # CRITICAL: Process interrupt immediately, do NOT queue it
                    log(f"[INTERRUPT] Received interrupt, active_task: {context.get('active_task')}")
                    
                    # Set cancel flag FIRST
                    context["cancel_flag"] = True
                    
                    # Cancel active task if running
                    if context.get("active_task"):
                        log("[INTERRUPT] Cancelling active_task...")
                        context["active_task"].cancel()
                    
                    # Send acknowledgment IMMEDIATELY
                    log("[INTERRUPT] Sending acknowledgment...")
                    await websocket.send_json({"type": "interrupted", "source": "client"})
                    log("[INTERRUPT] Acknowledgment sent ✅")
                    
                    # Reset state
                    context["state"] = "IDLE"
                    await websocket.send_json({"type": "status", "state": "IDLE"})
                    context["cancel_flag"] = False
                    
                    # Do NOT queue interrupt - it's fully handled here
                    
                else:
                    # All non-interrupt messages go to the queue
                    log(f"[LISTENER] Queueing {msg_type} for processor")
                    await message_queue.put(data)
                    
        except WebSocketDisconnect:
            log("[AERIVON LISTENER] WebSocket disconnected")
        except Exception as e:
            log(f"[AERIVON LISTENER] Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def main_processor():
        """
        Main processor task - consumes messages from the queue.
        Does NOT poll WebSocket directly (that's the listener's job).
        """
        log("[AERIVON] main_processor task started")
        try:
            while True:
                # Get next message from queue
                data = await message_queue.get()
                msg_type = data.get("type", "")
                
                log(f"[AERIVON PROCESSOR] Processing: {msg_type}")
                print(f"[AERIVON PROCESSOR] Processing: {msg_type}", file=sys.stderr)
                
                if msg_type == "text":
                    text = data.get("text", "").strip()
                    if text:
                        await handle_text_message(text)
                
                elif msg_type == "audio":
                    # Buffer PCM audio chunks
                    b64 = str(data.get("data_b64") or "")
                    if b64:
                        try:
                            chunk = base64.b64decode(b64, validate=True)
                            # Cap at ~2MB to prevent memory issues
                            if len(audio_pcm) + len(chunk) <= 2 * 1024 * 1024:
                                audio_pcm.extend(chunk)
                        except Exception:
                            pass
                
                elif msg_type == "audio_end":
                    # Process the buffered audio
                    if audio_pcm:
                        context["state"] = "LISTENING"
                        await websocket.send_json({"type": "status", "state": "LISTENING"})
                        
                        # Convert PCM to WAV
                        wav = _pcm_s16le_to_wav(bytes(audio_pcm), sample_rate=16000, channels=1)
                        audio_pcm.clear()
                        
                        # Use Live API to transcribe and respond to audio
                        await handle_audio_input(wav)
                
                elif msg_type == "start":
                    # Allow override of default user_id
                    if data.get("user_id"):
                        context["user_id"] = data.get("user_id")
                    context["memory_scope"] = data.get("memory_scope", "aerivon_global")
                    
                    # Load user memory from GCS
                    if context["user_id"]:
                        memory_user_id = _memory_user_key(user_id=context["user_id"], scope=context["memory_scope"])
                        user_memory = await _load_user_memory(user_id=memory_user_id)
                        if user_memory:
                            context["conversation_history"] = user_memory.get("exchanges", [])
                            print(f"[MEMORY] Loaded {len(context['conversation_history'])} exchanges for user {context['user_id']}", file=sys.stderr)
                        else:
                            print(f"[MEMORY] No existing memory for user {context['user_id']}", file=sys.stderr)
                    
                    await websocket.send_json({
                        "type": "status",
                        "state": "IDLE",
                        "message": f"Session started for user {context['user_id']} with {len(context['conversation_history'])} previous exchanges"
                    })
                    
        except Exception as e:
            log(f"[AERIVON PROCESSOR] Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Start both tasks concurrently
    log("[AERIVON] Starting dual-task architecture (listener + processor)")
    
    # ============ STEP 4b: PROVE DUAL-TASK STARTED ============
    try:
        Path("/tmp/aerivon_dualtask_started.log").write_text(
            f"DUALTASK_STARTED::{time.time()}::{os.getpid()}::{__file__}\n"
        )
        print("✅ Created /tmp/aerivon_dualtask_started.log", file=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to create dualtask_started log: {e}", file=sys.stderr)
    sys.stderr.flush()
    
    await websocket.send_json({
        "type": "debug",
        "text": "🔧 Dual-task architecture starting..."
    })
    
    try:
        listener_task = asyncio.create_task(message_listener())
        processor_task = asyncio.create_task(main_processor())
        
        log("[AERIVON] Both tasks created, starting gather...")
        
        await websocket.send_json({
            "type": "debug",
            "text": "🔧 Both tasks created, running concurrently..."
        })
        
        # Run both tasks concurrently
        await asyncio.gather(listener_task, processor_task)
    
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        # Clean up browser resources
        try:
            if context.get("page"):
                await context["page"].close()
            if context.get("browser_context"):
                await context["browser_context"].close()
            if context.get("browser"):
                await context["browser"].close()
            if context.get("pw"):
                await context["pw"].stop()
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
            or "gemini-3-flash-preview"
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
