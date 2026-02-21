from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from agent import AerivonLiveAgent
from gemini_client import check_live_model_availability


app = FastAPI(title="Aerivon Live Agent API")
agent: AerivonLiveAgent | None = None
SESSION_TOOL_RESULTS: dict[str, list[dict[str, Any]]] = {}
LAST_REQUEST_TIME: dict[str, float] = {}

MAX_MESSAGE_LENGTH = 4000
MAX_SESSION_RESULTS = 100
RATE_LIMIT_SECONDS = 1.0
MAX_RESULT_SIZE = 20000
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


class AgentMessageRequest(BaseModel):
    message: str = Field(min_length=1)


class AgentMessageResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]]


class ToolResultRequest(BaseModel):
    session_id: str
    tool_name: str
    tool_call_id: str | None = None
    result: dict[str, Any]


@app.get("/health")
async def health() -> dict[str, Any]:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    status = check_live_model_availability(project, location)

    return {
        "status": "ok" if status["live_models_available"] else "live_model_unavailable",
        "project": project,
        "location": location,
    }


@app.get("/agent/startup-check")
async def startup_check() -> dict[str, Any]:
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    status = check_live_model_availability(project, location)

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


@app.get("/agent/architecture")
async def architecture() -> dict[str, Any]:
    return {
        "agent": "Aerivon Live",
        "entrypoints": [
            {"method": "POST", "path": "/agent/message"},
            {"method": "POST", "path": "/agent/tool-result"},
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

    session_id = "default"
    SESSION_TOOL_RESULTS.setdefault(session_id, [])

    try:
        turn = await _get_agent().process_message(payload.message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    SESSION_TOOL_RESULTS[session_id].extend(turn.tool_calls)
    SESSION_TOOL_RESULTS[session_id] = SESSION_TOOL_RESULTS[session_id][-MAX_SESSION_RESULTS:]

    return AgentMessageResponse(
        response=turn.response_text,
        tool_calls=turn.tool_calls,
    )


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
