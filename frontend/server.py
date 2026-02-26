from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index.html"

PAGES = {
    "index.html",
    "live_agent.html",
    "translator.html",
    "vision_tutor.html",
    "customer_support.html",
    "ui_navigator.html",
    "workflow_automator.html",
    "visual_qa.html",
    "story.html",
}

BACKEND_BASE = (os.getenv("AERIVON_BACKEND_BASE") or "").strip().rstrip("/")

app = FastAPI(title="Aerivon Live Frontend")


def _render_index() -> str:
    return _render_page("index.html")


def _render_page(name: str) -> str:
    path = APP_DIR / name
    html = path.read_text(encoding="utf-8")
    if not BACKEND_BASE:
        return html

    inject = f"<script>window.AERIVON_BACKEND_BASE={BACKEND_BASE!r};</script>\n"
    if "window.AERIVON_BACKEND_BASE" in html:
        return html

    if "</head>" in html:
        return html.replace("</head>", inject + "</head>", 1)
    return inject + html


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_render_index())


@app.get("/index.html", response_class=HTMLResponse)
def index_html() -> HTMLResponse:
    return HTMLResponse(_render_index())


@app.get("/live_agent.html", response_class=HTMLResponse)
def live_agent_html() -> HTMLResponse:
    return HTMLResponse(_render_page("live_agent.html"))


@app.get("/translator.html", response_class=HTMLResponse)
def translator_html() -> HTMLResponse:
    return HTMLResponse(_render_page("translator.html"))


@app.get("/vision_tutor.html", response_class=HTMLResponse)
def vision_tutor_html() -> HTMLResponse:
    return HTMLResponse(_render_page("vision_tutor.html"))


@app.get("/customer_support.html", response_class=HTMLResponse)
def customer_support_html() -> HTMLResponse:
    return HTMLResponse(_render_page("customer_support.html"))


@app.get("/ui_navigator.html", response_class=HTMLResponse)
def ui_navigator_html() -> HTMLResponse:
    return HTMLResponse(_render_page("ui_navigator.html"))


@app.get("/workflow_automator.html", response_class=HTMLResponse)
def workflow_automator_html() -> HTMLResponse:
    return HTMLResponse(_render_page("workflow_automator.html"))


@app.get("/visual_qa.html", response_class=HTMLResponse)
def visual_qa_html() -> HTMLResponse:
    return HTMLResponse(_render_page("visual_qa.html"))


@app.get("/story.html", response_class=HTMLResponse)
def story_html() -> HTMLResponse:
    return HTMLResponse(_render_page("story.html"))


# Serve static assets (main.js, etc.).
app.mount("/", StaticFiles(directory=str(APP_DIR), html=False), name="static")
