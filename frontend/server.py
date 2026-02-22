from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index.html"

BACKEND_BASE = (os.getenv("AERIVON_BACKEND_BASE") or "").strip().rstrip("/")

app = FastAPI(title="Aerivon Live Frontend")


def _render_index() -> str:
    html = INDEX_PATH.read_text(encoding="utf-8")
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


# Serve static assets (main.js, etc.).
app.mount("/", StaticFiles(directory=str(APP_DIR), html=False), name="static")
