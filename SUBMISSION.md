# Aerivon Live — Submission Writeup (Live Agents)

## Category

Live Agents 🗣️ (real-time, interruptible audio + multimodal context)

## One-liner

Aerivon Live is a real-time multimodal agent that can **hear**, **see**, and **speak** over an always-on WebSocket session, supporting natural interruption (barge-in) and grounded vision inputs.

## What it does (features)

- **Live voice agent (interruptible)**: speak naturally, barge-in mid-response, and get low-latency audio back over WebSockets (`WS /ws/live`).
- **Multimodal in the same session**: upload an image / send vision frames and get grounded responses (Gemini interprets what it sees).
- **Persistent user memory (optional)**: stores a compact per-user memory file in Google Cloud Storage and injects it into new sessions.
- **Reliability + safety**: bounded payload sizes, clear status/errors, SSRF protections, tool allowlist/validation.

Bonus capability:

- **UI Navigator (visual + actions)**: observes browser screenshots, outputs structured JSON actions, backend executes them with Playwright (`WS /ws/ui`).

## Why it breaks the “text box” paradigm

- The primary experience is **real-time** (audio streaming in and out), not request/response chat.
- Users can **interrupt** the agent naturally (barge-in) and the system handles it gracefully.
- The agent uses **multimodal context** (images/screenshots) to answer and act, not just text.

## Tech stack

### Frontend

- Static HTML/JS browser client
- WebSocket streaming to the backend
- Mic streaming + audio playback
- Image upload preview + automatic compression for WS size limits

### Backend

- FastAPI (Python)
- Google GenAI SDK (`google-genai`) on **Vertex AI**
- Gemini Live API via WebSockets (`client.aio.live.connect`)
- Playwright (headless Chromium) for the UI Navigator execution layer

## Google Cloud services used

- **Vertex AI (Gemini Live + Gemini models)**: real-time multimodal generation and speech.
- **Cloud Storage (GCS)**: optional persistent memory storage (`memory/<user_id>.json`).
- **Cloud Run**: deployment target for the backend (repo includes deployment automation).

## Agent architecture (high-level)

1. Browser client streams mic audio (and optionally images) to `WS /ws/live`.
2. Backend connects to Gemini Live on Vertex and streams model audio/text back to the client.
3. Optional persistent memory:

   - Frontend includes a stable `user_id` on WS URLs.
   - Backend loads and injects `gs://<bucket>/memory/<user_id>.json` on connect.
   - Backend appends short “exchanges” on `turn_complete` and saves back to GCS.

4. (Bonus) UI Navigator:

   - Backend captures screenshot via Playwright.
   - Gemini plans **JSON actions** using the screenshot + task context.
   - Backend executes actions and returns updated screenshots and action results.

## Grounding and hallucination avoidance

- **Vision grounding**: answers are based on uploaded images/screenshots.
- **UI Navigator grounding**: actions are planned against real screenshots and executed in a real browser session.
- **Safety controls**:

  - SSRF blocking for private/localhost/metadata targets
  - Tool allowlist and argument validation
  - Bounded message/result sizes
  - Clear status and error reporting over WS

## Data sources

- Live inputs: user audio + optional uploaded images/screenshots.
- Optional storage: a compact per-user memory JSON in GCS.
- No external private datasets required.

## Findings / learnings

- Real-time agents fail “silently” unless you surface clear status/errors; visible WS status + transcript logging was critical.
- Many phone photos exceed WS payload limits; client-side compression avoids confusing “cut off” responses.
- Barge-in is powerful but easy to mis-trigger due to speaker leakage; conservative thresholds and consecutive-frame gating improve UX.
- A small persistent memory (last few exchanges + summary) is usually enough to make a session feel “context aware” without bloating prompts.

## Repro instructions (judges)

### 1) One command local run

From repo root:

```bash
./aerivon
```

This starts:

- Backend on `http://127.0.0.1:8080`
- Frontend on `http://127.0.0.1:5173` (or next free port)

### 2) Required env/auth

- Uses Vertex/ADC. For local dev:
  - `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON, OR
  - `gcloud auth application-default login`
- Required:
  - `GOOGLE_GENAI_USE_VERTEXAI=True`
  - `GOOGLE_CLOUD_PROJECT=<your-project>`
  - `GOOGLE_CLOUD_LOCATION=us-central1` (recommended)

### 3) Optional: enable persistent memory

Set a bucket and run normally:

```bash
export AERIVON_MEMORY_BUCKET="your-gcs-bucket"
./aerivon
```

### 4) Quick verification endpoints

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/agent/startup-check
```

## Proof of Google Cloud deployment

- Backend deploy target: Cloud Run.
- Repo includes deployment artifacts:
  - `cloudbuild.yaml`
  - `scripts/deploy_cloud_run.sh`

## Demo video checklist (3–4 minutes)

Use the checklist in the next section: [DEMO_CHECKLIST.md](DEMO_CHECKLIST.md)
