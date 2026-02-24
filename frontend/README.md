# Aerivon Live – Demo Frontend (Multi-page)

This frontend serves separate pages for each demo feature (Live Agent, Translator, Vision Tutor, Customer Support, UI Navigator variants).

## Run

1. Start the backend:

```bash
cd backend
export GOOGLE_GENAI_USE_VERTEXAI=True
export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="global"
uvicorn server:app --host 127.0.0.1 --port 8080 --app-dir .
```

1. Serve this folder:

```bash
cd frontend
python3 -m http.server 5173
```

1. Open:

- <http://localhost:5173>

Pages:

- `/live_agent.html`
- `/translator.html`
- `/vision_tutor.html`
- `/customer_support.html`
- `/ui_navigator.html`
- `/workflow_automator.html`
- `/visual_qa.html`

### Connect to Cloud Run backend

If your backend is deployed to Cloud Run, serve the frontend locally and pass the backend base URL:

Example:

`http://localhost:5173/?backend=https://aerivon-live-agent-yt33hll5ka-uc.a.run.app`

Click **Start** (where available) to stream microphone audio to Live over WebSocket and play back the model audio output.

## Demo scenario (3 minutes)

Pick one story and run it cleanly end-to-end. This repo is currently best suited for **Voice Web Research** (no extra UI needed).

1. Click **Start**.
2. Say:

    “Open example.com, tell me what the page is about in one sentence, then read the title out loud.”

3. While the agent is speaking, barge in (talk over it) and say:

    “Stop — just tell me the title only.”

Watch for the UI state badge transitions and the 🔴 **Interrupted** cue.
