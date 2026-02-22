# Aerivon Live — 3–4 Minute Demo Checklist (Live Agents)

## Goal

Demonstrate a **Live**, **interruptible**, **multimodal voice agent** (not turn-based chat) and show clear evidence of Google Cloud + Gemini Live usage.

## Pre-demo setup (30 seconds)

- Run `./aerivon`
- Open the frontend URL printed by the script
- In another terminal, keep these ready:
  - `curl http://127.0.0.1:8080/health`
  - `curl http://127.0.0.1:8080/agent/startup-check`

## Demo script (suggested)

### 1) “Live + interruptible” (60–90 seconds)

1. Click **Start** (mic).
2. Say something that produces a longer response, e.g.
   - “Explain what this project does, then give me three ways to use it.”
3. While the model is speaking, barge in:
   - “Stop. Just give me the one-sentence summary.”

What judges should see:

- State changes (Listening → Speaking → Interrupted)
- Model audio stops promptly and continues with the new instruction

### 2) “See” (vision) (45–60 seconds)

1. Click **Upload Image** and select an image.
2. Ask:
   - “What am I looking at? Give me the key details.”

What judges should see:

- Uploaded image preview appears
- Response references visible details (grounded)

### 3) “Context-aware memory” (30–45 seconds)

1. Say:
   - “Remember that I prefer short bullet answers.”
2. Refresh the page.
3. Ask a new question.

What judges should see:

- The agent follows the preference after reconnect (if GCS memory is enabled)

### 4) (Optional bonus) UI Navigator quick hit (45–60 seconds)

1. In UI Navigator, open a simple site.
2. Ask it to click something obvious and narrate.

What judges should see:

- Screenshot updates
- Structured actions logged

## Cloud proof clip (separate, 20–40 seconds)

Record one of:

- Cloud Run service logs while you hit `/health`
- Cloud Console showing the Cloud Run service + revision, then curl output

## Common failure modes to avoid

- Live unavailable: make sure `/health` returns `status: ok` before recording.
- Audio too short: increase `AERIVON_LIVE_MAX_OUTPUT_TOKENS` if needed.
- Self-interrupt: keep speakers low or use headphones.
