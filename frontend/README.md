# Aerivon Live – Frontend

Web interface for Aerivon Live Agent powered by Gemini 2.0 Flash Live API with real-time voice interaction, barge-in detection, and persistent memory across sessions.

## Quick Start

From the repository root:

```bash
./aerivon
```

This launches both backend (port 8081) and frontend (port 5174).

Open: <http://localhost:5174/live_agent.html>

## Manual Setup

1. Start the backend:

```bash
cd backend
export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="us-central1"
uvicorn server:app --host 127.0.0.1 --port 8081 --app-dir .
```

1. Serve this folder:

```bash
cd frontend
python server.py
```

1. Open <http://localhost:5174>

## Cloud Run Deployment

Production instance: <https://aerivon-live-frontend-621908229087.us-central1.run.app>

The frontend automatically connects to the deployed backend at:
<https://aerivon-live-agent-621908229087.us-central1.run.app>

## Features

- **Real-time duplex audio**: WebSocket streaming with PCM audio (16kHz)
- **Voice Activity Detection**: Client-side VAD with dynamic noise floor learning
- **Barge-in interrupts**: Speak over the agent to interrupt mid-response
- **Persistent memory**: Conversation history saved to GCS, reloaded on reconnect
- **Auto-reconnect**: Handles upstream disconnects gracefully with exponential backoff

## Demo (3-minute hackathon test)

1. Click **Start Mic**
2. Say: "What's the weather like in San Francisco?"
3. While the agent responds, interrupt by saying: "Actually, tell me about Tokyo instead"
4. Watch for the 🔴 **Interrupted** indicator
5. Refresh the page and click **Start Mic** again - the agent remembers your conversation

## Debug Mode

Add `?debug=1` to see detailed logs in the browser console:

<http://localhost:5174/live_agent.html?debug=1>

## Architecture

- **Audio Input**: ScriptProcessor (4096 samples) → downsample 48kHz→16kHz → RMS VAD → WebSocket
- **Audio Output**: Base64 PCM → decode → AudioBuffer → scheduled playback via AudioContext
- **VAD Thresholds**: Speech start (0.0025 RMS), Barge-in (0.045 RMS)
- **Memory Scope**: UUID-based session identifier sent with each connection
