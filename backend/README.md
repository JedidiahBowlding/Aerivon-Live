# Aerivon Live Agent (Backend)

FastAPI backend for Aerivon Live Agent using Gemini 2.0 Flash Live API (preview-04-09) via Vertex AI. Features real-time duplex audio streaming, interrupt handling, persistent memory via GCS, and automatic upstream disconnect recovery.

## Why Hackathon-Ready

- **Gemini 2.0 Flash Live** multimodal API with streaming audio/text
- **Handles interruptions gracefully** - key requirement for "Live Agents 🗣️" category
- **Persistent memory** - conversation context preserved across network disconnects
- **WebSocket duplex audio** - real-time bidirectional streaming
- **Production deployed** on Google Cloud Run
- **Auto-reconnect logic** - handles frequent upstream_disconnected events (5-15s intervals)

## Quick Start (Local)

From the repository root:

```bash
./aerivon
```

Or manually from this folder:

```bash
export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="us-central1"
export AERIVON_MEMORY_BUCKET="aerivon-live-agent-memory-1771792693"

uvicorn server:app --host 127.0.0.1 --port 8081 --app-dir .
```

Test the health endpoint:

```bash
curl http://localhost:8081/health
```

## Key Endpoints

### `GET /health`
Returns `ok` if Gemini Live connection is active, otherwise `live_model_unavailable` (normal when no client connected).

### `WS /ws/live`
WebSocket endpoint for real-time duplex audio streaming.

**Query Parameters:**
- `memory_scope` - Session identifier for persistent memory (e.g., `live_agent_{UUID}`)

**Client → Server Messages:**
```json
{"type": "audio", "data": "<base64 PCM>"}
{"type": "interrupt"}
{"type": "text", "text": "message"}
```

**Server → Client Messages:**
```json
{"type": "audio", "data": "<base64 PCM>"}
{"type": "transcript", "text": "..."}
{"type": "status", "status": "..."}
{"type": "turn_complete"}
```

## Memory Architecture

- **Storage**: Google Cloud Storage bucket (configurable via `AERIVON_MEMORY_BUCKET`)
- **Scope**: Per-user sessions identified by `memory_scope` query parameter
- **Format**: JSON array of `{user: "...", agent: "...", timestamp: "..."}` exchanges
- **Lifecycle**: 
  - Loaded at start of each WebSocket session
  - Saved after each exchange (including partial exchanges before upstream disconnects)
  - Reloaded automatically after upstream reconnects

## Upstream Disconnect Handling

The Gemini Live API frequently disconnects (every 5-15 seconds). The backend handles this gracefully:

1. Saves partial exchanges before restart
2. Reloads memory from GCS at start of new session
3. Rebuilds system instruction with updated conversation history
4. Client auto-reconnects with exponential backoff (no user interruption)

## Production Deployment (Cloud Run)

**Backend**: <https://aerivon-live-agent-621908229087.us-central1.run.app>  
**Frontend**: <https://aerivon-live-frontend-621908229087.us-central1.run.app>

Deploy updates:

```bash
./scripts/deploy_cloud_run.sh
```

Or manually:

```bash
gcloud run deploy aerivon-live-agent \
  --source ./backend \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars AERIVON_MEMORY_BUCKET=aerivon-live-agent-memory-1771792693
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture with Mermaid diagrams.

**Key Components:**
- FastAPI WebSocket server
- Gemini 2.0 Flash Live client (Vertex AI SDK)
- GCS-backed memory persistence  
- Audio processing: PCM16 at 16kHz (input) and 24kHz (output)
- Client-side VAD with server-side barge-in coordination

## Development

**Launch script**: `./aerivon` from repo root (starts both backend and frontend)

**Environment Variables:**
- `GOOGLE_CLOUD_PROJECT` - GCP project ID (required)
- `GOOGLE_CLOUD_LOCATION` - Region for Vertex AI (default: `us-central1`)
- `AERIVON_MEMORY_BUCKET` - GCS bucket for memory storage
- `AERIVON_BACKEND_PORT` - Backend port (default: 8081)
- `AERIVON_FRONTEND_PORT` - Frontend port (default: 5174)

**No reload mode** (for debugging):
```bash
export AERIVON_RELOAD=0
./aerivon
```
