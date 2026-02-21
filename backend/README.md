# Aerivon Live (Hackathon Backend)

Aerivon Live is a secure autonomous business agent backend built on FastAPI and Gemini (Vertex AI). It supports tool calling, browsing with Playwright, lead scraping, outreach message generation, and layered defenses against prompt injection, SSRF, and runaway costs.

## Why this is hackathon-ready

- Gemini on Vertex AI (ADC auth) with automatic **Live → standard Gemini fallback**
- Tool calling with allowlist + argument validation + relevance gating
- SSRF protections (blocks localhost, metadata endpoints, private IPs)
- Prompt injection protection at API + agent + model-instruction layers
- Cost controls: per-turn timeout, per-turn tool-call cap, rate limiting
- Verification endpoints for judges: `/health`, `/agent/startup-check`, `/agent/security-check`, `/agent/self-test`, `/agent/architecture`

## Quickstart (local)

From this folder:

```bash
# Recommended env vars
export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="us-central1"

# If running locally with a key file (DO NOT COMMIT):
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

uvicorn server:app --reload --app-dir . --host 127.0.0.1 --port 8080
```

Test endpoints:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/agent/startup-check
curl http://localhost:8080/agent/security-check
curl http://localhost:8080/agent/self-test
curl http://localhost:8080/agent/architecture
```

## Demo request

```bash
curl -X POST http://localhost:8080/agent/message \
  -H "Content-Type: application/json" \
  -d '{"message":"Find dentists in Miami and generate outreach message"}'
```

Note: `scrape_leads` may return a `robots.txt` block depending on the target site. This is intentional and demonstrates compliant scraping behavior.

## Endpoints

- `GET /health`
  - Returns `ok` if Gemini Live models are available, otherwise `live_model_unavailable`.
- `GET /agent/startup-check`
  - Shows project/region and whether Live models are available.
- `GET /agent/security-check`
  - Shows active limits and security policy settings.
- `GET /agent/self-test`
  - Runs a lightweight self-test suite (no destructive actions).
- `GET /agent/architecture`
  - Returns a JSON description of the system architecture.
- `POST /agent/message`
  - Input: `{ "message": "..." }`
  - Output: `{ "response": "...", "tool_calls": [...] }`
- `POST /agent/tool-result`
  - Allows externally provided tool results to be stored (bounded by size and session caps).

## Security model (summary)

### API layer
- Rate limiting per client IP (`RATE_LIMIT_SECONDS`)
- Request size limits (`MAX_MESSAGE_LENGTH`, `MAX_RESULT_SIZE`)
- Prompt-injection phrase filtering
- SSRF pre-check to block unsafe targets before model execution

### Agent layer
- Tool allowlist
- Tool relevance gating vs explicit user intent
- URL/tool argument validation
- Tool output is wrapped as `untrusted_data` with a security note
- Tool call limit per turn (max 6)
- Turn timeout (30s)

### Tool layer
- `is_safe_url()` blocks localhost, metadata endpoints, private ranges
- `browse_url()` uses Playwright headless Chromium
- `content_preview` capped to 1200 chars
- `scrape_leads()` returns 5–10 leads (bounded)

## Deployment (Cloud Run)

Build and deploy from this folder:

```bash
gcloud run deploy aerivon-live \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60 \
  --max-instances 2
```

Cloud Run should use the attached service account automatically. Do not ship credential JSON files.

## Secrets

- Service account JSON files must never be committed.
- Root repo `.gitignore` includes `*.json` and `snark-shot-5f3a4ada1b7c.json`.

## Architecture diagram

See ARCHITECTURE.md.
