# Aerivon-Live

Secure autonomous Gemini-powered business agent backend (FastAPI) designed for hackathon demos and Cloud Run deployment.

## Architecture Diagram

![Aerivon Live Architecture](docs/images/aerivon-live-architecture.png)

## What this is

Aerivon Live is an autonomous agent runtime that can browse sites (Playwright), scrape leads, and generate outreach messages using Gemini on Vertex AI.

Key features:

- Vertex AI auth via ADC
- Gemini Live availability detection + automatic fallback to standard Gemini Flash when Live isn’t enabled
- Tool calling with allowlist, relevance gating, arg validation, SSRF protection
- Prompt-injection hardening (API + agent + model instruction)
- Cost controls (timeout, tool-call cap) + API limits (rate limit, size caps, bounded session memory)

## Run locally

```bash
cd backend

export GOOGLE_CLOUD_PROJECT="aerivon-live-agent"
export GOOGLE_CLOUD_LOCATION="us-central1"

uvicorn server:app --reload --app-dir . --host 127.0.0.1 --port 8080
```

## Judge / demo endpoints

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
  -d '{"message":"Find 3 dentists in Miami and generate outreach messages"}'
```

## Cloud Run deploy

```bash
gcloud run deploy aerivon-live \
  --source backend \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60 \
  --max-instances 2
```

## Notes

- Live models may not be enabled in all projects/regions; Aerivon Live will automatically fall back.
- Scraping respects robots.txt; a robots block is expected on some sources.
