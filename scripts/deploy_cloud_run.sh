#!/usr/bin/env bash
set -euo pipefail

# Automated Cloud Run deployment for Aerivon Live.
#
# Prereqs:
# - gcloud installed and authenticated
# - Cloud Run + Cloud Build APIs enabled
#
# Usage:
#   scripts/deploy_cloud_run.sh
#
# Optional env vars:
#   SERVICE_NAME            (default: aerivon-live)
#   REGION                  (default: us-central1)
#   PROJECT_ID              (default: from gcloud config)
#   SOURCE_DIR              (default: backend)
#   SERVICE_ACCOUNT         (default: none; uses Cloud Run default)
#   ALLOW_UNAUTHENTICATED   (default: true)
#   MEMORY                  (default: 512Mi)
#   CPU                     (default: 1)
#   TIMEOUT                 (default: 600)
#   MAX_INSTANCES           (default: 2)
#   MIN_INSTANCES           (default: 0)
#   GOOGLE_CLOUD_LOCATION    (default: us-central1)
#   AERIVON_MEMORY_BUCKET    (default: empty)
#   AERIVON_LIVE_MAX_OUTPUT_TOKENS (default: 2500)
#

SERVICE_NAME="${SERVICE_NAME:-aerivon-live-agent}"
REGION="${REGION:-us-central1}"
SOURCE_DIR="${SOURCE_DIR:-backend}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"
MEMORY="${MEMORY:-512Mi}"
CPU="${CPU:-1}"
TIMEOUT="${TIMEOUT:-600}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"

GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
AERIVON_MEMORY_BUCKET="${AERIVON_MEMORY_BUCKET:-}"
AERIVON_LIVE_MAX_OUTPUT_TOKENS="${AERIVON_LIVE_MAX_OUTPUT_TOKENS:-2500}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found. Install Google Cloud SDK first." >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "ERROR: SOURCE_DIR '$SOURCE_DIR' not found (cwd: $(pwd))." >&2
  exit 1
fi

PROJECT_ID="${PROJECT_ID:-gemini-live-488120}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: PROJECT_ID not set." >&2
  exit 1
fi

AUTH_FLAG=""
if [[ "$ALLOW_UNAUTHENTICATED" == "true" ]]; then
  AUTH_FLAG="--allow-unauthenticated"
else
  AUTH_FLAG="--no-allow-unauthenticated"
fi

echo "Deploying Cloud Run service '$SERVICE_NAME' to project '$PROJECT_ID' region '$REGION' from source '$SOURCE_DIR'..."

SA_FLAG=()
if [[ -n "$SERVICE_ACCOUNT" ]]; then
  SA_FLAG=(--service-account "$SERVICE_ACCOUNT")
fi

ENV_VARS=("GOOGLE_GENAI_USE_VERTEXAI=True" "GOOGLE_CLOUD_PROJECT=$PROJECT_ID" "GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION" "AERIVON_LIVE_MAX_OUTPUT_TOKENS=$AERIVON_LIVE_MAX_OUTPUT_TOKENS")
if [[ -n "$AERIVON_MEMORY_BUCKET" ]]; then
  ENV_VARS+=("AERIVON_MEMORY_BUCKET=$AERIVON_MEMORY_BUCKET")
fi

# Add API key if available (for Live API narration which doesn't work with Vertex AI)
if [[ -n "$GEMINI_API_KEY" ]]; then
  ENV_VARS+=("GEMINI_API_KEY=$GEMINI_API_KEY")
  echo "✓ GEMINI_API_KEY will be set in Cloud Run"
elif [[ -n "$GOOGLE_API_KEY" ]]; then
  ENV_VARS+=("GOOGLE_API_KEY=$GOOGLE_API_KEY")
  echo "✓ GOOGLE_API_KEY will be set in Cloud Run"
elif [[ -n "$GOOGLE_CLOUD_API_KEY" ]]; then
  ENV_VARS+=("GOOGLE_CLOUD_API_KEY=$GOOGLE_CLOUD_API_KEY")
  echo "✓ GOOGLE_CLOUD_API_KEY will be set in Cloud Run"
else
  echo "⚠ No API key found - narration features may not work. Set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_CLOUD_API_KEY"
fi

ENV_JOINED=$(IFS=, ; echo "${ENV_VARS[*]}")

gcloud run deploy "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source "$SOURCE_DIR" \
  $AUTH_FLAG \
  ${SA_FLAG[@]+"${SA_FLAG[@]}"} \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --timeout "$TIMEOUT" \
  --max-instances "$MAX_INSTANCES" \
  --min-instances "$MIN_INSTANCES" \
  --set-env-vars "$ENV_JOINED" \
  --session-affinity \
  --execution-environment=gen2 \
  --quiet

echo "Done."
