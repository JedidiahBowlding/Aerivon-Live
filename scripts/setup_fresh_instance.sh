#!/usr/bin/env bash
# Aerivon Live - Quick Setup Script for New Cloud Run Instance
# 
# This script sets up a fresh Google Cloud project with all requirements
# and deploys both backend and frontend services.
#
# Usage:
#   1. Edit the PROJECT_ID below
#   2. chmod +x scripts/setup_fresh_instance.sh
#   3. ./scripts/setup_fresh_instance.sh

set -euo pipefail

# =============================================================================
# CONFIGURATION - Edit these values for your project
# =============================================================================

PROJECT_ID="${PROJECT_ID:-gemini-live-488120}"
REGION="${REGION:-us-central1}"
BACKEND_SERVICE_NAME="aerivon-live-agent"
FRONTEND_SERVICE_NAME="aerivon-live-frontend"

# =============================================================================
# Pre-flight checks
# =============================================================================

echo "========================================="
echo "Aerivon Live - Fresh Instance Setup"
echo "========================================="
echo ""

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud CLI not found. Please install it first:"
  echo "  https://cloud.google.com/sdk/docs/install"
  exit 1
fi

if [[ "$PROJECT_ID" == "your-project-id-here" ]]; then
  echo "ERROR: Please edit this script and set your PROJECT_ID"
  exit 1
fi

echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo ""
read -p "Proceed with setup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Setup cancelled."
  exit 0
fi

# =============================================================================
# Step 1: Configure gcloud
# =============================================================================

echo ""
echo "Step 1: Configuring gcloud..."
gcloud config set project "$PROJECT_ID"

# =============================================================================
# Step 2: Enable APIs
# =============================================================================

echo ""
echo "Step 2: Enabling required Google Cloud APIs..."
echo "  (This may take 1-2 minutes)"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  storage-api.googleapis.com \
  firestore.googleapis.com \
  --project="$PROJECT_ID" \
  --quiet

echo "✓ APIs enabled"

# =============================================================================
# Step 3: Create storage bucket
# =============================================================================

echo ""
echo "Step 3: Creating Cloud Storage bucket for memory..."

BUCKET_NAME="$PROJECT_ID-aerivon-memory-$(date +%s)"

if gcloud storage buckets create "gs://$BUCKET_NAME" \
  --project="$PROJECT_ID" \
  --location="$REGION" \
  --uniform-bucket-level-access \
  --quiet 2>/dev/null; then
  echo "✓ Bucket created: $BUCKET_NAME"
else
  echo "⚠ Bucket creation failed or already exists"
  BUCKET_NAME=""
fi

# =============================================================================
# Step 4: Create service account
# =============================================================================

echo ""
echo "Step 4: Creating service account..."

SERVICE_ACCOUNT_NAME="aerivon-live-run"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
  --display-name="Aerivon Live Cloud Run Service Account" \
  --project="$PROJECT_ID" \
  --quiet 2>/dev/null; then
  echo "✓ Service account created: $SERVICE_ACCOUNT_EMAIL"
else
  echo "⚠ Service account already exists: $SERVICE_ACCOUNT_EMAIL"
fi

# Grant AI Platform permissions
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/aiplatform.user" \
  --quiet >/dev/null 2>&1 || true

# Grant storage permissions
if [[ -n "$BUCKET_NAME" ]]; then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.objectAdmin" \
    --quiet >/dev/null 2>&1 || true
fi

echo "✓ Permissions granted"

# =============================================================================
# Step 5: Deploy backend
# =============================================================================

echo ""
echo "Step 5: Deploying backend to Cloud Run..."
echo "  (This will take 3-5 minutes for the first deployment)"

env \
  SERVICE_NAME="$BACKEND_SERVICE_NAME" \
  REGION="$REGION" \
  PROJECT_ID="$PROJECT_ID" \
  SOURCE_DIR="backend" \
  SERVICE_ACCOUNT="$SERVICE_ACCOUNT_EMAIL" \
  GOOGLE_CLOUD_LOCATION="$REGION" \
  AERIVON_MEMORY_BUCKET="$BUCKET_NAME" \
  ./scripts/deploy_cloud_run.sh

BACKEND_URL=$(gcloud run services describe "$BACKEND_SERVICE_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(status.url)')

echo "✓ Backend deployed: $BACKEND_URL"

# =============================================================================
# Step 6: Deploy frontend
# =============================================================================

echo ""
echo "Step 6: Deploying frontend to Cloud Run..."

env \
  SERVICE_NAME="$FRONTEND_SERVICE_NAME" \
  REGION="$REGION" \
  PROJECT_ID="$PROJECT_ID" \
  SOURCE_DIR="frontend" \
  ./scripts/deploy_cloud_run.sh

FRONTEND_URL=$(gcloud run services describe "$FRONTEND_SERVICE_NAME" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(status.url)')

echo "✓ Frontend deployed: $FRONTEND_URL"

# =============================================================================
# Step 7: Verify deployment
# =============================================================================

echo ""
echo "Step 7: Verifying deployment..."

if curl -sf "$BACKEND_URL/health" >/dev/null 2>&1; then
  echo "✓ Backend health check passed"
else
  echo "⚠ Backend health check failed (may take a moment to start)"
fi

# =============================================================================
# Success summary
# =============================================================================

echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Backend URL:  $BACKEND_URL"
echo "Frontend URL: $FRONTEND_URL"
echo ""
echo "Storage Bucket: ${BUCKET_NAME:-none}"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "Next steps:"
echo "  1. Open frontend: $FRONTEND_URL"
echo "  2. Test backend:  curl $BACKEND_URL/health"
echo "  3. View logs:     gcloud run services logs read $BACKEND_SERVICE_NAME --region=$REGION"
echo ""
echo "Documentation: See SETUP.md for details"
echo ""
