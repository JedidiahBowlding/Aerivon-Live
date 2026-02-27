# Aerivon Live - Fresh Cloud Run Setup Guide

Complete setup checklist for deploying Aerivon Live to a new Google Cloud project from scratch.

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed ([install guide](https://cloud.google.com/sdk/docs/install))
- Git installed

## 1. Initial Google Cloud Setup

### Set your project ID
```bash
# Replace with your actual project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID
```

### Authenticate with Google Cloud
```bash
# Login to your Google account
gcloud auth login

# Set up Application Default Credentials (for local dev)
gcloud auth application-default login
```

## 2. Enable Required Google Cloud APIs

```bash
# Enable all required APIs at once
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  storage-api.googleapis.com \
  firestore.googleapis.com \
  --project=$PROJECT_ID
```

**APIs enabled:**
- Cloud Run - serverless container hosting
- Cloud Build - builds Docker containers from source
- Vertex AI - Gemini models
- Cloud Storage - persistent memory storage (optional)
- Firestore - alternative memory storage (optional)

## 3. Create Storage Bucket (Optional but Recommended)

For persistent user memory across sessions:

```bash
# Create a unique bucket name
export BUCKET_NAME="$PROJECT_ID-aerivon-memory-$(date +%s)"

# Create the bucket
gcloud storage buckets create gs://$BUCKET_NAME \
  --project=$PROJECT_ID \
  --location=us-central1 \
  --uniform-bucket-level-access

# Verify bucket exists
gcloud storage buckets describe gs://$BUCKET_NAME
```

## 4. Create Service Account (Recommended for Production)

```bash
# Create service account for Cloud Run
gcloud iam service-accounts create aerivon-live-run \
  --display-name="Aerivon Live Cloud Run Service Account" \
  --project=$PROJECT_ID

# Get the service account email
export SERVICE_ACCOUNT="aerivon-live-run@$PROJECT_ID.iam.gserviceaccount.com"

# Grant required permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/aiplatform.user"

# Grant storage permissions (if using GCS bucket)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/storage.objectAdmin"
```

## 5. Deploy Backend to Cloud Run

```bash
# Navigate to your project directory
cd /path/to/aerivon-live

# Deploy using the deployment script
SERVICE_NAME=aerivon-live-agent \
REGION=us-central1 \
PROJECT_ID=$PROJECT_ID \
SERVICE_ACCOUNT=$SERVICE_ACCOUNT \
GOOGLE_CLOUD_LOCATION=us-central1 \
AERIVON_MEMORY_BUCKET=$BUCKET_NAME \
./scripts/deploy_cloud_run.sh
```

**Wait 2-5 minutes for build and deployment to complete.**

### Get Backend URL
```bash
export BACKEND_URL=$(gcloud run services describe aerivon-live-agent \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --format='value(status.url)')

echo "Backend URL: $BACKEND_URL"
```

## 6. Deploy Frontend to Cloud Run

```bash
# Deploy frontend pointing to backend
SERVICE_NAME=aerivon-live-frontend \
SOURCE_DIR=frontend \
PROJECT_ID=$PROJECT_ID \
REGION=us-central1 \
./scripts/deploy_cloud_run.sh
```

### Get Frontend URL
```bash
export FRONTEND_URL=$(gcloud run services describe aerivon-live-frontend \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --format='value(status.url)')

echo "Frontend URL: $FRONTEND_URL"
```

## 7. Verify Deployment

### Test backend health
```bash
curl $BACKEND_URL/health
```

**Expected:** `{"status":"ok","project":"your-project-id","location":"us-central1"}`

### Test agent startup
```bash
curl $BACKEND_URL/agent/startup-check
```

### Open frontend in browser
```bash
# macOS
open $FRONTEND_URL

# Linux
xdg-open $FRONTEND_URL

# Or just print and copy to browser
echo $FRONTEND_URL
```

## 8. Troubleshooting WebSocket Issues

If WebSocket endpoints fail with 404 or connection errors:

```bash
# Verify HTTP/2 is disabled for WebSocket compatibility
gcloud run services update aerivon-live-agent \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --no-use-http2

# Verify session affinity is enabled
gcloud run services describe aerivon-live-agent \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --format='value(spec.template.metadata.annotations)'
```

Should include: `run.googleapis.com/sessionAffinity: 'true'`

## 9. Environment Variables Reference

### Required (automatically set by deployment script)
- `GOOGLE_GENAI_USE_VERTEXAI=True` - Use Vertex AI instead of API key
- `GOOGLE_CLOUD_PROJECT` - Your GCP project ID
- `GOOGLE_CLOUD_LOCATION` - Region for Vertex AI (us-central1)

### Optional
- `AERIVON_MEMORY_BUCKET` - GCS bucket for persistent memory
- `AERIVON_FIRESTORE_COLLECTION` - Firestore collection (alternative to GCS)
- `AERIVON_LIVE_MAX_OUTPUT_TOKENS` - Max tokens per response (default: 2500)
- `GEMINI_LIVE_VISION_MODEL` - Specific vision model to use

## 10. Cost Estimates

**Cloud Run:**
- Backend: ~$5-15/month (512Mi RAM, 1 CPU, light usage)
- Frontend: ~$3-8/month (minimal resources)

**Vertex AI (Gemini):**
- Gemini Flash: ~$0.075 per 1M input tokens, $0.30 per 1M output tokens
- Gemini Live: ~$0.32 per 1M input tokens, $1.28 per 1M output tokens

**Cloud Storage:**
- Standard storage: ~$0.02 per GB/month
- Operations: minimal cost for memory reads/writes

## 11. Quick Reference

### View logs
```bash
# Backend logs
gcloud run services logs read aerivon-live-agent \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --limit=50

# Frontend logs
gcloud run services logs read aerivon-live-frontend \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --limit=50
```

### Update deployment
```bash
# Redeploy backend after code changes
cd /path/to/aerivon-live
git pull  # if using git
./scripts/deploy_cloud_run.sh

# Redeploy frontend
SERVICE_NAME=aerivon-live-frontend \
SOURCE_DIR=frontend \
./scripts/deploy_cloud_run.sh
```

### Delete services
```bash
# Delete backend
gcloud run services delete aerivon-live-agent \
  --region=us-central1 \
  --project=$PROJECT_ID

# Delete frontend
gcloud run services delete aerivon-live-frontend \
  --region=us-central1 \
  --project=$PROJECT_ID
```

## 12. Security Checklist

- [ ] Service account has minimum required permissions
- [ ] Cloud Run services use `--no-allow-unauthenticated` for production
- [ ] CORS origins configured in `backend/server.py` (if needed)
- [ ] Secrets stored in Secret Manager (not environment variables)
- [ ] Rate limiting enabled for production workloads
- [ ] VPC connector configured for private GCS/Firestore access (optional)

## 13. Production Recommendations

### Scale settings
```bash
# Update for production load
gcloud run services update aerivon-live-agent \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --memory=1Gi \
  --cpu=2 \
  --max-instances=10 \
  --min-instances=1 \
  --timeout=300
```

### Custom domain
```bash
# Map custom domain
gcloud run services add-iam-policy-binding aerivon-live-frontend \
  --region=us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"

gcloud beta run domain-mappings create \
  --service=aerivon-live-frontend \
  --domain=your-domain.com \
  --region=us-central1
```

## Support

- Issues: [GitHub Issues](https://github.com/JedidiahBowlding/Aerivon-Live/issues)
- Documentation: See [README.md](README.md)
- Architecture: See [ARCHITECTURE.md](ARCHITECTURE.md)
