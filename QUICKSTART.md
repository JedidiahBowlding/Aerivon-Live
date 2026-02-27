# Quick Start - Deploy to New Cloud Instance

## Option 1: Automated Setup (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/JedidiahBowlding/Aerivon-Live.git
cd Aerivon-Live

# 2. Edit the setup script with your project ID
nano scripts/setup_fresh_instance.sh
# Change: PROJECT_ID="your-project-id-here"
# To: PROJECT_ID="my-actual-project"

# 3. Run the automated setup
./scripts/setup_fresh_instance.sh
```

**This will automatically:**
- ✓ Enable all required Google Cloud APIs
- ✓ Create a Cloud Storage bucket for memory
- ✓ Create and configure a service account
- ✓ Deploy backend to Cloud Run
- ✓ Deploy frontend to Cloud Run
- ✓ Verify everything is working

**Total time:** 5-7 minutes

---

## Option 2: Manual Setup

### Step-by-step commands:

```bash
# Set your project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  aiplatform.googleapis.com storage-api.googleapis.com \
  --project=$PROJECT_ID

# Create storage bucket
export BUCKET_NAME="$PROJECT_ID-memory-$(date +%s)"
gcloud storage buckets create gs://$BUCKET_NAME \
  --project=$PROJECT_ID --location=us-central1

# Create service account
gcloud iam service-accounts create aerivon-live-run \
  --project=$PROJECT_ID

export SA="aerivon-live-run@$PROJECT_ID.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/storage.objectAdmin"

# Deploy backend
SERVICE_NAME=aerivon-live-agent \
PROJECT_ID=$PROJECT_ID \
SERVICE_ACCOUNT=$SA \
AERIVON_MEMORY_BUCKET=$BUCKET_NAME \
./scripts/deploy_cloud_run.sh

# Deploy frontend
SERVICE_NAME=aerivon-live-frontend \
SOURCE_DIR=frontend \
PROJECT_ID=$PROJECT_ID \
./scripts/deploy_cloud_run.sh

# Get URLs
gcloud run services describe aerivon-live-agent \
  --region=us-central1 --format='value(status.url)'

gcloud run services describe aerivon-live-frontend \
  --region=us-central1 --format='value(status.url)'
```

---

## Option 3: Minimal Setup (No Service Account)

Fastest setup using Cloud Run default service account:

```bash
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  aiplatform.googleapis.com --project=$PROJECT_ID

# Deploy backend (no service account)
PROJECT_ID=$PROJECT_ID ./scripts/deploy_cloud_run.sh

# Deploy frontend
SERVICE_NAME=aerivon-live-frontend \
SOURCE_DIR=frontend \
PROJECT_ID=$PROJECT_ID \
./scripts/deploy_cloud_run.sh
```

**Note:** Default service account has broad permissions. For production, use Option 1 or 2.

---

## What Gets Created

| Resource | Purpose | Cost Estimate |
|----------|---------|---------------|
| Cloud Run (backend) | FastAPI server with Gemini agent | ~$5-15/month |
| Cloud Run (frontend) | Static HTML/JS frontend | ~$3-8/month |
| Cloud Storage bucket | Persistent user memory | ~$0.02/GB/month |
| Service Account | Secure API access | Free |

**Vertex AI usage:** Pay per token (~$0.075-$1.28 per 1M tokens)

---

## Verify Deployment

```bash
# Get URLs
export BACKEND=$(gcloud run services describe aerivon-live-agent \
  --region=us-central1 --format='value(status.url)')
export FRONTEND=$(gcloud run services describe aerivon-live-frontend \
  --region=us-central1 --format='value(status.url)')

# Test backend
curl $BACKEND/health

# Open frontend
open $FRONTEND  # macOS
xdg-open $FRONTEND  # Linux
```

---

## Troubleshooting

### "Permission denied" errors
```bash
# Grant yourself admin access
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="user:your-email@gmail.com" \
  --role="roles/owner"
```

### "API not enabled" errors
```bash
# Re-run API enablement
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  aiplatform.googleapis.com storage-api.googleapis.com firestore.googleapis.com \
  --project=$PROJECT_ID
```

### WebSocket connection fails
```bash
# Disable HTTP/2 for WebSocket compatibility
gcloud run services update aerivon-live-agent \
  --region=us-central1 --no-use-http2
```

### View logs
```bash
# Backend logs
gcloud run services logs read aerivon-live-agent \
  --region=us-central1 --limit=50

# Frontend logs  
gcloud run services logs read aerivon-live-frontend \
  --region=us-central1 --limit=50
```

---

## Update After Changes

```bash
# Redeploy backend
./scripts/deploy_cloud_run.sh

# Redeploy frontend
SERVICE_NAME=aerivon-live-frontend \
SOURCE_DIR=frontend \
./scripts/deploy_cloud_run.sh
```

---

## Clean Up / Delete Everything

```bash
# Delete Cloud Run services
gcloud run services delete aerivon-live-agent --region=us-central1
gcloud run services delete aerivon-live-frontend --region=us-central1

# Delete storage bucket
gcloud storage rm -r gs://your-bucket-name

# Delete service account
gcloud iam service-accounts delete aerivon-live-run@$PROJECT_ID.iam.gserviceaccount.com
```

---

## Full Documentation

See [SETUP.md](SETUP.md) for complete setup guide with all options and explanations.
