# GCP Deployment Instructions — SE3 Forecast Dashboard

## What you'll have at the end
- A public URL for your Streamlit dashboard
- The API running internally in the same container
- Data syncing hourly via Cloud Scheduler
- Model retraining daily via Cloud Scheduler
- Everything on GCP free tier ($0/month)

---

## Prerequisites (one-time)

### 1. Install Google Cloud CLI
Download from: https://cloud.google.com/sdk/docs/install
Then run:
```bash
gcloud init
gcloud auth login
```

### 2. Create a GCP project
```bash
gcloud projects create se3-forecast-YOUR-INITIALS --name="SE3 Forecast"
gcloud config set project se3-forecast-YOUR-INITIALS
```

### 3. Enable billing
Go to: https://console.cloud.google.com/billing
Link a billing account to your project (free tier — no charges unless you exceed limits).

### 4. Enable required APIs
```bash
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

---

## Step 1 — Create a Cloud Storage bucket

This bucket stores the DuckDB cache and model artifacts.
Both the container and scheduler jobs read/write from here.

```bash
# Replace YOUR-INITIALS with something unique (bucket names are global)
gsutil mb -l europe-north1 gs://se3-forecast-YOUR-INITIALS

# Upload your local data and model
gsutil -m cp -r data/ gs://se3-forecast-YOUR-INITIALS/data/
gsutil -m cp -r model/ gs://se3-forecast-YOUR-INITIALS/model/
```

---

## Step 2 — Create Artifact Registry repository

```bash
gcloud artifacts repositories create se3-forecast \
  --repository-format=docker \
  --location=europe-north1 \
  --description="SE3 forecast app"
```

---

## Step 3 — Build and push the Docker image

Run this from your project root (where Dockerfile lives):

```bash
# Set your project ID
PROJECT_ID=$(gcloud config get-value project)
REGION=europe-north1
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/se3-forecast/app

# Authenticate Docker with GCP
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build the image
docker build -t $IMAGE .

# Push to Artifact Registry
docker push $IMAGE
```

> If you don't have Docker installed locally, use Cloud Build instead:
> ```bash
> gcloud builds submit --tag $IMAGE .
> ```

---

## Step 4 — Deploy to Cloud Run

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=europe-north1
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/se3-forecast/app
BUCKET=gs://se3-forecast-YOUR-INITIALS

gcloud run deploy se3-forecast \
  --image=$IMAGE \
  --region=$REGION \
  --platform=managed \
  --port=8080 \
  --memory=2Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=2 \
  --timeout=300 \
  --set-env-vars="ENTSOE_API_KEY=YOUR_KEY_HERE" \
  --set-env-vars="RETRAIN_SECRET=YOUR_SECRET_HERE" \
  --set-env-vars="API_URL=http://localhost:8000" \
  --set-env-vars="SE3_DB_PATH=/app/data/se3_cache.duckdb" \
  --set-env-vars="MODEL_DIR=/app/model" \
  --allow-unauthenticated
```

> **Replace:**
> - `YOUR_KEY_HERE` with your ENTSO-E API key
> - `YOUR_SECRET_HERE` with a strong random string for the /retrain endpoint
> - `YOUR-INITIALS` in the bucket name

At the end of this command GCP prints a URL like:
```
https://se3-forecast-xxxxxxxx-ew.a.run.app
```
That's your dashboard. Open it in a browser.

---

## Step 5 — Mount Cloud Storage for persistent data

Cloud Run containers are ephemeral — they reset on every restart.
We need to mount the GCS bucket so data and model files persist.

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=europe-north1
BUCKET=se3-forecast-YOUR-INITIALS

# Create a service account for the Cloud Run service
gcloud iam service-accounts create se3-forecast-sa \
  --display-name="SE3 Forecast Service Account"

SA=se3-forecast-sa@$PROJECT_ID.iam.gserviceaccount.com

# Grant access to the bucket
gsutil iam ch serviceAccount:$SA:objectAdmin gs://$BUCKET

# Update the Cloud Run service to use the service account and mount the bucket
gcloud run services update se3-forecast \
  --region=$REGION \
  --service-account=$SA \
  --add-volume=name=data-vol,type=cloud-storage,bucket=$BUCKET \
  --add-volume-mount=volume=data-vol,mount-path=/app/gcs \
  --update-env-vars="SE3_DB_PATH=/app/gcs/data/se3_cache.duckdb" \
  --update-env-vars="MODEL_DIR=/app/gcs/model"
```

---

## Step 6 — Cloud Scheduler jobs

### Hourly data sync
```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=europe-north1
SERVICE_URL=$(gcloud run services describe se3-forecast \
  --region=$REGION --format='value(status.url)')

# Trigger the /retrain endpoint hourly
# (pipeline.py sync is triggered by the API on every /forecast call
#  so a dedicated sync job is optional — add it if you want fresh data
#  even when nobody visits the dashboard)
gcloud scheduler jobs create http se3-hourly-sync \
  --location=$REGION \
  --schedule="0 * * * *" \
  --uri="$SERVICE_URL/health" \
  --http-method=GET \
  --description="Keep Cloud Run warm + trigger data freshness check"
```

### Daily retrain
```bash
RETRAIN_SECRET=YOUR_SECRET_HERE

gcloud scheduler jobs create http se3-daily-retrain \
  --location=$REGION \
  --schedule="0 3 * * *" \
  --uri="$SERVICE_URL/retrain" \
  --http-method=POST \
  --headers="X-Secret=$RETRAIN_SECRET" \
  --description="Daily model retrain at 03:00 Stockholm time" \
  --time-zone="Europe/Stockholm"
```

---

## Step 7 — Verify everything works

```bash
# Get your service URL
SERVICE_URL=$(gcloud run services describe se3-forecast \
  --region=europe-north1 --format='value(status.url)')

# Health check
curl $SERVICE_URL/health

# Forecast
curl $SERVICE_URL/forecast | python -m json.tool

# Dashboard
echo "Open in browser: $SERVICE_URL"
```

---

## Updating the app (after code changes)

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION=europe-north1
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/se3-forecast/app

# Rebuild and push
docker build -t $IMAGE . && docker push $IMAGE

# Deploy new revision (zero downtime)
gcloud run services update se3-forecast \
  --region=$REGION \
  --image=$IMAGE
```

Or with Cloud Build (no local Docker needed):
```bash
gcloud builds submit --tag $IMAGE .
gcloud run services update se3-forecast --region=$REGION --image=$IMAGE
```

---

## Cost estimate

At light usage (a few visits per day, daily retrain):

| Service | Usage | Cost |
|---|---|---|
| Cloud Run | ~100k requests/month | Free (limit: 2M) |
| Cloud Storage | ~500 MB | Free (limit: 5 GB) |
| Cloud Scheduler | 2 jobs | Free (limit: 3) |
| Artifact Registry | ~500 MB image | Free (limit: 0.5 GB) |
| **Total** | | **$0/month** |

---

## Troubleshooting

**Container crashes on startup**
```bash
gcloud logging read "resource.type=cloud_run_revision" --limit=50
```

**Dashboard can't reach API**
The `API_URL=http://localhost:8000` env var tells the dashboard to use the
internal API. If you see connection errors, check supervisord started both
processes — look for both `[api]` and `[dashboard]` in the Cloud Run logs.

**Model not found error on /forecast**
The model artifacts aren't on the GCS mount path. Re-run Step 5 and verify:
```bash
gsutil ls gs://se3-forecast-YOUR-INITIALS/model/
```
Should show `models.pkl`, `linear_baseline.pkl`, `neutralizers.pkl`, `config.json`, `metrics.json`.

**DuckDB file locking error**
DuckDB only allows one writer at a time. If the API and a scheduler job both
try to write simultaneously, one will fail. The pipeline uses `get_conn()` which
opens and closes the connection per operation — this minimises the window but
doesn't eliminate it. For production, consider upgrading to BigQuery which
handles concurrent writes natively.
