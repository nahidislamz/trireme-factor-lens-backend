# Factor Lens - Cloud Run Deployment Guide

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- Docker installed and running
- Access to `trireme-backend` GCP project
- Project files ready in the `Factor_Lens` directory

## Deployment Steps

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
```

This will open a browser window for you to authenticate with your Google account.

### 2. Configure Docker for GCR (Google Container Registry)

```bash
gcloud auth configure-docker
```

### 3. Build the Docker Image

```bash
docker build -t gcr.io/trireme-backend/factor_lens_app:latest .
```

This command:
- Builds a Docker image from your Dockerfile
- Tags it for Google Container Registry
- Uses the `trireme-backend` project

### 4. Push Image to Google Container Registry

```bash
docker push gcr.io/trireme-backend/factor_lens_app:latest
```

This uploads your Docker image to GCP's container registry.

### 5. Deploy to Cloud Run
### Bash
```bash
gcloud run deploy factor-lens \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --timeout=600
```
### Powershell
```powershell
gcloud run deploy factor-lens `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 2 `
  --timeout 600
```
**Configuration Details:**
- `--source .`: Deploy from current directory
- `--region us-central1`: Deploy to US Central region
- `--allow-unauthenticated`: Allow public access (no authentication required)
- `--memory=2Gi`: Allocate 2 GiB of memory (needed for pandas/numpy operations)
- `--cpu=2`: Allocate 2 vCPUs
- `--timeout=600`: 10-minute request timeout



