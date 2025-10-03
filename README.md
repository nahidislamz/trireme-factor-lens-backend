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



# Factor Lens API Documentation

## Base URL

```
Production: https://factor-lens-1018759291561.us-central1.run.app
Interactive Docs: https://factor-lens-1018759291561.us-central1.run.app/docs
```

## Quick Links

- **Interactive API Docs (Swagger UI)**: [/docs](https://factor-lens-1018759291561.us-central1.run.app/docs)
- **Alternative Docs (ReDoc)**: [/redoc](https://factor-lens-1018759291561.us-central1.run.app/redoc)
- **OpenAPI Schema**: [/openapi.json](https://factor-lens-1018759291561.us-central1.run.app/openapi.json)

---

## API Overview

The Factor Lens API provides endpoints for cryptocurrency factor analysis, regime detection using Gaussian Mixture Models (GMM), and correlation analysis. All responses are in JSON format.

---

## Endpoints

### 1. Root / Health Check

Get service status and basic information.

**Endpoint:** `GET /`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/`

---

### 2. Service Status

Get detailed service health status.

**Endpoint:** `GET /api/status`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/api/status`
---

### 3. Get Factor Data

Retrieve computed factor data with BTC price.

**Endpoint:** `GET /api/factors`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/api/factors`

### 4. Get Correlation Matrix

Retrieve correlation matrix between factors.

**Endpoint:** `GET /api/correlation`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/api/correlation`

### 5. Get Regime Data

Get regime classifications using Gaussian Mixture Model (GMM).

**Endpoint:** `GET /api/regimes/{k}`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/api/regimes/{k}`

### 6. Rebuild Regime Data

Force recomputation of regime data for a specific k value.

**Endpoint:** `GET /api/regimes/rebuild/{k}`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/api/regimes/rebuild/{k}`

### 7. Get Available Factors

List all available factors and date range.

**Endpoint:** `GET /api/available-factors`

**URL:** `https://factor-lens-1018759291561.us-central1.run.app/api/available-factors`


## Support

- **Interactive Docs**: [https://factor-lens-1018759291561.us-central1.run.app/docs](https://factor-lens-1018759291561.us-central1.run.app/docs)




