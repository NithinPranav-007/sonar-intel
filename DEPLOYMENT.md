# SONAR-INTEL Deployment Guide: Vercel (Frontend) & Render (Backend)

This document provides a step-by-step guide to deploying the **SONAR-INTEL** platform with the **Frontend on Vercel** and the **FastAPI + ML Backend on Render**.

---

## 1. System Architecture Overview

```
 ┌───────────────────────────────────────┐
 │        Vercel (Frontend SPA)          │
 │   - React 18 + Vite + Tailwind CSS    │
 │   - URL: https://sonar-intel.vercel.app│
 └──────────────────┬────────────────────┘
                    │
                    │ REST API & Image Streaming (HTTPS / CORS enabled)
                    │ VITE_API_URL -> Render Backend
                    ▼
 ┌───────────────────────────────────────┐
 │          Render (Web Service)         │
 │   - FastAPI + Uvicorn Python 3.11/3.13│
 │   - Ultralytics YOLOv11/v8 + PyTorch  │
 │   - URL: https://<your-app>.onrender.com
 └──────────────────┬────────────────────┘
                    │
                    │ Database Connection (psycopg / SQLAlchemy)
                    ▼
 ┌───────────────────────────────────────┐
 │   PostgreSQL + PostGIS or Fallback    │
 │   - Render Managed PostgreSQL         │
 │   - (Auto-falls back to SQLite if none)│
 └───────────────────────────────────────┘
```

---

## 2. Deploying Backend on Render

You can deploy the backend to Render either using the automated **Blueprint (`render.yaml`)** or as a **Manual Web Service**.

### Option A: 1-Click Blueprint (Recommended)
1. Push your project repository to GitHub or GitLab.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Blueprint**.
3. Connect your repository.
4. Render will automatically detect `render.yaml` and configure the Web Service with all build and start commands!
5. Click **Apply**.

---

### Option B: Manual Web Service Setup
1. Log in to [Render](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect your Git repository.
3. Configure the service settings:
   - **Name**: `sonar-intel-backend`
   - **Environment**: `Python 3`
   - **Region**: Any (e.g., `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch**: `main` (or your active branch)
   - **Root Directory**: Leave blank (root `.`)
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```
     *(If you set Root Directory to `backend`, use `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`)*
   - **Plan**: `Free` (or higher)

4. Under **Advanced** -> **Health Check Path**, enter:
   ```
   /api/health
   ```

5. Under **Environment Variables**, configure the following:

| Variable | Recommended Value | Description |
|---|---|---|
| `PYTHON_VERSION` | `3.11.9` | Stable Python runtime for PyTorch & OpenCV |
| `OMP_NUM_THREADS` | `1` | Strictly caps OpenMP thread memory pool (<512MB RAM) |
| `MKL_NUM_THREADS` | `1` | Strictly caps Intel MKL thread memory pool |
| `OPENBLAS_NUM_THREADS` | `1` | Prevents OpenBLAS thread allocation bloat |
| `MALLOC_ARENA_MAX` | `2` | Prevents Linux glibc memory arena bloat on Render |
| `MODEL_PATH` | `ml/models/best_distilled_yolo11n.pt` | Ultralightweight distilled YOLO11n checkpoint (~5MB) |
| `ALLOWED_ORIGINS` | `https://<your-vercel-app>.vercel.app,http://localhost:5173` | Comma-separated list of allowed frontend origins |
| `FRONTEND_URL` | `https://<your-vercel-app>.vercel.app` | Primary frontend Vercel URL |
| `CORS_ORIGIN_REGEX` | `^https:\/\/.*\.vercel\.app$` | Automatically permits all Vercel production & preview URLs |
| `DATABASE_URL` | *(Optional)* | Render PostgreSQL connection string (`postgres://...` is automatically normalized) |
| `HF_MODEL_ID` | `Samyukta31/sonar_yolo` | Hugging Face model repository |
| `HF_MODEL_FILE` | `best_distilled_yolo11n.pt` | Pretrained YOLO weights file on Hugging Face |
| `HF_TOKEN` | *(Optional)* | Your Hugging Face read token (avoids rate limits) |
| `CONFIDENCE_THRESHOLD` | `0.25` | Detector inference confidence cut-off |
| `DEVICE` | `cpu` | Inference device (`cpu` on free tier, `cuda:0` if GPU enabled) |

6. Click **Create Web Service**. Once deployed, Render will display your public URL:
   `https://<your-service-name>.onrender.com`

---

## 3. Deploying Frontend on Vercel

1. Log in to [Vercel](https://vercel.com/) and click **Add New...** -> **Project**.
2. Import your Git repository.
3. In the project configuration screen:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click **Edit** and select **`frontend`**
   - **Build Command**: `npm run build` (or `tsc && vite build`)
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
4. Expand **Environment Variables** and add:

| Key | Value | Notes |
|---|---|---|
| `VITE_API_URL` | `https://<your-render-backend-url>.onrender.com` | **No trailing slash**. Points all frontend requests to Render |

5. Click **Deploy**.
6. Once deployed, copy your Vercel URL (e.g. `https://sonar-intel.vercel.app`).
7. Return to your Render Web Service settings and ensure `FRONTEND_URL` or `ALLOWED_ORIGINS` includes this Vercel URL.

---

## 4. Hardcode Audits & Fixes Applied

The codebase has been refactored to eliminate all deployment-blocking hardcodes:

1. **Dynamic Backend Port Handling**:
   - Replaced hardcoded `port=8000` with Render's `$PORT` environment variable (`os.environ.get("PORT")`).
2. **PostgreSQL / SQLite Connection Normalization**:
   - Render's PostgreSQL connection URLs beginning with `postgres://` or `postgresql://` are automatically normalized to `postgresql+psycopg://`.
   - If no database is configured, the server connects instantly to local SQLite without 2-second timeout stalls.
3. **CORS Architecture**:
   - `CORS_ORIGIN_REGEX` automatically allows all `*.vercel.app` domains, ensuring both production and PR preview builds can communicate with the backend without CORS blocks.
4. **Resilient Model Resolution**:
   - `DrishtiDetector` automatically retrieves the model from Hugging Face (`Samyukta31/sonar_yolo`) if local weights are not present.
   - Startup initialization is deferred/non-blocking so Render health checks pass immediately within seconds.
5. **Frontend API URL Resolution**:
   - Image endpoints (`raw_image_url`, `processed_image_url`) and report downloads (`.csv`, `.geojson`, `.summary`) use `apiService.resolveUrl()` to prepend `VITE_API_URL`.
   - `frontend/vercel.json` provides SPA fallback routing to prevent 404s on page reloads.

---

## 5. Verification Checklist

- [ ] Visit `https://<your-backend>.onrender.com/api/health` -> should return `{"status": "healthy"}`.
- [ ] Visit `https://<your-backend>.onrender.com/docs` -> interactive Swagger documentation.
- [ ] Open your Vercel URL: the Placely dark operations console should load with real-time health indicator showing **API: ONLINE**.
- [ ] In the dashboard, click **Load Held-Out Benchmark (Viator-04)** to test end-to-end inference and target visualization.
