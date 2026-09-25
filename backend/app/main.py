"""
SONAR-INTEL FastAPI Application Entry Point.
"""

import os
import sys
from pathlib import Path
import datetime

# Ensure project repository root is always in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.database.connection import init_db
from backend.app.api.upload import router as upload_router
from backend.app.api.analysis import router as analysis_router
from backend.app.api.contacts import router as contacts_router
from backend.app.api.review import router as review_router
from backend.app.api.reports import router as reports_router
from backend.app.api.demo import router as demo_router
from backend.app.api.dashboard import router as dashboard_router
from backend.app.api.pipeline import router as pipeline_router
from backend.app.api.inference import router as inference_router

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="SONAR-INTEL API",
    description="AI-Powered Side-Scan Sonar Marine Debris & Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for React Frontend (Local & Vercel Deployments)
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
frontend_url = os.environ.get("FRONTEND_URL", "")

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
origins = list(default_origins)
if allowed_origins_env:
    origins.extend([o.strip() for o in allowed_origins_env.split(",") if o.strip()])
if frontend_url:
    origins.append(frontend_url.strip())

seen = set()
unique_origins = [x for x in origins if not (x in seen or seen.add(x))]
allow_all = "*" in unique_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else unique_origins,
    allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX", r"^https:\/\/.*\.vercel\.app$"),
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(upload_router)
app.include_router(analysis_router)
app.include_router(contacts_router)
app.include_router(review_router)
app.include_router(reports_router)
app.include_router(demo_router)
app.include_router(dashboard_router)
app.include_router(pipeline_router)
app.include_router(inference_router)

# Ensure storage directories exist
os.makedirs(settings.STORAGE_RAW_DIR, exist_ok=True)
os.makedirs(settings.STORAGE_PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.STORAGE_OUTPUTS_DIR, exist_ok=True)


@app.get("/api/health", tags=["System"])
def health_check():
    """Operational health probe."""
    return {
        "status": "healthy",
        "service": "SONAR-INTEL API",
        "database": "active",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", 8000)))
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=reload)
