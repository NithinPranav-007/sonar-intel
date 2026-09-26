"""
SONAR-INTEL FastAPI Application Entry Point.
"""

import os
import sys
from pathlib import Path
import datetime

# Strict memory and thread allocation constraints for 512MB Render free tier
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# Ensure project repository root is always in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
except Exception:
    pass

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:
    pass

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
    version="1.0.2",
    lifespan=lifespan
)

# CORS Middleware for React Frontend (Local & Vercel Deployments)
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
frontend_url = os.environ.get("FRONTEND_URL", "")

default_origins = [
    "https://frontend-sigma-bay-90.vercel.app",
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

raw_regex = os.environ.get("CORS_ORIGIN_REGEX", r"^https:\/\/.*\.vercel\.app$")
# Normalize double-escaped dots from YAML
cors_regex = raw_regex.replace(r"\\.", r"\.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else unique_origins,
    allow_origin_regex=cors_regex,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

from fastapi.responses import JSONResponse, Response
from fastapi import Request

@app.middleware("http")
async def ensure_cors_middleware(request: Request, call_next):
    """Guarantees CORS headers on every response, including errors and preflight OPTIONS."""
    origin = request.headers.get("origin")
    if request.method == "OPTIONS":
        response = Response(status_code=200)
    else:
        try:
            response = await call_next(request)
        except Exception as exc:
            response = JSONResponse(
                status_code=500,
                content={"detail": f"Internal server error: {str(exc)}"}
            )

    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Expose-Headers"] = "*"
    elif not response.headers.get("Access-Control-Allow-Origin"):
        response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure unhandled 500 errors always include CORS headers so browsers see real error details."""
    origin = request.headers.get("origin")
    headers = {
        "Access-Control-Allow-Origin": origin if origin else "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    }
    if origin:
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers=headers
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
        "version": "1.0.2-streaming",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", os.environ.get("BACKEND_PORT", 8000)))
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=reload)
