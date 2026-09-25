FROM python:3.11-slim

# Install system dependencies needed for OpenCV, LibGL, and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight PyTorch CPU wheel first for minimal build time and disk quota
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install application dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy application source code and models
COPY backend ./backend
COPY ml ./ml
COPY database ./database
COPY data ./data

# Runtime environment settings
ENV PORT=8000
ENV MODEL_PATH=ml/models/dristri/best_detector.pt
ENV INFERENCE_DEVICE=cpu

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
