
FROM node:20-slim AS frontend

WORKDIR /build

# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# --------------------------------------------------------------------------- #
# Stage 2 - runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim

# libgl / libglib are needed by opencv, which anomalib pulls in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000; everything below is owned by it.
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # No GPU on the free tier. The app reads this and pins itself to CPU.
    DEFECT_DETECTOR_DEVICE=cpu \
    # Keep model/config downloads inside the writable home directory.
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib


RUN pip install --no-cache-dir \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
# torch/torchvision are already satisfied above; pip leaves them alone because the
# pinned versions match.
RUN pip install --no-cache-dir -r requirements.txt

# Application code and trained artefacts. .dockerignore keeps the raw dataset,
# anomalib checkpoints, node_modules and the local venv out of this.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser outputs/ ./outputs/
COPY --chown=appuser:appuser deploy/samples/ ./deploy/samples/

# The compiled single-page app from stage 1.
COPY --from=frontend --chown=appuser:appuser /build/dist ./frontend/dist

RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser

USER appuser


EXPOSE 7860

# Entry point reads PORT (Render, Cloud Run, Fly, Heroku) and falls back to 7860.
# Exec form, so uvicorn is PID 1 and receives the platform's stop signal directly.
CMD ["python", "-m", "src.api"]
