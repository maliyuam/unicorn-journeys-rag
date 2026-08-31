# Unicorn Journeys RAG — single-stage image, runs the API and the bundled UI.
#
#   docker build -t unicorn-rag .
#   docker run --rm -p 8017:8017 unicorn-rag
#
# With no credentials the app starts in offline extractive mode, so this image
# is useful with nothing configured. Pass credentials at run time (never bake
# them into the image):
#
#   docker run --rm -p 8017:8017 --env-file .env unicorn-rag

FROM python:3.12-slim

# Dependencies are pinned by requirements.txt; keep the image lean otherwise.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # fastembed caches its ONNX model here; see the volume note in the README.
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# Copy requirements first so dependency layers cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Run as a non-root user, and give it ownership of the paths written at
# runtime: the vector index, collected transcripts, and the model cache.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data/index /app/data/transcripts /home/app/.cache \
    && chown -R app:app /app/data /home/app
USER app

EXPOSE 8017

# Single worker by design: the in-process job manager (backend/jobs.py) holds
# job state in memory, so a second worker would not see jobs the first started.
# For multi-worker deployments, swap in a shared queue first — see README.
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8017"]
