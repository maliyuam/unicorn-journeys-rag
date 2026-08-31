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
    # Model cache MUST point somewhere the non-root user can write. Left to its
    # default this resolves under the user's home; pinning it here keeps it on
    # the volume mounted by docker-compose so restarts do not re-download, and
    # guarantees the write cannot land in the root-owned /app. A failed mkdir
    # here does not crash — it silently downgrades retrieval to TF-IDF.
    MODEL_CACHE_DIR=/home/app/.cache/unicorn-rag \
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
    && mkdir -p /app/data/index /app/data/transcripts \
                /home/app/.cache/unicorn-rag /home/app/.cache/huggingface \
    && chown -R app:app /app/data /home/app
USER app

EXPOSE 8017

# Single worker by design: the in-process job manager (backend/jobs.py) holds
# job state in memory, so a second worker would not see jobs the first started.
# For multi-worker deployments, swap in a shared queue first — see README.
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8017"]
