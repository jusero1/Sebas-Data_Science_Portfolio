# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/resnet50_finetuned.h5
ENV IMG_SIZE=224
ENV THRESHOLD=0.5

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8001/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
