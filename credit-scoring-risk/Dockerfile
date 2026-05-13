# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/xgboost_best.joblib
ENV PIPELINE_PATH=/app/models/preprocessing_pipeline.joblib
ENV SCALER_PATH=/app/models/clustering_scaler.joblib
ENV KMEANS_PATH=/app/models/kmeans_k4.joblib

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8002/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
