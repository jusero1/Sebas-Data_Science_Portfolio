# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — credit-scoring-risk
#
# Cambios de seguridad:
#   - python:3.11-slim — mínima superficie de ataque
#   - Multi-stage: herramientas de build aisladas en el stage builder
#   - Usuario sin privilegios (uid/gid 1000)
#   - HEALTHCHECK con curl
#   - Ningún secreto ni credencial está en la imagen
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="jssegurarod96@gmail.com" \
      project="credit-scoring-risk" \
      version="0.1.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios
# Los modelos de scoring manejan datos financieros — principio de mínimo privilegio
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid 1000 --no-create-home appuser

COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/

RUN mkdir -p /app/models && chown appuser:appuser /app/models

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV MODEL_PATH=/app/models/xgboost_best.joblib \
    PIPELINE_PATH=/app/models/preprocessing_pipeline.joblib \
    SCALER_PATH=/app/models/clustering_scaler.joblib \
    PCA_PATH=/app/models/clustering_pca.joblib \
    KMEANS_PATH=/app/models/kmeans_k4.joblib \
    LOG_LEVEL=info \
    API_PORT=8002

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "1", \
     "--access-log"]
