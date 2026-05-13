# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — computer-vision-classification
#
# Cambios de seguridad:
#   - Base python:3.11-slim (parches de seguridad actualizados)
#   - Librerías de sistema (libglib, libsm, etc.) solo en runtime, no en builder
#   - Usuario appuser (uid/gid 1000) — sin privilegios root
#   - HEALTHCHECK con curl
#   - PYTHONDONTWRITEBYTECODE evita archivos .pyc innecesarios
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

LABEL maintainer="zebas.sr@gmail.com" \
      project="computer-vision-classification" \
      version="0.1.0"

WORKDIR /app

# Librerías de sistema requeridas por OpenCV y TensorFlow en runtime
# curl: para el HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios
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

ENV MODEL_PATH=/app/models/resnet50_finetuned.h5 \
    IMG_SIZE=224 \
    THRESHOLD=0.5 \
    LOG_LEVEL=info \
    API_PORT=8001

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--access-log"]
