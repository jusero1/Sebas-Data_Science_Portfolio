# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — time-series-forecast
#
# Cambios de seguridad respecto a la versión original:
#   - Base: python:3.11-slim (últimos parches de seguridad vs 3.10)
#   - Multi-stage: el stage runtime NO incluye gcc ni herramientas de build
#   - Usuario sin privilegios (uid/gid 1000): nunca root en producción
#   - curl instalado solo en runtime para el HEALTHCHECK (mínimo impacto)
#   - Variables de entorno no hardcodeadas — se leen en tiempo de ejecución
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Instalar dependencias de compilación en el stage builder (NO en runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instalar paquetes en el directorio del usuario para copiarlos al runtime
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Metadatos de la imagen
LABEL maintainer="zebas.sr@gmail.com" \
      project="time-series-forecast" \
      version="0.1.0"

WORKDIR /app

# Instalar curl para el HEALTHCHECK (mínimo y sin herramientas de build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario sin privilegios ANTES de copiar archivos
# Razón: ejecutar como root en producción viola el principio de mínimo privilegio
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid 1000 --no-create-home appuser

# Copiar paquetes instalados desde el builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar solo el código de producción (src/ y api/)
# Los tests, notebooks y datos están excluidos por .dockerignore
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/

# Directorio para modelos (se monta como volumen en producción)
RUN mkdir -p /app/models && chown appuser:appuser /app/models

# Cambiar al usuario sin privilegios
USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Variables de entorno — los valores se sobreescriben en docker-compose o k8s
ENV MODEL_PATH=/app/models/lstm_best.h5 \
    SCALER_PATH=/app/models/scaler.joblib \
    LOG_LEVEL=info \
    API_PORT=8000

EXPOSE 8000

# HEALTHCHECK — curl es más fiable que python -c en producción
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log"]
