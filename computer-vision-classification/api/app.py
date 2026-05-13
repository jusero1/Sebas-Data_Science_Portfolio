"""
FastAPI inference service — Chest X-Ray Pneumonia Classification.

Medidas de seguridad implementadas:
  - CORS con origins explícitos (leídos desde env var)
  - Rate limiting por IP con slowapi
  - Cabeceras de seguridad HTTP en todas las respuestas
  - Validación de tipo MIME y tamaño máximo de archivo antes de procesar
  - Errores internos nunca expuestos al cliente
  - MAX_FILE_SIZE configurable desde variable de entorno
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuración desde variables de entorno ──────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet50_finetuned.h5")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "10/minute")

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# Tipos MIME aceptados — solo formatos médicos estándar
ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/jpg"})


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])


# ── Middleware de cabeceras de seguridad ──────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Precargando modelo desde %s", MODEL_PATH)
    try:
        from inference import _load_model

        _load_model(MODEL_PATH)
        logger.info("Modelo cargado correctamente.")
    except Exception:
        logger.warning(
            "No se pudo precargar el modelo; se cargará en la primera petición.",
            exc_info=True,
        )
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Clasificación binaria de radiografías torácicas: NORMAL vs PNEUMONIA",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Modelos de respuesta Pydantic ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    gradcam_base64: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check", tags=["Sistema"])
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Clasificar imagen de rayos X",
    tags=["Inferencia"],
)
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Imagen JPEG o PNG, máx. 5 MB"),
):
    """
    Clasifica una imagen de rayos X torácica como NORMAL o PNEUMONIA.
    Devuelve la probabilidad y un mapa Grad-CAM (base64) como overlay visual.
    """
    # Validación de tipo MIME antes de leer el cuerpo completo
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Tipo de archivo no soportado. Usa JPEG o PNG.",
        )

    image_bytes = await file.read()

    # Validación de tamaño después de leer (límite defensivo)
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande. Máximo permitido: {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        from inference import predict_image

        result = predict_image(
            image_bytes=image_bytes,
            model_path=MODEL_PATH,
            threshold=THRESHOLD,
            img_size=IMG_SIZE,
            return_gradcam=True,
        )
        return PredictionResponse(**result)
    except Exception:
        logger.exception("Error interno en clasificación de imagen")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Contacta al administrador.",
        )


@app.get("/", include_in_schema=False)
def root():
    return {"message": "CV Classification API — visita /docs para Swagger UI."}
