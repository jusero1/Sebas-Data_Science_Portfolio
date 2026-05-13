"""
FastAPI inference service — Time Series Forecast.

Medidas de seguridad implementadas:
  - CORS con origins explícitos leídos desde variable de entorno (nunca "*")
  - Rate limiting por IP con slowapi (10 req/min en /predict)
  - Cabeceras de seguridad HTTP en todas las respuestas
  - Validación estricta de entrada con Pydantic v2
  - Errores internos nunca expuestos al cliente
  - Carga de configuración solo desde variables de entorno
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Carga .env si existe (desarrollo local); en Docker se usan variables del sistema
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuración desde variables de entorno (sin hardcoding) ─────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_best.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "10/minute")

# CORS: lista de orígenes permitidos separados por coma
# NUNCA usar "*" — restringe siempre a dominios conocidos
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

DEFAULT_HORIZON = 24
MIN_WINDOW = 168
MAX_WINDOW = 8760  # 1 año de datos horarios como límite razonable


# ── Rate limiter por dirección IP ─────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])


# ── Middleware de cabeceras de seguridad HTTP ─────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Añade cabeceras de seguridad estándar a todas las respuestas.
    Mitiga clickjacking, MIME sniffing y ataques XSS reflected.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS: fuerza HTTPS durante 1 año (solo efectivo en producción con TLS)
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


# ── Lifespan: carga de artefactos al arrancar ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando artefactos del modelo...")
    try:
        from inference import _load_artifacts

        _load_artifacts(MODEL_PATH, SCALER_PATH)
        logger.info("Artefactos cargados correctamente.")
    except Exception:
        # El error real se registra en el log del servidor, no se expone al cliente
        logger.warning(
            "No se pudo precargar el modelo; se cargará en la primera petición.",
            exc_info=True,
        )
    yield
    logger.info("Apagando API.")


# ── Instancia de FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="Time Series Forecast API",
    description="LSTM/GRU forecasting for household electric power consumption",
    version="0.1.0",
    lifespan=lifespan,
    # En producción, desactiva los docs si la API no es pública:
    # docs_url=None, redoc_url=None
)

# Orden de middleware importa: primero seguridad, luego CORS
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Lista explícita — NUNCA ["*"]
    allow_credentials=False,         # Sin cookies de sesión en esta API
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Modelos Pydantic con validación estricta ───────────────────────────────────
class ForecastRequest(BaseModel):
    historical_values: List[float] = Field(
        ...,
        min_length=MIN_WINDOW,
        max_length=MAX_WINDOW,
        description=(
            f"Lecturas horarias de consumo (kWh). "
            f"Mínimo {MIN_WINDOW}, máximo {MAX_WINDOW} valores."
        ),
    )
    horizon: int = Field(
        DEFAULT_HORIZON,
        ge=1,
        le=96,
        description="Horizonte de predicción en horas (1–96)",
    )

    @field_validator("historical_values")
    @classmethod
    def validate_power_values(cls, values: List[float]) -> List[float]:
        if any(v < 0 for v in values):
            raise ValueError("Los valores de consumo no pueden ser negativos.")
        if any(v > 20.0 for v in values):
            # 20 kW es un límite razonable para un hogar — rechaza datos aberrantes
            raise ValueError("Valor de consumo fuera del rango esperado (máx. 20 kWh).")
        return values


class ForecastResponse(BaseModel):
    forecast: List[float]
    horizon: int
    unit: str = "kWh"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check", tags=["Sistema"])
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Generar forecast multistep",
    tags=["Inferencia"],
)
@limiter.limit(RATE_LIMIT_PREDICT)
async def predict(request: Request, body: ForecastRequest):
    """
    Acepta una secuencia de lecturas históricas horarias y devuelve
    un forecast de `horizon` horas hacia adelante.
    """
    try:
        from inference import predict as run_inference

        result = run_inference(
            historical_values=body.historical_values,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            horizon=body.horizon,
        )
        return ForecastResponse(
            forecast=result["forecast"],
            horizon=result["horizon"],
        )
    except ValueError as exc:
        # ValueError de validación de negocio — seguro de exponer
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        # Logueamos el traceback completo en el servidor, NUNCA al cliente
        logger.exception("Error interno en inferencia de series temporales")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Contacta al administrador.",
        )


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Time Series Forecast API — visita /docs para Swagger UI."}


def start() -> None:
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
