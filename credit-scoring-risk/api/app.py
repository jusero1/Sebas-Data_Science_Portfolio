"""
FastAPI credit scoring and risk segmentation service.

Medidas de seguridad implementadas:
  - CORS con origins explícitos (leídos desde env var) — crítico en sistemas financieros
  - Rate limiting agresivo (5 req/min): el scoring es costoso y sensible
  - Cabeceras de seguridad HTTP en todas las respuestas
  - Validación estricta de rangos con Pydantic v2
  - Detalles internos y rutas de modelo nunca expuestos en respuestas de error
  - Carga de configuración solo desde variables de entorno
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# ── Configuración desde variables de entorno (sin hardcoding) ─────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_best.joblib")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/clustering_scaler.joblib")
PCA_PATH = os.getenv("PCA_PATH", "models/clustering_pca.joblib")
KMEANS_PATH = os.getenv("KMEANS_PATH", "models/kmeans_k4.joblib")
# El scoring crediticio tiene límite más bajo: previene enumeración de perfiles
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "5/minute")

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])


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
    logger.info("Precargando artefactos de scoring...")
    try:
        from inference import _load_scoring_artifacts

        _load_scoring_artifacts(MODEL_PATH, PIPELINE_PATH)
        logger.info("Artefactos cargados.")
    except Exception:
        logger.warning(
            "No se pudo precargar el modelo; se cargará en la primera petición.",
            exc_info=True,
        )
    yield
    logger.info("Apagando API.")


app = FastAPI(
    title="Credit Scoring API",
    description=(
        "Predice probabilidad de impago, score crediticio y segmento de riesgo. "
        "Solo para uso en entornos autorizados."
    ),
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


# ── Modelos Pydantic con validación de rangos ────────────────────────────────
class ApplicantFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0.0, le=50.0, examples=[0.52]
    )
    age: int = Field(..., ge=18, le=110, examples=[45])
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, le=100, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(..., ge=0.0, le=500.0, examples=[0.38])
    MonthlyIncome: Optional[float] = Field(None, ge=0.0, le=1_000_000.0, examples=[5500])
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, le=100, examples=[8])
    NumberOfTimes90DaysLate: int = Field(..., ge=0, le=100, examples=[0])
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, le=50, examples=[1])
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, le=100, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: Optional[int] = Field(None, ge=0, le=20, examples=[2])

    model_config = {"populate_by_name": True}


class SHAPFeature(BaseModel):
    feature: str
    impact: float


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: int
    risk_segment: str
    decision: str
    cluster_id: Optional[int] = None
    shap_top_features: List[SHAPFeature]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check", tags=["Sistema"])
def health():
    # No exponer rutas de modelos ni información de sistema en el health check
    return {"status": "ok"}


@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Evaluar solicitante de crédito",
    tags=["Inferencia"],
)
@limiter.limit(RATE_LIMIT_PREDICT)
async def score(request: Request, applicant: ApplicantFeatures):
    """
    Calcula la probabilidad de impago, score crediticio (300-850),
    segmento de riesgo y las variables más influyentes (SHAP).
    """
    try:
        from inference import score_applicant

        features = {
            "RevolvingUtilizationOfUnsecuredLines": applicant.RevolvingUtilizationOfUnsecuredLines,
            "age": applicant.age,
            "NumberOfTime30-59DaysPastDueNotWorse": applicant.NumberOfTime30_59DaysPastDueNotWorse,
            "DebtRatio": applicant.DebtRatio,
            "MonthlyIncome": applicant.MonthlyIncome,
            "NumberOfOpenCreditLinesAndLoans": applicant.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate": applicant.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines": applicant.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": applicant.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents": applicant.NumberOfDependents,
        }

        result = score_applicant(
            features=features,
            model_path=MODEL_PATH,
            pipeline_path=PIPELINE_PATH,
            scaler_path=SCALER_PATH,
            pca_path=PCA_PATH,
            kmeans_path=KMEANS_PATH,
        )
        return ScoreResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        # Nunca exponer detalles internos, rutas de modelos ni stack traces al cliente
        logger.exception("Error interno en scoring crediticio")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor. Contacta al administrador.",
        )


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Credit Scoring API — visita /docs para Swagger UI."}


def start() -> None:
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8002")),
        reload=False,
    )
