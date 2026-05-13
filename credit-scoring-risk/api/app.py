"""FastAPI credit scoring and risk segmentation service."""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_best.joblib")
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/clustering_scaler.joblib")
PCA_PATH = os.getenv("PCA_PATH", "models/clustering_pca.joblib")
KMEANS_PATH = os.getenv("KMEANS_PATH", "models/kmeans_k4.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading scoring artifacts...")
    try:
        from inference import _load_scoring_artifacts
        _load_scoring_artifacts(MODEL_PATH, PIPELINE_PATH)
        logger.info("Artifacts loaded.")
    except Exception as exc:
        logger.warning("Could not pre-load artifacts: %s", exc)
    yield
    logger.info("API shutdown.")


app = FastAPI(
    title="Credit Scoring API",
    description="Predict probability of default, credit score, and risk segment for loan applicants",
    version="0.1.0",
    lifespan=lifespan,
)


class ApplicantFeatures(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=50, examples=[0.52])
    age: int = Field(..., ge=18, le=110, examples=[45])
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, le=100, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., ge=0, examples=[0.38])
    MonthlyIncome: Optional[float] = Field(None, ge=0, examples=[5500])
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, examples=[8])
    NumberOfTimes90DaysLate: int = Field(..., ge=0, le=100, examples=[0])
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, examples=[1])
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, le=100, alias="NumberOfTime60-89DaysPastDueNotWorse")
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


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/score", response_model=ScoreResponse, summary="Score a credit applicant")
def score(applicant: ApplicantFeatures):
    """
    Compute the probability of default, credit score (300-850), risk segment,
    and SHAP-based feature contributions for a loan applicant.
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
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail="Internal scoring error.") from exc


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Credit Scoring API — visit /docs for Swagger UI."}


def start():
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8002, reload=False)
