"""FastAPI inference service for time series forecasting."""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/lstm_best.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
DEFAULT_HORIZON = 24
MIN_WINDOW = 168


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts on startup...")
    try:
        from inference import _load_artifacts
        _load_artifacts(MODEL_PATH, SCALER_PATH)
        logger.info("Artifacts loaded successfully.")
    except Exception as exc:
        logger.warning("Could not pre-load model: %s — will load on first request.", exc)
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Time Series Forecast API",
    description="LSTM/GRU forecasting for household electric power consumption",
    version="0.1.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    historical_values: List[float] = Field(
        ...,
        min_length=MIN_WINDOW,
        description=f"Hourly Global_active_power readings (kWh). Minimum {MIN_WINDOW} values.",
        examples=[[3.2, 3.0, 2.8]],
    )
    horizon: int = Field(DEFAULT_HORIZON, ge=1, le=96, description="Forecast horizon in hours")

    @field_validator("historical_values")
    @classmethod
    def check_non_negative(cls, values: List[float]) -> List[float]:
        if any(v < 0 for v in values):
            raise ValueError("Power consumption values must be non-negative.")
        return values


class ForecastResponse(BaseModel):
    forecast: List[float]
    horizon: int
    unit: str = "kWh"
    model: str


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ForecastResponse, summary="Generate multi-step forecast")
def predict(request: ForecastRequest):
    """
    Accepts a sequence of historical hourly power readings and returns a
    multi-step forecast for the next `horizon` hours.
    """
    try:
        from inference import predict as run_inference

        result = run_inference(
            historical_values=request.historical_values,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            horizon=request.horizon,
        )
        return ForecastResponse(
            forecast=result["forecast"],
            horizon=result["horizon"],
            model=MODEL_PATH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal inference error.") from exc


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Time Series Forecast API — visit /docs for Swagger UI."}


def start():
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)
