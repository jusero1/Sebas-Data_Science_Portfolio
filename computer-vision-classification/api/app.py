"""FastAPI inference service for chest X-ray pneumonia classification."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet50_finetuned.h5")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
MAX_FILE_SIZE_MB = 5
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading model from %s", MODEL_PATH)
    try:
        from inference import _load_model
        _load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.warning("Could not pre-load model: %s", exc)
    yield
    logger.info("API shutdown.")


app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    description="Binary classification of chest X-ray images: NORMAL vs PNEUMONIA",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    gradcam_base64: str | None = None


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse, summary="Classify chest X-ray image")
async def predict(file: UploadFile = File(..., description="Chest X-ray image (JPEG or PNG, max 5 MB)")):
    """
    Upload a chest X-ray image and receive a prediction (NORMAL or PNEUMONIA)
    along with the predicted probability and an optional Grad-CAM heatmap.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum allowed: {MAX_FILE_SIZE_MB} MB.")

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
    except Exception as exc:
        logger.exception("Prediction failed for file %s", file.filename)
        raise HTTPException(status_code=500, detail="Internal inference error.") from exc


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Chest X-Ray Classification API — visit /docs for Swagger UI."}
