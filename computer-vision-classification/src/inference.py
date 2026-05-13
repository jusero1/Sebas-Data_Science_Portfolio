"""Inference module: load model and classify chest X-ray images."""

import base64
import io
import logging
import os
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

CLASSES = ["NORMAL", "PNEUMONIA"]
_model = None


def _load_model(model_path: str):
    global _model
    if _model is None:
        from tensorflow import keras
        logger.info("Loading model from %s", model_path)
        _model = keras.models.load_model(model_path)
    return _model


def preprocess_image(image_bytes: bytes, img_size: int = 224) -> np.ndarray:
    """Decode image bytes, resize, normalize, and add batch dimension."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # shape: (1, H, W, 3)


def predict_image(
    image_bytes: bytes,
    model_path: str = "models/resnet50_finetuned.h5",
    threshold: float = 0.5,
    img_size: int = 224,
    return_gradcam: bool = True,
) -> Dict:
    """
    Classify a single chest X-ray image.

    Args:
        image_bytes: Raw bytes of a JPEG/PNG image.
        model_path: Path to the saved Keras model.
        threshold: Classification decision boundary.
        img_size: Target image size (px).
        return_gradcam: If True, include Grad-CAM overlay as base64 PNG.

    Returns:
        dict with prediction, probability, confidence, and optional gradcam_base64.
    """
    model = _load_model(model_path)
    img_array = preprocess_image(image_bytes, img_size)
    probability = float(model.predict(img_array, verbose=0)[0, 0])
    predicted_class = CLASSES[int(probability >= threshold)]

    confidence = (
        "HIGH" if probability > 0.85 or probability < 0.15
        else "MEDIUM" if probability > 0.70 or probability < 0.30
        else "LOW"
    )

    result: Dict = {
        "prediction": predicted_class,
        "probability": round(probability, 4),
        "confidence": confidence,
    }

    if return_gradcam:
        try:
            gradcam_b64 = _generate_gradcam_b64(model, img_array, img_size)
            result["gradcam_base64"] = gradcam_b64
        except Exception as exc:
            logger.warning("Grad-CAM generation failed: %s", exc)
            result["gradcam_base64"] = None

    logger.info("Prediction: %s (prob=%.4f, conf=%s)", predicted_class, probability, confidence)
    return result


def _generate_gradcam_b64(model, img_array: np.ndarray, img_size: int) -> str:
    """Generate Grad-CAM heatmap and return as base64-encoded PNG."""
    import cv2
    from utils import grad_cam

    heatmap = grad_cam(model, img_array)
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay on original image
    original = (img_array[0] * 255).astype(np.uint8)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

    _, buffer = cv2.imencode(".png", superimposed)
    return base64.b64encode(buffer).decode("utf-8")
