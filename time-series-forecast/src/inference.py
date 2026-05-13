"""Inference module: load trained model and generate multi-step forecasts."""

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_model = None
_scaler = None


def _load_artifacts(model_path: str, scaler_path: str):
    """Lazy-load model and scaler (cached after first call)."""
    global _model, _scaler

    if _model is None:
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError("TensorFlow is required for inference.") from exc

        logger.info("Loading model from %s", model_path)
        _model = keras.models.load_model(model_path)

    if _scaler is None:
        logger.info("Loading scaler from %s", scaler_path)
        _scaler = joblib.load(scaler_path)

    return _model, _scaler


def predict(
    historical_values: List[float],
    model_path: str = "models/lstm_best.h5",
    scaler_path: str = "models/scaler.joblib",
    horizon: int = 24,
) -> dict:
    """
    Generate a multi-step forecast from a list of historical hourly readings.

    Args:
        historical_values: List of ``Global_active_power`` values (kWh) in
            chronological order. Must be at least as long as the model's
            input window (typically 168 = 7 days × 24 h).
        model_path: Path to the saved Keras model (.h5 or SavedModel dir).
        scaler_path: Path to the joblib-serialized MinMaxScaler.
        horizon: Number of future hours to forecast (default: 24).

    Returns:
        dict with keys ``forecast`` (list of floats) and ``horizon``.
    """
    model, scaler = _load_artifacts(model_path, scaler_path)

    n_features = scaler.n_features_in_
    input_len = model.input_shape[1]

    if len(historical_values) < input_len:
        raise ValueError(
            f"Need at least {input_len} historical values, got {len(historical_values)}."
        )

    # Use the last `input_len` values
    window = np.array(historical_values[-input_len:], dtype=np.float32)

    # Build a full-feature array (pad extra features with zeros — they're scaled independently)
    full = np.zeros((input_len, n_features), dtype=np.float32)
    full[:, 0] = window
    full_scaled = scaler.transform(full)

    X = full_scaled[np.newaxis, ...]  # shape: (1, input_len, n_features)
    y_scaled = model.predict(X, verbose=0)[0]  # shape: (horizon,)

    # Inverse-transform predictions
    dummy = np.zeros((horizon, n_features), dtype=np.float32)
    dummy[:, 0] = y_scaled
    y_inv = scaler.inverse_transform(dummy)[:, 0]

    forecast = np.clip(y_inv, a_min=0, a_max=None).tolist()
    logger.info("Forecast generated: %d steps, first=%.3f, last=%.3f", horizon, forecast[0], forecast[-1])

    return {"forecast": forecast, "horizon": horizon}


def batch_predict(
    sequences: List[List[float]],
    model_path: str = "models/lstm_best.h5",
    scaler_path: str = "models/scaler.joblib",
    horizon: int = 24,
) -> List[dict]:
    """Run inference over a list of input sequences (batch mode)."""
    return [predict(seq, model_path, scaler_path, horizon) for seq in sequences]
