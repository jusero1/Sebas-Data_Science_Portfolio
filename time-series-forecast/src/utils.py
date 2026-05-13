"""Utility functions for data loading, preprocessing, and sliding-window generation."""

import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

RAW_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)
FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
TARGET_COL = "Global_active_power"


def download_dataset(dest_dir: str = "data/raw") -> Path:
    """Download and unzip the UCI power consumption dataset."""
    import urllib.request
    import zipfile

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "dataset.zip"

    if not zip_path.exists():
        logger.info("Downloading dataset from UCI repository...")
        urllib.request.urlretrieve(RAW_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    csv_path = dest / "household_power_consumption.txt"
    logger.info("Dataset ready at %s", csv_path)
    return csv_path


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load raw CSV, parse datetime index, replace missing values, resample to 1H."""
    df = pd.read_csv(
        csv_path,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        dayfirst=True,
        infer_datetime_format=True,
        na_values=["?"],
        low_memory=False,
    )
    df = df.set_index("datetime").sort_index()

    # Forward-fill short gaps (≤ 1 h), drop larger ones
    df = df.asfreq("1T").interpolate(method="time", limit=60)
    df = df.dropna()

    # Resample to hourly mean
    df_hourly = df[FEATURE_COLS].resample("1H").mean()
    logger.info("Loaded %d hourly records from %s to %s", len(df_hourly), df_hourly.index[0], df_hourly.index[-1])
    return df_hourly


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features and rolling statistics."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Rolling statistics on the target
    for window in [24, 168]:
        df[f"rolling_mean_{window}h"] = (
            df[TARGET_COL].rolling(window, min_periods=1).mean()
        )

    return df


def train_test_split_temporal(
    df: pd.DataFrame, test_ratio: float = 0.15, val_ratio: float = 0.10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split preserving temporal order (no shuffling)."""
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))

    train = df.iloc[:val_start]
    val = df.iloc[val_start:test_start]
    test = df.iloc[test_start:]

    logger.info("Split — train: %d | val: %d | test: %d rows", len(train), len(val), len(test))
    return train, val, test


def fit_scaler(train: pd.DataFrame, scaler_path: str = "models/scaler.joblib") -> MinMaxScaler:
    """Fit MinMaxScaler on training data and persist it."""
    scaler = MinMaxScaler()
    scaler.fit(train)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)
    return scaler


def load_scaler(scaler_path: str = "models/scaler.joblib") -> MinMaxScaler:
    return joblib.load(scaler_path)


def create_sequences(
    data: np.ndarray,
    input_len: int = 168,
    output_len: int = 24,
    target_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences for supervised learning.

    Returns:
        X: shape (n_samples, input_len, n_features)
        y: shape (n_samples, output_len)
    """
    X, y = [], []
    total = len(data) - input_len - output_len + 1
    for i in range(total):
        X.append(data[i : i + input_len])
        y.append(data[i + input_len : i + input_len + output_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, MAPE and R² between true and predicted arrays."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4), "MAPE": round(float(mape), 2), "R2": round(float(r2), 4)}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download the UCI dataset")
    args = parser.parse_args()

    if args.download:
        download_dataset()
