"""Training script for LSTM/GRU electric power consumption forecasting.

Usage:
    python src/train.py --model lstm --epochs 50 --window 168
    python src/train.py --model gru --epochs 50 --batch-size 64
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from utils import (
    add_temporal_features,
    compute_metrics,
    create_sequences,
    fit_scaler,
    load_and_clean,
    train_test_split_temporal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_lstm(input_shape: tuple, output_len: int, units: int = 128, dropout: float = 0.2) -> keras.Model:
    """Stacked LSTM with two recurrent layers and dropout regularization."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units // 2),
            layers.Dropout(dropout),
            layers.Dense(output_len),
        ],
        name="stacked_lstm",
    )
    return model


def build_gru(input_shape: tuple, output_len: int, units: int = 64, dropout: float = 0.2) -> keras.Model:
    """Bidirectional GRU with BatchNormalization."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.GRU(units, return_sequences=True)),
            layers.BatchNormalization(),
            layers.Bidirectional(layers.GRU(units // 2)),
            layers.Dropout(dropout),
            layers.Dense(output_len),
        ],
        name="bidirectional_gru",
    )
    return model


def build_cnn_lstm(input_shape: tuple, output_len: int) -> keras.Model:
    """CNN-LSTM hybrid: Conv1D feature extraction followed by LSTM."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(output_len),
        ],
        name="cnn_lstm",
    )
    return model


MODEL_BUILDERS = {
    "lstm": build_lstm,
    "gru": build_gru,
    "cnn_lstm": build_cnn_lstm,
}


def train(
    csv_path: str,
    model_name: str = "lstm",
    input_window: int = 168,
    output_horizon: int = 24,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    models_dir: str = "models",
) -> dict:
    """Full training pipeline with MLflow tracking."""

    mlflow.set_experiment("ts-electric-forecast")

    with mlflow.start_run(run_name=f"{model_name}_w{input_window}_h{output_horizon}"):
        params = {
            "model": model_name,
            "input_window": input_window,
            "output_horizon": output_horizon,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        mlflow.log_params(params)

        # ── Data preparation ────────────────────────────────────────────────
        logger.info("Loading and preprocessing data...")
        df = load_and_clean(csv_path)
        df = add_temporal_features(df)
        train_df, val_df, test_df = train_test_split_temporal(df)

        scaler = fit_scaler(train_df, scaler_path=f"{models_dir}/scaler.joblib")
        train_scaled = scaler.transform(train_df)
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)

        X_train, y_train = create_sequences(train_scaled, input_window, output_horizon)
        X_val, y_val = create_sequences(val_scaled, input_window, output_horizon)
        X_test, y_test = create_sequences(test_scaled, input_window, output_horizon)

        logger.info("Shapes — X_train: %s | X_val: %s | X_test: %s", X_train.shape, X_val.shape, X_test.shape)

        # ── Model construction ──────────────────────────────────────────────
        build_fn = MODEL_BUILDERS[model_name]
        model = build_fn(input_shape=(input_window, X_train.shape[-1]), output_len=output_horizon)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"])
        model.summary(print_fn=logger.info)

        # ── Callbacks ──────────────────────────────────────────────────────
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = f"{models_dir}/{model_name}_best.h5"
        cbs = [
            callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ]

        # ── Training ───────────────────────────────────────────────────────
        logger.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cbs,
            verbose=1,
        )

        # ── Evaluation ─────────────────────────────────────────────────────
        y_pred_scaled = model.predict(X_test)

        # Inverse-transform only the target column (index 0)
        n_features = X_train.shape[-1]
        dummy = np.zeros((len(y_test.ravel()), n_features))
        dummy[:, 0] = y_test.ravel()
        y_true_inv = scaler.inverse_transform(dummy)[:, 0].reshape(y_test.shape)
        dummy[:, 0] = y_pred_scaled.ravel()
        y_pred_inv = scaler.inverse_transform(dummy)[:, 0].reshape(y_pred_scaled.shape)

        metrics = compute_metrics(y_true_inv, y_pred_inv)
        logger.info("Test metrics: %s", metrics)
        mlflow.log_metrics(metrics)
        mlflow.keras.log_model(model, artifact_path="model")

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM/GRU forecasting model")
    parser.add_argument("--csv", default="data/raw/household_power_consumption.txt")
    parser.add_argument("--model", choices=["lstm", "gru", "cnn_lstm"], default="lstm")
    parser.add_argument("--window", type=int, default=168, help="Input window in hours (default: 168 = 7 days)")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon in hours")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    metrics = train(
        csv_path=args.csv,
        model_name=args.model,
        input_window=args.window,
        output_horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        models_dir=args.models_dir,
    )
    print("\n=== Final Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
