"""Training script for Chest X-Ray pneumonia classification.

Usage:
    python src/train.py --model cnn --epochs 30 --batch-size 32
    python src/train.py --model resnet50 --epochs 20 --fine-tune-epochs 10
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from utils import (
    build_tf_dataset,
    classification_report_dict,
    compute_class_weights,
    find_optimal_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
tf.random.set_seed(SEED)


def build_cnn_baseline(img_size: int = 224) -> keras.Model:
    """Custom CNN — 3 Conv-Pool blocks followed by dense head."""
    return keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            # Block 1
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.25),
            # Head
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_baseline",
    )


def build_resnet50_feature_extractor(img_size: int = 224) -> keras.Model:
    """ResNet50 with frozen base — stage 1 of transfer learning."""
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.resnet50.preprocess_input(inputs * 255)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="resnet50_feature_extractor")


def unfreeze_top_layers(model: keras.Model, n_layers: int = 20) -> keras.Model:
    """Unfreeze the last `n_layers` of the ResNet50 base for fine-tuning."""
    base = model.get_layer("resnet50")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    frozen = sum(1 for l in base.layers if not l.trainable)
    trainable = sum(1 for l in base.layers if l.trainable)
    logger.info("Fine-tune: %d frozen, %d trainable layers in ResNet50 base", frozen, trainable)
    return model


def train(
    data_dir: str = "data/raw",
    model_name: str = "resnet50",
    img_size: int = 224,
    epochs: int = 20,
    fine_tune_epochs: int = 10,
    batch_size: int = 32,
    models_dir: str = "models",
) -> dict:
    """Full transfer learning pipeline with two-stage training and MLflow tracking."""

    mlflow.set_experiment("cv-pneumonia-classification")

    with mlflow.start_run(run_name=f"{model_name}_e{epochs}_ft{fine_tune_epochs}"):
        params = {
            "model": model_name,
            "img_size": img_size,
            "epochs": epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "batch_size": batch_size,
        }
        mlflow.log_params(params)

        # ── Data loaders ────────────────────────────────────────────────────
        train_ds = build_tf_dataset(f"{data_dir}/train", img_size, batch_size, augment=True)
        val_ds = build_tf_dataset(f"{data_dir}/val", img_size, batch_size, augment=False, shuffle=False)
        test_ds = build_tf_dataset(f"{data_dir}/test", img_size, batch_size, augment=False, shuffle=False)

        class_weights = compute_class_weights(f"{data_dir}/train")

        # ── Model selection ─────────────────────────────────────────────────
        if model_name == "cnn":
            model = build_cnn_baseline(img_size)
            fine_tune_epochs = 0  # CNN trains end-to-end
        elif model_name == "resnet50":
            model = build_resnet50_feature_extractor(img_size)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # ── Stage 1: Feature extraction ─────────────────────────────────────
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        Path(models_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = f"{models_dir}/{model_name}_best.h5"

        stage1_cbs = [
            callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_auc", mode="max"),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4),
            callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ]

        logger.info("Stage 1: training %s for %d epochs...", model_name, epochs)
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=stage1_cbs, class_weight=class_weights)

        # ── Stage 2: Fine-tuning (ResNet50 only) ────────────────────────────
        if fine_tune_epochs > 0 and model_name == "resnet50":
            model = unfreeze_top_layers(model, n_layers=20)
            model.compile(
                optimizer=keras.optimizers.Adam(1e-5),  # Very low LR for fine-tuning
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")],
            )
            logger.info("Stage 2: fine-tuning for %d epochs...", fine_tune_epochs)
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=fine_tune_epochs,
                callbacks=stage1_cbs,
                class_weight=class_weights,
            )

        # ── Evaluation ──────────────────────────────────────────────────────
        y_true, y_prob = [], []
        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch, verbose=0)
            y_prob.extend(preds.ravel().tolist())
            y_true.extend(y_batch.numpy().ravel().tolist())

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        best_thr = find_optimal_threshold(y_true, y_prob)
        metrics = classification_report_dict(y_true, y_prob, threshold=best_thr)
        metrics["threshold"] = round(best_thr, 2)

        logger.info("Test metrics: %s", metrics)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.keras.log_model(model, artifact_path="model")

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train X-Ray pneumonia classifier")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model", choices=["cnn", "resnet50"], default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    metrics = train(
        data_dir=args.data_dir,
        model_name=args.model,
        img_size=args.img_size,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        models_dir=args.models_dir,
    )
    print("\n=== Final Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
