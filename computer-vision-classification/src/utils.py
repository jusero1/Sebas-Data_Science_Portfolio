"""Data loading, augmentation, metrics, and Grad-CAM utilities for CV project."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}


def load_image(path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Load a single image, convert to RGB, resize, and normalize to [0, 1]."""
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def build_tf_dataset(
    data_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Build a tf.data.Dataset from a directory structured as:
        data_dir/
            NORMAL/    *.jpeg
            PNEUMONIA/ *.jpeg
    """
    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=CLASSES,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    if augment:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.15),
                tf.keras.layers.RandomZoom(0.10),
                tf.keras.layers.RandomTranslation(0.10, 0.10),
                tf.keras.layers.RandomBrightness(0.15),
            ]
        )
        dataset = dataset.map(
            lambda x, y: (augmentation(normalization(x), training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.cache().prefetch(tf.data.AUTOTUNE)


def compute_class_weights(data_dir: str) -> Dict[int, float]:
    """Compute inverse-frequency class weights to handle dataset imbalance."""
    counts = {i: len(list(Path(data_dir, cls).glob("*"))) for i, cls in enumerate(CLASSES)}
    total = sum(counts.values())
    n_classes = len(CLASSES)
    weights = {i: total / (n_classes * cnt) for i, cnt in counts.items()}
    logger.info("Class weights: %s", weights)
    return weights


def classification_report_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute accuracy, precision, recall, F1, and AUC for binary classification."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the classification threshold that maximises F1 score."""
    from sklearn.metrics import f1_score

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.1, 0.95, 0.05):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    logger.info("Optimal threshold: %.2f (F1=%.4f)", best_thr, best_f1)
    return float(best_thr)


def grad_cam(
    model,
    img_array: np.ndarray,
    last_conv_layer_name: str = "conv5_block3_out",
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single image.

    Args:
        model: Keras model.
        img_array: Preprocessed image, shape (1, H, W, 3), values in [0, 1].
        last_conv_layer_name: Name of the last convolutional layer.

    Returns:
        Heatmap as numpy array with values in [0, 1], shape (H, W).
    """
    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_class = tf.cast(predictions[0] > 0.5, tf.float32)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap
