"""Unit tests for computer vision utility functions."""

import io
import sys
import os
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    CLASSES,
    classification_report_dict,
    find_optimal_threshold,
    load_image,
)


def make_dummy_image_bytes(width: int = 256, height: int = 256, color: str = "RGB") -> bytes:
    """Create a solid-color image and return its bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestLoadImage:
    def test_output_shape(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8)).save(img_path)
        arr = load_image(str(img_path), img_size=224)
        assert arr.shape == (224, 224, 3)

    def test_normalized_range(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.fromarray(np.full((100, 100, 3), 200, dtype=np.uint8)).save(img_path)
        arr = load_image(str(img_path))
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_dtype_float32(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(img_path)
        arr = load_image(str(img_path))
        assert arr.dtype == np.float32


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.9, 0.9])
        metrics = classification_report_dict(y_true, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] == 1.0

    def test_metric_keys(self):
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.3, 0.7, 0.6, 0.4])
        metrics = classification_report_dict(y_true, y_prob)
        for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            assert key in metrics

    def test_metrics_in_valid_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        metrics = classification_report_dict(y_true, y_prob)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0.0 <= metrics[key] <= 1.0


class TestFindOptimalThreshold:
    def test_returns_float(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
        thr = find_optimal_threshold(y_true, y_prob)
        assert isinstance(thr, float)

    def test_threshold_in_valid_range(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        thr = find_optimal_threshold(y_true, y_prob)
        assert 0.0 <= thr <= 1.0

    def test_perfect_separation_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        thr = find_optimal_threshold(y_true, y_prob)
        # Any threshold between 0.2 and 0.8 gives F1=1.0
        assert thr < 0.8


class TestClassLabels:
    def test_classes_defined(self):
        assert "NORMAL" in CLASSES
        assert "PNEUMONIA" in CLASSES

    def test_binary_classes(self):
        assert len(CLASSES) == 2
