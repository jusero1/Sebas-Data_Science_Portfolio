"""Unit tests for time series utility functions."""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import add_temporal_features, compute_metrics, create_sequences, train_test_split_temporal


def make_dummy_df(n: int = 1000) -> pd.DataFrame:
    """Create a minimal DataFrame with a datetime index and one column."""
    idx = pd.date_range("2020-01-01", periods=n, freq="1H")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Global_active_power": rng.uniform(0.5, 5.0, n),
            "Global_reactive_power": rng.uniform(0.0, 0.5, n),
            "Voltage": rng.uniform(230, 250, n),
            "Global_intensity": rng.uniform(1, 20, n),
            "Sub_metering_1": rng.uniform(0, 40, n),
            "Sub_metering_2": rng.uniform(0, 40, n),
            "Sub_metering_3": rng.uniform(0, 40, n),
        },
        index=idx,
    )


class TestTemporalFeatures:
    def test_columns_added(self):
        df = make_dummy_df(200)
        result = add_temporal_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]:
            assert col in result.columns

    def test_sine_cosine_range(self):
        df = make_dummy_df(200)
        result = add_temporal_features(df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_is_weekend_binary(self):
        df = make_dummy_df(200)
        result = add_temporal_features(df)
        assert set(result["is_weekend"].unique()).issubset({0, 1})


class TestTrainTestSplit:
    def test_no_overlap(self):
        df = make_dummy_df(1000)
        train, val, test = train_test_split_temporal(df, test_ratio=0.15, val_ratio=0.10)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_correct_sizes(self):
        df = make_dummy_df(1000)
        train, val, test = train_test_split_temporal(df, test_ratio=0.15, val_ratio=0.10)
        assert len(train) + len(val) + len(test) == 1000

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(Exception):
            train_test_split_temporal(df)


class TestCreateSequences:
    def test_output_shapes(self):
        data = np.random.rand(500, 7).astype(np.float32)
        X, y = create_sequences(data, input_len=24, output_len=6)
        assert X.shape[1] == 24
        assert X.shape[2] == 7
        assert y.shape[1] == 6

    def test_number_of_samples(self):
        n, input_len, output_len = 200, 24, 6
        data = np.random.rand(n, 3).astype(np.float32)
        X, y = create_sequences(data, input_len=input_len, output_len=output_len)
        expected = n - input_len - output_len + 1
        assert len(X) == expected

    def test_insufficient_data_raises(self):
        data = np.random.rand(10, 3).astype(np.float32)
        with pytest.raises(Exception):
            create_sequences(data, input_len=20, output_len=5)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y)
        assert metrics["MAE"] == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R2"] == pytest.approx(1.0)

    def test_metric_keys_present(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        metrics = compute_metrics(y_true, y_pred)
        for key in ["MAE", "RMSE", "MAPE", "R2"]:
            assert key in metrics

    def test_positive_mae_on_imperfect_predictions(self):
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([1.0, 3.0])
        assert compute_metrics(y_true, y_pred)["MAE"] > 0
