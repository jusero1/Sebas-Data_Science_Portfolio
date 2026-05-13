"""Unit tests for credit scoring utility functions."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    FEATURE_COLS,
    TARGET,
    assign_risk_segment,
    cap_outliers,
    compute_gini,
    compute_ks,
    compute_psi,
    create_derived_features,
    impute_missing,
    prob_to_score,
)


def make_dummy_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1.5, n),
            "age": rng.integers(18, 80, n),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 15, n),
            "DebtRatio": rng.uniform(0, 20, n),
            "MonthlyIncome": np.where(rng.random(n) < 0.18, np.nan, rng.uniform(500, 50000, n)),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 30, n),
            "NumberOfTimes90DaysLate": rng.integers(0, 15, n),
            "NumberRealEstateLoansOrLines": rng.integers(0, 10, n),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 15, n),
            "NumberOfDependents": np.where(rng.random(n) < 0.05, np.nan, rng.integers(0, 8, n)),
            TARGET: rng.integers(0, 2, n),
        }
    )


class TestImputation:
    def test_no_missing_after_imputation(self):
        df = make_dummy_df(100)
        result = impute_missing(df)
        assert result["MonthlyIncome"].isna().sum() == 0
        assert result["NumberOfDependents"].isna().sum() == 0

    def test_median_imputation_preserves_shape(self):
        df = make_dummy_df(100)
        result = impute_missing(df)
        assert result.shape == df.shape

    def test_non_missing_values_unchanged(self):
        df = make_dummy_df(100)
        non_missing_idx = df["MonthlyIncome"].notna()
        result = impute_missing(df)
        pd.testing.assert_series_equal(
            df.loc[non_missing_idx, "MonthlyIncome"],
            result.loc[non_missing_idx, "MonthlyIncome"],
        )


class TestOutlierCapping:
    def test_revolving_utilization_capped_at_1(self):
        df = make_dummy_df(100)
        result = cap_outliers(df)
        assert result["RevolvingUtilizationOfUnsecuredLines"].max() <= 1.0

    def test_monthly_income_capped(self):
        df = make_dummy_df(100)
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(5000)
        result = cap_outliers(df)
        assert result["MonthlyIncome"].max() <= 30000.0


class TestDerivedFeatures:
    def test_columns_created(self):
        df = make_dummy_df(100)
        df = impute_missing(df)
        result = create_derived_features(df)
        assert "total_late_payments" in result.columns
        assert "high_utilization" in result.columns
        assert "debt_per_dependent" in result.columns

    def test_total_late_payments_non_negative(self):
        df = make_dummy_df(100)
        df = impute_missing(df)
        result = create_derived_features(df)
        assert (result["total_late_payments"] >= 0).all()

    def test_high_utilization_binary(self):
        df = make_dummy_df(100)
        df = impute_missing(df)
        result = create_derived_features(df)
        assert set(result["high_utilization"].unique()).issubset({0, 1})


class TestRiskMetrics:
    def test_gini_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_gini(y_true, y_prob) == pytest.approx(1.0, abs=1e-3)

    def test_gini_range(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        gini = compute_gini(y_true, y_prob)
        assert -1.0 <= gini <= 1.0

    def test_ks_positive(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert compute_ks(y_true, y_prob) > 0

    def test_psi_stable_distribution(self):
        rng = np.random.default_rng(1)
        dist = rng.normal(0, 1, 1000)
        psi = compute_psi(dist, dist)
        assert psi < 0.01


class TestScorecardConversion:
    def test_score_range(self):
        for prob in [0.01, 0.05, 0.1, 0.5, 0.9]:
            score = prob_to_score(prob)
            assert 300 <= score <= 850

    def test_lower_prob_higher_score(self):
        score_low_risk = prob_to_score(0.01)
        score_high_risk = prob_to_score(0.50)
        assert score_low_risk > score_high_risk

    def test_assign_risk_segment_coverage(self):
        for score in [300, 450, 560, 640, 750]:
            segment = assign_risk_segment(score)
            assert segment in {"BAJO", "NEAR_PRIME", "SUBPRIME", "ALTO"}
