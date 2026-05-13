"""Utility functions for credit risk: WoE/IV, binning, metrics, PSI, scorecard."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "SeriousDlqin2yrs"

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

NUMERIC_COLS_WITH_MISSING = ["MonthlyIncome", "NumberOfDependents"]

OUTLIER_CAPS = {
    "RevolvingUtilizationOfUnsecuredLines": 1.0,
    "DebtRatio": 10.0,
    "MonthlyIncome": 30000.0,
    "NumberOfTime30-59DaysPastDueNotWorse": 10,
    "NumberOfTimes90DaysLate": 10,
    "NumberOfTime60-89DaysPastDueNotWorse": 10,
}


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw Kaggle CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path, index_col=0)
    df = df[FEATURE_COLS + [TARGET]].copy()
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with median (computed on non-null values)."""
    df = df.copy()
    for col in NUMERIC_COLS_WITH_MISSING:
        median = df[col].median()
        n_missing = df[col].isna().sum()
        df[col] = df[col].fillna(median)
        logger.info("Imputed %d missing values in '%s' with median=%.2f", n_missing, col, median)
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme values at domain-knowledge thresholds."""
    df = df.copy()
    for col, cap in OUTLIER_CAPS.items():
        if col in df.columns:
            n_capped = (df[col] > cap).sum()
            df[col] = df[col].clip(upper=cap)
            if n_capped > 0:
                logger.info("Capped %d outliers in '%s' at %.2f", n_capped, col, cap)
    return df


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute Weight of Evidence (WoE) and Information Value (IV) for a feature.

    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = Σ (Distribution of Events - Distribution of Non-Events) × WoE

    Returns:
        woe_table: DataFrame with bins, WoE, and IV contribution.
        iv: Total Information Value for the feature.
    """
    df_temp = df[[feature, target]].copy()
    df_temp["bin"] = pd.qcut(df_temp[feature], q=n_bins, duplicates="drop")

    grouped = df_temp.groupby("bin")[target].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    grouped["dist_events"] = grouped["events"] / total_events
    grouped["dist_non_events"] = grouped["non_events"] / total_non_events

    epsilon = 1e-6
    grouped["woe"] = np.log(
        (grouped["dist_events"] + epsilon) / (grouped["dist_non_events"] + epsilon)
    )
    grouped["iv_contribution"] = (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]
    iv = grouped["iv_contribution"].sum()

    return grouped.reset_index(), iv


def compute_all_iv(df: pd.DataFrame) -> pd.Series:
    """Compute IV for all features and return a ranked Series."""
    ivs = {}
    for col in FEATURE_COLS:
        try:
            _, iv = compute_woe_iv(df, col)
            ivs[col] = iv
        except Exception as exc:
            logger.warning("IV computation failed for '%s': %s", col, exc)
            ivs[col] = 0.0
    return pd.Series(ivs).sort_values(ascending=False)


def compute_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1."""
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    return round(2 * auc - 1, 4)


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between CDF of events and non-events."""
    from scipy.stats import ks_2samp
    probs_events = y_prob[y_true == 1]
    probs_non_events = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(probs_events, probs_non_events)
    return round(float(ks_stat), 4)


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI) — measures drift between two distributions.

    PSI < 0.1:  Insignificant change
    PSI 0.1-0.25: Moderate shift (monitor)
    PSI > 0.25: Significant drift (retrain)
    """
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)

    epsilon = 1e-6
    psi = np.sum(
        (actual_pct - expected_pct) * np.log((actual_pct + epsilon) / (expected_pct + epsilon))
    )
    return round(float(psi), 4)


def prob_to_score(prob: float, pdo: float = 50, ref_score: float = 600, ref_odds: float = 19.0) -> int:
    """
    Convert a default probability to a credit score (FICO-like scale).

    Args:
        prob: Predicted probability of default.
        pdo: Points to Double the Odds.
        ref_score: Score at the reference odds.
        ref_odds: Odds ratio at the reference score (non-events / events).
    """
    factor = pdo / np.log(2)
    offset = ref_score - factor * np.log(ref_odds)
    odds = (1 - prob) / (prob + 1e-9)
    score = int(np.clip(offset + factor * np.log(odds + 1e-9), 300, 850))
    return score


def assign_risk_segment(score: int) -> str:
    """Map credit score to a risk label."""
    if score >= 700:
        return "BAJO"
    elif score >= 620:
        return "NEAR_PRIME"
    elif score >= 550:
        return "SUBPRIME"
    else:
        return "ALTO"


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific engineered features."""
    df = df.copy()
    df["total_late_payments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["high_utilization"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.7).astype(int)
    income = df["MonthlyIncome"].replace(0, np.nan)
    df["debt_per_dependent"] = df["DebtRatio"] / (df["NumberOfDependents"] + 1)
    return df
