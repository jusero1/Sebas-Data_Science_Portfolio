"""Inference module: credit scoring, risk segmentation, and SHAP explanations."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_model = None
_pipeline = None
_scaler = None
_pca = None
_kmeans = None


def _load_scoring_artifacts(model_path: str, pipeline_path: str):
    global _model, _pipeline
    if _model is None:
        import joblib
        logger.info("Loading scoring model from %s", model_path)
        _model = joblib.load(model_path)
    if _pipeline is None:
        import joblib
        logger.info("Loading preprocessing pipeline from %s", pipeline_path)
        _pipeline = joblib.load(pipeline_path)
    return _model, _pipeline


def _load_clustering_artifacts(scaler_path: str, pca_path: str, kmeans_path: str):
    global _scaler, _pca, _kmeans
    if _scaler is None:
        import joblib
        _scaler = joblib.load(scaler_path)
        _pca = joblib.load(pca_path)
        _kmeans = joblib.load(kmeans_path)
    return _scaler, _pca, _kmeans


def score_applicant(
    features: Dict,
    model_path: str = "models/xgboost_best.joblib",
    pipeline_path: str = "models/preprocessing_pipeline.joblib",
    scaler_path: str = "models/clustering_scaler.joblib",
    pca_path: str = "models/clustering_pca.joblib",
    kmeans_path: str = "models/kmeans_k4.joblib",
    top_shap_k: int = 5,
) -> Dict:
    """
    Score a single credit applicant.

    Args:
        features: Dictionary mapping feature names to their values.
        top_shap_k: Number of top SHAP features to include in the response.

    Returns:
        dict with probability_of_default, score (int), risk_segment, decision, and SHAP explanations.
    """
    from utils import FEATURE_COLS, assign_risk_segment, prob_to_score

    model, pipeline = _load_scoring_artifacts(model_path, pipeline_path)

    df_input = pd.DataFrame([features])[FEATURE_COLS]
    X_processed = pipeline.transform(df_input)

    prob = float(model.predict_proba(X_processed)[0, 1])
    score = prob_to_score(prob)
    segment = assign_risk_segment(score)
    decision = "APROBAR" if score >= 600 else "REVISAR" if score >= 550 else "RECHAZAR"

    shap_features = _compute_shap(model, X_processed, top_k=top_shap_k)

    cluster_id = None
    try:
        cluster_features = [
            "RevolvingUtilizationOfUnsecuredLines", "DebtRatio",
            "total_late_payments", "MonthlyIncome", "age",
        ]
        scaler, pca, kmeans = _load_clustering_artifacts(scaler_path, pca_path, kmeans_path)
        from utils import cap_outliers, create_derived_features, impute_missing
        df_temp = impute_missing(df_input)
        df_temp = cap_outliers(df_temp)
        df_temp = create_derived_features(df_temp)
        X_clust = scaler.transform(df_temp[cluster_features].fillna(0))
        X_pca = pca.transform(X_clust)
        cluster_id = int(kmeans.predict(X_pca)[0])
    except Exception as exc:
        logger.warning("Clustering inference failed: %s", exc)

    return {
        "probability_of_default": round(prob, 4),
        "score": score,
        "risk_segment": segment,
        "decision": decision,
        "cluster_id": cluster_id,
        "shap_top_features": shap_features,
    }


def _compute_shap(model, X_processed: np.ndarray, top_k: int = 5) -> List[Dict]:
    """Return top-k SHAP feature impacts (signed contribution to log-odds)."""
    try:
        import shap
        from utils import ALL_FEATURES

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        feature_names = ALL_FEATURES[: len(sv)]
        impacts = sorted(zip(feature_names, sv.tolist()), key=lambda x: abs(x[1]), reverse=True)

        return [
            {"feature": feat, "impact": round(imp, 4)}
            for feat, imp in impacts[:top_k]
        ]
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return []
