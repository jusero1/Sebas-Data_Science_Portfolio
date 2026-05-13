"""Training script: XGBoost / LightGBM credit scoring + KMeans segmentation.

Usage:
    python src/train.py --model xgboost --n-trials 50
    python src/train.py --model lightgbm --n-trials 50
    python src/train.py --mode clustering --n-clusters 4
"""

import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    FEATURE_COLS,
    TARGET,
    assign_risk_segment,
    cap_outliers,
    compute_gini,
    compute_ks,
    create_derived_features,
    impute_missing,
    load_dataset,
    prob_to_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DERIVED_COLS = ["total_late_payments", "high_utilization", "debt_per_dependent"]
ALL_FEATURES = FEATURE_COLS + DERIVED_COLS


def build_preprocessing_pipeline() -> Pipeline:
    """Preprocessing pipeline: imputation + capping + feature engineering embedded."""
    from sklearn.base import BaseEstimator, TransformerMixin

    class CreditPreprocessor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            df = pd.DataFrame(X, columns=FEATURE_COLS) if not isinstance(X, pd.DataFrame) else X.copy()
            df = impute_missing(df)
            df = cap_outliers(df)
            df = create_derived_features(df)
            return df[ALL_FEATURES].values

    return Pipeline([("credit_preprocessor", CreditPreprocessor())])


def train_gradient_boosting(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    n_trials: int = 50,
    models_dir: str = "models",
) -> dict:
    """Train XGBoost or LightGBM with Optuna hyperparameter search and 5-fold CV."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    mlflow.set_experiment("credit-scoring")

    with mlflow.start_run(run_name=f"{model_name}_optuna_{n_trials}trials"):
        X = df[FEATURE_COLS].copy()
        y = df[TARGET].values

        preprocessing = build_preprocessing_pipeline()
        X_processed = preprocessing.fit_transform(X)

        joblib.dump(preprocessing, f"{models_dir}/preprocessing_pipeline.joblib")
        logger.info("Preprocessing pipeline saved.")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        def objective(trial: "optuna.Trial") -> float:
            if model_name == "xgboost":
                import xgboost as xgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "scale_pos_weight": 14,  # Handles class imbalance (~1:14 ratio)
                    "random_state": SEED,
                    "eval_metric": "auc",
                    "use_label_encoder": False,
                }
                clf = xgb.XGBClassifier(**params)
            else:
                import lightgbm as lgb
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "class_weight": "balanced",
                    "random_state": SEED,
                    "verbose": -1,
                }
                clf = lgb.LGBMClassifier(**params)

            oof_scores = []
            for train_idx, val_idx in skf.split(X_processed, y):
                clf.fit(X_processed[train_idx], y[train_idx])
                prob = clf.predict_proba(X_processed[val_idx])[:, 1]
                oof_scores.append(roc_auc_score(y[val_idx], prob))

            return np.mean(oof_scores)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_auc = study.best_value
        logger.info("Best CV AUC: %.4f | Params: %s", best_auc, best_params)
        mlflow.log_params(best_params)

        # Retrain on full data with best params
        if model_name == "xgboost":
            import xgboost as xgb
            best_model = xgb.XGBClassifier(**best_params, scale_pos_weight=14, random_state=SEED, use_label_encoder=False)
        else:
            import lightgbm as lgb
            best_model = lgb.LGBMClassifier(**best_params, class_weight="balanced", random_state=SEED, verbose=-1)

        best_model.fit(X_processed, y)

        # Evaluation metrics
        y_prob_full = best_model.predict_proba(X_processed)[:, 1]
        gini = compute_gini(y, y_prob_full)
        ks = compute_ks(y, y_prob_full)

        metrics = {"cv_auc": round(best_auc, 4), "gini": gini, "ks_statistic": ks}
        mlflow.log_metrics(metrics)

        Path(models_dir).mkdir(parents=True, exist_ok=True)
        model_path = f"{models_dir}/{model_name}_best.joblib"
        joblib.dump(best_model, model_path)
        logger.info("Model saved to %s", model_path)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        return metrics


def train_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    models_dir: str = "models",
) -> pd.DataFrame:
    """
    Segment the customer portfolio using KMeans on behavioral features.

    Returns a DataFrame with segment assignments and cluster profiles.
    """
    mlflow.set_experiment("credit-clustering")

    with mlflow.start_run(run_name=f"kmeans_k{n_clusters}"):
        cluster_features = [
            "RevolvingUtilizationOfUnsecuredLines",
            "DebtRatio",
            "total_late_payments",
            "MonthlyIncome",
            "age",
        ]

        df_processed = impute_missing(df)
        df_processed = cap_outliers(df_processed)
        df_processed = create_derived_features(df_processed)

        X_cluster = df_processed[cluster_features].fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        pca = PCA(n_components=0.95, random_state=SEED)
        X_pca = pca.fit_transform(X_scaled)
        logger.info("PCA reduced to %d components (95%% variance)", X_pca.shape[1])

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
        labels = kmeans.fit_predict(X_pca)

        silhouette = _silhouette(X_pca, labels)
        logger.info("Silhouette score: %.4f", silhouette)
        mlflow.log_metrics({"n_clusters": n_clusters, "silhouette_score": silhouette})

        Path(models_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, f"{models_dir}/clustering_scaler.joblib")
        joblib.dump(pca, f"{models_dir}/clustering_pca.joblib")
        joblib.dump(kmeans, f"{models_dir}/kmeans_k{n_clusters}.joblib")

        df_result = df_processed.copy()
        df_result["cluster"] = labels
        profile = df_result.groupby("cluster")[cluster_features + [TARGET]].mean().round(3)
        logger.info("Cluster profiles:\n%s", profile.to_string())

        return df_result


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    return round(float(silhouette_score(X, labels, sample_size=10000, random_state=SEED)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit scoring training pipeline")
    parser.add_argument("--csv", default="data/raw/cs-training.csv")
    parser.add_argument("--mode", choices=["scoring", "clustering"], default="scoring")
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    df = load_dataset(args.csv)

    if args.mode == "scoring":
        metrics = train_gradient_boosting(df, args.model, args.n_trials, args.models_dir)
        print("\n=== Credit Scoring Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        result = train_clustering(df, args.n_clusters, args.models_dir)
        print(f"\nSegmentation complete. {args.n_clusters} clusters assigned to {len(result)} customers.")


if __name__ == "__main__":
    main()
