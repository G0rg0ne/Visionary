"""
Model training nodes for the Visionary pipeline.

Trains a multi-output (vector) regression model: targets are price_1, price_2, ...
as produced by the feature engineering pipeline (build_target_vector).
"""
import re
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List
import shap
import matplotlib.pyplot as plt
import tempfile
import os
from dotenv import load_dotenv


def _get_target_columns(df: pd.DataFrame, horizon: int | None) -> List[str]:
    """
    Infer multi-target columns from data. Supports price_1, price_2, ... (feature pipeline)
    or price_d1, price_d2, ... (alternative naming).
    """
    # Prefer price_1, price_2, ... (matches feature_engineering build_target_vector)
    pattern = re.compile(r"^price_(\d+)$")
    candidates = [c for c in df.columns if pattern.match(c)]
    if candidates:
        target_cols = sorted(candidates, key=lambda x: int(pattern.match(x).group(1)))
        return target_cols
    # Fallback: price_d1, price_d2, ...
    pattern_d = re.compile(r"^price_d(\d+)$")
    candidates_d = [c for c in df.columns if pattern_d.match(c)]
    if candidates_d:
        return sorted(candidates_d, key=lambda x: int(pattern_d.match(x).group(1)))
    # Build from horizon if provided
    if horizon is not None and horizon >= 1:
        for prefix in ("price_", "price_d"):
            try_cols = [f"{prefix}{k}" for k in range(1, horizon + 1)]
            if all(c in df.columns for c in try_cols):
                return try_cols
    return []


def log_shap_feature_importance(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    params: Dict[str, Any],
    top_n: int = 5,
    multi_output: bool = True,
) -> None:
    """
    Generate and log SHAP feature importance plot to MLflow.

    For multi-output models, SHAP values are aggregated across all target dimensions.

    Args:
        model: Trained CatBoost model
        X_test: Test dataset features
        params: Dictionary containing parameters (for random_state)
        top_n: Number of top features to display (default: 5)
        multi_output: If True, aggregate SHAP over multiple target dimensions
    """
    logger.info("Computing SHAP values...")
    sample_size = min(100, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=params.get("random_state", 42))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Multi-output: shap_values shape (n_samples, n_features, n_targets)
    # Single-output: shape (n_samples, n_features)
    if multi_output and isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_shap_values = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap_values = np.abs(shap_values).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature": X_test_sample.columns,
        "shap_importance": mean_shap_values,
    }).sort_values("shap_importance", ascending=False)

    top_features = shap_importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features["shap_importance"].values)
    plt.yticks(range(len(top_features)), top_features["feature"].values)
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.title(f"Top {top_n} Features by SHAP Importance", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp_file.name, "shap_feature_importance")
        os.unlink(tmp_file.name)

    plt.close()
    logger.info("SHAP feature importance plot logged to MLflow")


def train_model(
    tickets_train_data: pd.DataFrame,
    tickets_test_data: pd.DataFrame,
    params: Dict[str, Any],
) -> None:
    """
    Train a CatBoost multi-output (vector) regression model and log to MLflow.

    Targets are inferred from data: price_1, price_2, ... (or price_d1, price_d2, ...).
    Uses MultiRMSE loss and uniform_average metrics across target dimensions.

    Args:
        tickets_train_data: Training dataset with features and target vector
        tickets_test_data: Test dataset with features and target vector
        params: Model training parameters (n_estimators, max_depth, random_state, etc.)
    """
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    horizon_param = params.get("multi_output_horizon")
    target_cols = _get_target_columns(tickets_train_data, horizon_param)
    if not target_cols:
        raise ValueError(
            "No multi-target columns found. Expected price_1, price_2, ... (or price_d1, price_d2, ...). "
            "Ensure feature pipeline build_target_vector has run."
        )

    categorical_features = tickets_train_data.select_dtypes(
        include=["object", "string", "bool"]
    ).columns.tolist()
    categorical_features = [c for c in categorical_features if c not in target_cols]

    for col in categorical_features:
        if col in tickets_train_data.columns and tickets_train_data[col].isna().any():
            tickets_train_data[col] = tickets_train_data[col].fillna("None").astype(str)
        if col in tickets_test_data.columns and tickets_test_data[col].isna().any():
            tickets_test_data[col] = tickets_test_data[col].fillna("None").astype(str)

    missing = [c for c in target_cols if c not in tickets_train_data.columns]
    if missing:
        raise ValueError(f"Target columns not found in training data: {missing}")

    X_train = tickets_train_data.drop(columns=target_cols)
    y_train = tickets_train_data[target_cols]
    X_test = tickets_test_data.drop(columns=target_cols)
    y_test = tickets_test_data[target_cols]

    categorical_features = X_train.select_dtypes(
        include=["object", "string", "bool"]
    ).columns.tolist()

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Target vector: {target_cols}")
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features: {len(X_train.columns) - len(categorical_features)}")

    catboost_params = {
        "iterations": params.get("n_estimators", 2500),
        "depth": params.get("max_depth", 10),
        "random_state": params.get("random_state", 42),
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "verbose": False,
        "grow_policy": "Lossguide",
        "max_leaves": 64,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.7,
        "random_strength": 1.5,
        "learning_rate": 0.02,
        "cat_features": categorical_features if categorical_features else None,
        "allow_writing_files": False,
    }
    catboost_params = {k: v for k, v in catboost_params.items() if v is not None}

    log_to_mlflow = params.get("log_to_mlflow", True)

    def _run_training_and_eval():
        logger.info("Training CatBoost multi-output model...")
        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            verbose=False,
        )
        logger.info("Making predictions...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, y_train_pred, multioutput="uniform_average")
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput="uniform_average"))
        train_r2 = r2_score(y_train, y_train_pred, multioutput="uniform_average")
        test_mae = mean_absolute_error(y_test, y_test_pred, multioutput="uniform_average")
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput="uniform_average"))
        test_r2 = r2_score(y_test, y_test_pred, multioutput="uniform_average")
        return model, train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2

    if log_to_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("visionary_price_prediction (Multi-target)")

        with mlflow.start_run(description=params.get("run_description", "")):
            mlflow.log_params(catboost_params)
            mlflow.log_params({
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": len(X_train.columns),
                "n_categorical_features": len(categorical_features),
                "multi_output_horizon": len(target_cols),
            })
            all_features = X_train.columns.tolist()
            features_metadata = pd.DataFrame({
                "feature": all_features,
                "type": ["categorical" if f in categorical_features else "numerical" for f in all_features],
            })
            mlflow.log_table(data=features_metadata, artifact_file="training_features.json")
            logger.info(f"Logged {len(all_features)} training features to MLflow")

            model, train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2 = _run_training_and_eval()

            mlflow.log_metrics({
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            })
            mlflow.catboost.log_model(model, artifact_path="model")
            feature_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.get_feature_importance(),
            }).sort_values("importance", ascending=False)
            mlflow.log_table(data=feature_importance, artifact_file="feature_importance.json")
            log_shap_feature_importance(model, X_test, params, top_n=5, multi_output=True)

            logger.info("Model training completed!")
            logger.info(f"Train RMSE: {train_rmse:.2f}, Train R²: {train_r2:.4f}")
            logger.info(f"Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.4f}")
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
    else:
        logger.info("MLflow logging disabled (log_to_mlflow=False)")
        model, train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2 = _run_training_and_eval()
        logger.info("Model training completed!")
        logger.info(f"Train RMSE: {train_rmse:.2f}, Train R²: {train_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.2f}, Test R²: {test_r2:.4f}")
