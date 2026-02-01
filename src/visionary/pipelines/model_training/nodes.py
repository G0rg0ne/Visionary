"""
Model training nodes for the Visionary pipeline.

Trains two multi-output quantile models: alpha=0.1 ("Chance of price drop") and
alpha=0.9 ("Risk of price hike"). Targets are delta_1, delta_2, ... (Δ = P_{t+n} - P_today)
as produced by the feature engineering pipeline (build_delta_targets).
"""
import re
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, List, Tuple
import shap
import matplotlib.pyplot as plt
import tempfile
import os
from dotenv import load_dotenv


def _get_target_columns(df: pd.DataFrame, horizon: int | None) -> List[str]:
    """
    Infer multi-target columns from data. Prefers delta_1, delta_2, ... (Δ = P_{t+n} - P_today)
    as produced by feature_engineering build_delta_targets; fallback to price_1, price_2, ...
    """
    # Prefer delta_1, delta_2, ... (matches feature_engineering build_delta_targets)
    pattern_delta = re.compile(r"^delta_(\d+)$")
    candidates = [c for c in df.columns if pattern_delta.match(c)]
    if candidates:
        return sorted(candidates, key=lambda x: int(pattern_delta.match(x).group(1)))
    # Fallback: price_1, price_2, ...
    pattern = re.compile(r"^price_(\d+)$")
    candidates = [c for c in df.columns if pattern.match(c)]
    if candidates:
        target_cols = sorted(candidates, key=lambda x: int(pattern.match(x).group(1)))
        return target_cols
    # Build from horizon if provided
    if horizon is not None and horizon >= 1:
        for prefix in ("delta_", "price_"):
            try_cols = [f"{prefix}{k}" for k in range(1, horizon)]
            if all(c in df.columns for c in try_cols):
                return try_cols
    return []


def log_shap_feature_importance(
    model: CatBoostRegressor,
    X_test: pd.DataFrame,
    params: Dict[str, Any],
    top_n: int = 5,
    multi_output: bool = True,
    artifact_subdir: str | None = None,
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
        artifact_subdir: Optional subdir under shap_feature_importance (e.g. "quantile_0.1")
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

    artifact_path = "shap_feature_importance"
    if artifact_subdir:
        artifact_path = f"{artifact_path}/{artifact_subdir}"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(tmp_file.name, artifact_path)
        os.unlink(tmp_file.name)

    plt.close()
    logger.info("SHAP feature importance plot logged to MLflow")


# Columns that define a unique flight (group-by); only columns present in the data are used.
UNIQUE_FLIGHTS_DEF_COLUMNS = [
    "origin",
    "destination",
    "departure_date",
    "departure_time",
    "airline",
    "flight_duration",
]


def log_test_flights_plots(
    y_test: pd.DataFrame,
    y_test_pred_series: List[Tuple[np.ndarray, str]],
    X_test: pd.DataFrame,
    target_cols: List[str],
    params: Dict[str, Any],
    n_samples: int = 5,
    unique_flights_def_columns: List[str] | None = None,
) -> None:
    """
    Create N sample line plots, one per unique flight in the test set: today's price +
    ground truth vs predicted future prices. Log each plot under MLflow artifact path "test_flights/".

    Flights are identified by grouping on unique_flights_def_columns (only columns present
    in X_test are used). One row per unique flight is plotted.

    Green line = ground truth (today + delta_1..delta_H, i.e. actual prices).
    Additional lines = each prediction series (today + pred_delta), e.g. "Chance of price drop" (10th pctl),
    "Risk of price hike" (90th pctl).

    Args:
        y_test: Test targets (delta_1, delta_2, ... or legacy price_1, price_2, ...)
        y_test_pred_series: List of (predictions, label), each predictions shape (n_samples, horizon)
        X_test: Test features (must contain 'todays_price')
        target_cols: List of target column names in order
        params: Parameters dict (for random_state)
        n_samples: Number of unique flights to plot
        unique_flights_def_columns: Columns defining a unique flight for group-by; default from constant.
    """
    if "todays_price" not in X_test.columns:
        logger.warning("todays_price not in X_test, skipping test_flights plots")
        return
    horizon = len(target_cols)
    n_available = len(y_test)
    if n_available == 0:
        logger.warning("No test samples for test_flights plots")
        return

    key_cols = unique_flights_def_columns or UNIQUE_FLIGHTS_DEF_COLUMNS
    flight_key_cols = [c for c in key_cols if c in X_test.columns]
    if not flight_key_cols:
        flight_key_cols = ["origin", "destination", "airline", "flight_duration"]
        flight_key_cols = [c for c in flight_key_cols if c in X_test.columns]
    if not flight_key_cols:
        logger.warning("No flight key columns in X_test, sampling random rows for test_flights")
        flight_key_cols = None

    rng = np.random.default_rng(params.get("random_state", 42))
    if flight_key_cols:
        # One row per unique flight: group by flight keys and take one index per group
        combined = X_test[flight_key_cols].copy()
        combined["_idx"] = np.arange(len(combined))
        first_per_flight = combined.groupby(flight_key_cols, dropna=False)["_idx"].first()
        unique_indices = first_per_flight.values
        n_samples = min(n_samples, len(unique_indices))
        chosen_positions = rng.choice(len(unique_indices), size=n_samples, replace=False)
        indices = unique_indices[chosen_positions]
    else:
        n_samples = min(n_samples, n_available)
        indices = rng.choice(n_available, size=n_samples, replace=False)

    # x: 0 = today, 1..horizon = day 1..H
    x_points = np.arange(0, horizon + 1)

    # When targets are deltas (Δ = P_{t+n} - P_today), plot actual prices = today + delta
    is_delta = target_cols and target_cols[0].startswith("delta_")
    pred_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for plot_idx, idx in enumerate(indices):
        today_price = float(X_test["todays_price"].iloc[idx])
        gt_row = y_test.iloc[idx]
        if is_delta:
            gt_prices = [today_price] + [today_price + float(gt_row[col]) for col in target_cols]
        else:
            gt_prices = [today_price] + [float(gt_row[col]) for col in target_cols]

        title = f"Test flight {plot_idx + 1}: Price today and next {horizon} days"
        if flight_key_cols:
            row = X_test.iloc[idx]
            parts = [str(row[c]) for c in flight_key_cols]
            flight_id = " | ".join(parts)
            title = f"{title}\n{flight_id}"

        plt.figure(figsize=(8, 5))
        plt.plot(x_points, gt_prices, color="green", linewidth=2, label="Ground truth")
        for series_idx, (y_pred, label) in enumerate(y_test_pred_series):
            pred_row = y_pred[idx]
            if is_delta:
                pred_prices = [today_price] + [
                    today_price + float(pred_row[j]) for j in range(horizon)
                ]
            else:
                pred_prices = [today_price] + [float(pred_row[j]) for j in range(horizon)]
            color = pred_colors[series_idx % len(pred_colors)]
            plt.plot(x_points, pred_prices, color=color, linewidth=2, label=label)
        plt.xlabel("Day (0 = today)")
        plt.ylabel("Price")
        plt.title(title)
        plt.legend()
        plt.xticks(x_points)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"sample_{plot_idx}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(path, "test_flights")
        plt.close()

    logger.info(f"Logged {n_samples} test_flights plots to MLflow (test_flights/)")


def train_model(
    tickets_train_data: pd.DataFrame,
    tickets_test_data: pd.DataFrame,
    params: Dict[str, Any],
) -> None:
    """
    Train two CatBoost multi-output quantile models and log to MLflow:
    - alpha=0.1: "Chance of price drop" (10th percentile)
    - alpha=0.9: "Risk of price hike" (90th percentile)

    Targets are inferred from data: delta_1, delta_2, ... (or legacy price_1, price_2, ...).

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
            "No multi-target columns found. Expected delta_1, delta_2, ... (or legacy price_1, price_2, ...). "
            "Ensure feature pipeline build_delta_targets has run."
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

    base_params = {
        "iterations": params.get("n_estimators", 2500),
        "depth": params.get("max_depth", 10),
        "random_state": params.get("random_state", 42),
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
    base_params = {k: v for k, v in base_params.items() if v is not None}

    quantile_configs = [
        (0.1, "Chance of price drop", "model_quantile_0.1"),
        (0.9, "Risk of price hike", "model_quantile_0.9"),
    ]

    log_to_mlflow = params.get("log_to_mlflow", True)

    def _run_training_and_eval(alpha: float):
        loss = f"MultiQuantile:alpha={alpha}"
        catboost_params = {**base_params, "loss_function": loss, "eval_metric": loss}
        logger.info("Training CatBoost multi-output quantile model (alpha=%s)...", alpha)
        model = CatBoostRegressor(**catboost_params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            verbose=False,
        )
        logger.info("Making predictions (alpha=%s)...", alpha)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, y_train_pred, multioutput="uniform_average")
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred, multioutput="uniform_average"))
        train_r2 = r2_score(y_train, y_train_pred, multioutput="uniform_average")
        test_mae = mean_absolute_error(y_test, y_test_pred, multioutput="uniform_average")
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput="uniform_average"))
        test_r2 = r2_score(y_test, y_test_pred, multioutput="uniform_average")
        return model, train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2, y_test_pred

    if log_to_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("visionary_price_prediction (Multi-target)")

        with mlflow.start_run(description=params.get("run_description", "")):
            mlflow.log_params(base_params)
            mlflow.log_params({
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": len(X_train.columns),
                "n_categorical_features": len(categorical_features),
                "multi_output_horizon": len(target_cols),
                "quantile_alphas": [0.1, 0.9],
            })
            all_features = X_train.columns.tolist()
            features_metadata = pd.DataFrame({
                "feature": all_features,
                "type": ["categorical" if f in categorical_features else "numerical" for f in all_features],
            })
            mlflow.log_table(data=features_metadata, artifact_file="training_features.json")
            logger.info(f"Logged {len(all_features)} training features to MLflow")

            y_test_pred_series: List[Tuple[np.ndarray, str]] = []
            for alpha, label, artifact_path in quantile_configs:
                model, train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2, y_test_pred = _run_training_and_eval(alpha)
                suffix = f"_quantile_{alpha}"
                mlflow.log_metrics({
                    f"train_mae{suffix}": train_mae,
                    f"train_rmse{suffix}": train_rmse,
                    f"train_r2{suffix}": train_r2,
                    f"test_mae{suffix}": test_mae,
                    f"test_rmse{suffix}": test_rmse,
                    f"test_r2{suffix}": test_r2,
                })
                mlflow.catboost.log_model(model, artifact_path=artifact_path)
                feature_importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": model.get_feature_importance(),
                }).sort_values("importance", ascending=False)
                mlflow.log_table(
                    data=feature_importance,
                    artifact_file=f"feature_importance{suffix}.json",
                )
                log_shap_feature_importance(
                    model, X_test, params, top_n=5, multi_output=True,
                    artifact_subdir=f"quantile_{alpha}",
                )
                y_test_pred_series.append((y_test_pred, label))

            n_flight_samples = params.get("test_flights_n_samples", 5)
            unique_flights_def = params.get(
                "unique_flights_def_columns", UNIQUE_FLIGHTS_DEF_COLUMNS
            )
            log_test_flights_plots(
                y_test,
                y_test_pred_series,
                X_test,
                target_cols,
                params,
                n_samples=n_flight_samples,
                unique_flights_def_columns=unique_flights_def,
            )

            logger.info("Model training completed (quantile 0.1 + 0.9).")
            logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
    else:
        logger.info("MLflow logging disabled (log_to_mlflow=False)")
        for alpha, label, _ in quantile_configs:
            model, train_mae, train_rmse, train_r2, test_mae, test_rmse, test_r2, _ = _run_training_and_eval(alpha)
            logger.info("Model (alpha=%s) - Train RMSE: %.2f, Test RMSE: %.2f", alpha, train_rmse, test_rmse)
        logger.info("Model training completed (quantile 0.1 + 0.9).")
