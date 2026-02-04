"""
Model training nodes for the Visionary pipeline.

Trains AutoGluon time series forecasting models to predict flight prices
up to a configurable horizon. Expects per-flight train/test split (last N
timesteps per flight in test) so every flight has the same test length for evaluation.
"""
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from dotenv import load_dotenv
from loguru import logger
import mlflow


def train_autogluon_model(
    timeseries_train: pd.DataFrame,
    timeseries_test: pd.DataFrame,
    params: Dict[str, Any],
) -> TimeSeriesPredictor:
    """
    Train AutoGluon time series models on flight price data and return the
    fitted predictor. Expects per-flight split: each flight has same number of test timesteps.

    Args:
        timeseries_train: Training data with item_id, timestamp, target.
        timeseries_test: Test data (later timestamps) for evaluation.
        params: forecast_horizon, time_limit, presets, eval_metric, log_to_mlflow, path.

    Returns:
        Fitted TimeSeriesPredictor (best model is used for predictions).
    """
    load_dotenv()
    path = params.get("path") or "data/06_models/autogluon"
    Path(path).mkdir(parents=True, exist_ok=True)

    train_ts = TimeSeriesDataFrame.from_data_frame(
        timeseries_train,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    test_ts = TimeSeriesDataFrame.from_data_frame(
        timeseries_test,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    prediction_length = params.get("forecast_horizon", 7)
    eval_metric = params.get("eval_metric", "MAPE")
    known_covariates = params.get("known_covariates_names")
    if known_covariates is not None:
        known_covariates = [c for c in known_covariates if c in timeseries_train.columns]

    freq = params.get("freq", "D")
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=prediction_length,
        eval_metric=eval_metric,
        path=path,
        freq=freq,
        known_covariates_names=known_covariates or None,
        verbosity=params.get("verbosity", 2),
    )

    time_limit = params.get("time_limit", 3600)
    presets = params.get("presets", "medium_quality")

    logger.info(
        f"Training AutoGluon time series models: prediction_length={prediction_length}, time_limit={time_limit}, presets={presets}"
    )
    predictor.fit(
        train_data=train_ts,
        time_limit=time_limit,
        presets=presets,
    )

    # AutoGluon requires each evaluation series to have length > prediction_length (at least 8)
    min_length_for_eval = prediction_length + 1
    steps_per_item = test_ts.num_timesteps_per_item()
    valid_item_ids = steps_per_item[steps_per_item >= min_length_for_eval].index
    if len(valid_item_ids) < len(steps_per_item):
        bad = steps_per_item[steps_per_item < min_length_for_eval]
        raise ValueError(
            f"Cannot evaluate: {len(bad)} test series have fewer than {min_length_for_eval} timesteps "
            f"(need length > prediction_length={prediction_length}). "
            "Re-run the data_preparation_pipeline so timeseries_test is regenerated with test_timesteps >= 8."
        )
    test_ts_eval = test_ts.loc[valid_item_ids] if len(valid_item_ids) > 0 else None

    test_metrics = None
    leaderboard = None
    if test_ts_eval is not None and len(test_ts_eval) > 0:
        logger.info(
            f"Evaluating on test set ({test_ts_eval.num_items} series, {min_length_for_eval}+ steps each)"
        )
        test_metrics = predictor.evaluate(test_ts_eval)
        leaderboard = predictor.leaderboard(test_ts_eval, silent=True)
        logger.info(f"Test metrics: {test_metrics}")
        if leaderboard is not None and not leaderboard.empty:
            best_model = leaderboard.index[0]
            logger.info(f"Best model: {best_model}")
            logger.info(f"Leaderboard:\n{leaderboard.to_string()}")
    else:
        logger.warning(
            f"Skipping evaluation: no test time series have length >= {min_length_for_eval} (got {len(valid_item_ids)} items)"
        )

    log_to_mlflow = params.get("log_to_mlflow", True)
    if log_to_mlflow:
        _log_autogluon_to_mlflow(
            params=params,
            predictor=predictor,
            train_ts=train_ts,
            test_ts=test_ts_eval if test_ts_eval is not None else test_ts,
            test_metrics=test_metrics,
            leaderboard=leaderboard,
        )

    return predictor


def create_performance_plots(
    predictor: TimeSeriesPredictor,
    timeseries_test: pd.DataFrame,
    params: Dict[str, Any],
) -> Dict[str, plt.Figure]:
    """
    Generate prediction vs ground truth plots for sample flights.

    Samples a configurable number of flights from the test set, generates
    predictions for each, and returns one matplotlib figure per flight with
    ground truth (green), predictions (red), and metrics (MAPE, RMSE, MAE).

    Args:
        predictor: Fitted TimeSeriesPredictor.
        timeseries_test: Test DataFrame with item_id, timestamp, target.
        params: num_sample_plots, forecast_horizon, plot_metrics.

    Returns:
        Dictionary mapping item_id (str) to matplotlib Figure for catalog/MLflow.
    """
    test_ts = TimeSeriesDataFrame.from_data_frame(
        timeseries_test,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    prediction_length = params.get("forecast_horizon", 7)
    num_samples = params.get("num_sample_plots", 10)
    plot_metrics_list = params.get("plot_metrics", ["MAPE", "RMSE", "MAE"])

    # Require enough history for prediction
    min_length = prediction_length + 1
    steps_per_item = test_ts.num_timesteps_per_item()
    valid_ids = steps_per_item[steps_per_item >= min_length].index.tolist()
    if not valid_ids:
        logger.warning("No test series long enough for performance plots; skipping.")
        return {}

    # Sample up to num_samples flights
    rng = np.random.default_rng(params.get("plot_random_seed"))
    n = min(num_samples, len(valid_ids))
    sample_ids = list(rng.choice(valid_ids, size=n, replace=False))

    # Predict for all sampled items at once
    test_subset = test_ts.loc[sample_ids]
    try:
        pred_ts = predictor.predict(test_subset)
    except Exception as e:
        logger.warning(f"Could not generate predictions for plots: {e}")
        return {}

    figures = {}
    for item_id in sample_ids:
        item_test = test_ts.loc[item_id]
        item_pred = pred_ts.loc[item_id] if item_id in pred_ts.item_ids else None
        if item_pred is None or len(item_pred) == 0:
            continue

        # Align: use last prediction_length steps as ground truth
        gt = item_test["target"].iloc[-prediction_length:]

        # X-axis: days before departure if available, else dates (timestamp index)
        if "days_before_departure" in item_test.columns:
            x_values = item_test["days_before_departure"].iloc[-prediction_length:].values
            x_label = "Days before departure"
        else:
            x_values = gt.index
            x_label = "Date"

        # Predictions may be mean or multi-column; use mean if available
        if hasattr(item_pred, "columns") and "mean" in item_pred.columns:
            pred_values = item_pred["mean"].values
        elif hasattr(item_pred, "columns") and "target" in item_pred.columns:
            pred_values = item_pred["target"].values
        elif hasattr(item_pred, "columns"):
            pred_values = item_pred.iloc[:, 0].values
        else:
            pred_values = np.asarray(item_pred).ravel()
        pred_values = np.asarray(pred_values).ravel()
        gt_values = np.asarray(gt.values).ravel()

        if len(pred_values) != len(gt_values):
            pred_values = pred_values[: len(gt_values)]
            gt_values = gt_values[: len(pred_values)]

        # Per-sample metrics
        mask = gt_values != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.nanmean(np.abs((gt_values - pred_values) / np.where(mask, gt_values, np.nan))) * 100 if np.any(mask) else float("nan")
        rmse = np.sqrt(np.mean((gt_values - pred_values) ** 2))
        mae = np.mean(np.abs(gt_values - pred_values))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_values, gt_values, color="green", marker="o", label="Ground truth", markersize=4)
        ax.plot(x_values, pred_values, color="red", marker="s", label="Prediction", markersize=4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Target")
        ax.set_title(f"Prediction vs ground truth — {item_id}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        metric_values = {"MAPE": f"{mape:.2f}%", "RMSE": f"{rmse:.2f}", "MAE": f"{mae:.2f}"}
        metrics_text = "\n".join(
            f"{m}: {metric_values[m]}" for m in plot_metrics_list if m in metric_values
        )
        if not metrics_text:
            metrics_text = f"MAPE: {mape:.2f}%\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}"
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        plt.tight_layout()
        figures[str(item_id)] = fig

    if params.get("log_to_mlflow", True) and figures and mlflow.active_run() is not None:
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"performance_plots/{name}.png")
        mlflow.end_run()

    logger.info(f"Created {len(figures)} performance plots.")
    return figures


def _log_autogluon_to_mlflow(
    params: Dict[str, Any],
    predictor: TimeSeriesPredictor,
    train_ts: TimeSeriesDataFrame,
    test_ts: TimeSeriesDataFrame,
    test_metrics: Dict[str, float],
    leaderboard: pd.DataFrame | None,
    performance_plots: Dict[str, plt.Figure] | None = None,
) -> None:
    """Log AutoGluon run to MLflow: params, metrics, leaderboard, model path. Optional performance_plots are logged when provided; otherwise the create_performance_plots node logs to this run later."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("visionary_autogluon_timeseries")

    mlflow.start_run(description=params.get("run_description", ""))
    mlflow.log_params(
        {
            "forecast_horizon": params.get("forecast_horizon", 7),
            "eval_metric": params.get("eval_metric", "MAPE"),
            "presets": params.get("presets", "medium_quality"),
            "time_limit": params.get("time_limit", 3600),
            "train_items": train_ts.num_items,
            "train_timesteps": len(train_ts),
            "test_items": test_ts.num_items,
            "test_timesteps": len(test_ts),
            "best_model": getattr(predictor, "model_best", None),
        }
    )
    if test_metrics is not None:
        if isinstance(test_metrics, dict):
            mlflow.log_metrics(test_metrics)
        else:
            mlflow.log_metric(params.get("eval_metric", "score"), float(test_metrics))
    if leaderboard is not None and not leaderboard.empty:
        mlflow.log_table(data=leaderboard.reset_index(), artifact_file="leaderboard.json")
    mlflow.log_param("predictor_path", str(predictor.path))
    if performance_plots:
        for name, fig in performance_plots.items():
            mlflow.log_figure(fig, f"performance_plots/{name}.png")
            plt.close(fig)
    logger.info(f"MLflow run logged: {mlflow.active_run().info.run_id}")
    # Run left open so create_performance_plots node can log artifacts and end_run()
