"""
Model training nodes for the Visionary pipeline.

Trains AutoGluon time series forecasting models to predict flight prices
up to a configurable horizon. Uses temporal train/test split on query_date
to avoid data leakage.
"""
import os
from pathlib import Path
from typing import Any, Dict

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
    fitted predictor. Uses temporal split: train = past, test = future.

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

    min_length_for_eval = prediction_length + 1
    steps_per_item = test_ts.num_timesteps_per_item()
    valid_item_ids = steps_per_item[steps_per_item >= min_length_for_eval].index
    test_ts_eval = test_ts.loc[valid_item_ids] if len(valid_item_ids) > 0 else None

    test_metrics = None
    leaderboard = None
    if test_ts_eval is not None and len(test_ts_eval) > 0:
        logger.info(
            f"Evaluating on test set (temporal holdout), {test_ts_eval.num_items} series with >= {min_length_for_eval} steps"
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


def _log_autogluon_to_mlflow(
    params: Dict[str, Any],
    predictor: TimeSeriesPredictor,
    train_ts: TimeSeriesDataFrame,
    test_ts: TimeSeriesDataFrame,
    test_metrics: Dict[str, float],
    leaderboard: pd.DataFrame | None,
) -> None:
    """Log AutoGluon run to MLflow: params, metrics, leaderboard, and model path."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("visionary_autogluon_timeseries")

    with mlflow.start_run(description=params.get("run_description", "")):
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
        logger.info(f"MLflow run logged: {mlflow.active_run().info.run_id}")
