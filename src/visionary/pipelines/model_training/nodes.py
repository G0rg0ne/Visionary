"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Any
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import os

def train_model(df_train_prophet: pd.DataFrame, store_holidays: pd.DataFrame, 
                future_evaluation: pd.DataFrame, mlflow_experiment_name: str, mlflow_run_name: str, model_training_parameters: Dict[str, Any]):
    """
    Train Prophet models for each store.
    Assumes data is clean and ready for training.
    
    Args:
        df_train_prophet: Training data with 'Store', 'Date', 'Sales' columns and optional regressors
        store_holidays: Holiday dataframe in Prophet format (ds, holiday columns)
        mlflow_experiment_name: Name of the MLflow experiment
        mlflow_run_name: Name of the MLflow run
        model_training_parameters: Dictionary of Prophet model parameters
    
    Returns:
        List of trained Prophet models (one per store)
    """
    # Data validation
    if df_train_prophet.empty:
        raise ValueError("Training dataframe is empty")
    
    mlflow.set_experiment(mlflow_experiment_name)
    logger.info(f"MLflow experiment: {mlflow_experiment_name}")
    logger.info("=" * 50)
    logger.info("TRAINING PROPHET MODEL")
    logger.info("=" * 50)
    available_stores = df_train_prophet['Store'].unique()
    logger.info(f"Training models for {len(available_stores)} stores")
    prophet_models = []
    regressors = ['Promo','Open','SchoolHoliday']
    
    # Start parent run for overall training process
    with mlflow.start_run(run_name=mlflow_run_name):
        logger.info(f"MLflow parent run: {mlflow_run_name}")
        
        # Log parent-level parameters
        mlflow.log_params({
            'total_stores': len(available_stores),
            'yearly_seasonality': model_training_parameters['yearly_seasonality'],
            'weekly_seasonality': model_training_parameters['weekly_seasonality'],
            'daily_seasonality': model_training_parameters['daily_seasonality'],
            'interval_width': model_training_parameters['interval_width'],
            'has_holidays': True if store_holidays is not None and not store_holidays.empty else False,
            'regressor_list': ','.join(regressors)
        })
        
        # Train models for each store as nested (child) runs
        for store in available_stores:
            try:
                df_train_prophet_store = df_train_prophet[df_train_prophet['Store'] == store].copy()
                
                # Validate store-specific data
                if df_train_prophet_store.empty:
                    logger.warning(f"Store {store} has no data, skipping")
                    continue
                
                # Create nested (child) run for this store
                with mlflow.start_run(run_name=f"store_{store}", nested=True):
                    logger.info(f"MLflow nested run: store_{store}")
                    
                    prophet_model = Prophet(
                        holidays=store_holidays,
                        yearly_seasonality=model_training_parameters['yearly_seasonality'],
                        weekly_seasonality=model_training_parameters['weekly_seasonality'],
                        daily_seasonality=model_training_parameters['daily_seasonality'],
                        interval_width=model_training_parameters['interval_width']
                    )
                    
                    # Track which regressors were actually added
                    added_regressors = []
                    for regressor in regressors:
                        if regressor in df_train_prophet_store.columns:
                            prophet_model.add_regressor(regressor)
                            added_regressors.append(regressor)
                            logger.info(f"Added regressor: {regressor}")
                        else:
                            logger.info(f"Regressor {regressor} not found in data")
                    
                    prophet_model.fit(df_train_prophet_store)
                    prophet_models.append(prophet_model)
                    logger.info(f"Trained model for store: {store}")
                    #Evaluate model on Evalueation data set
                    metrics = evaluate_model(prophet_model, future_evaluation[future_evaluation['Store'] == store])
                    mlflow.log_metrics(metrics)
                    logger.info(f"Logged metrics for store: {store}")

                    mlflow.prophet.log_model(prophet_model, name=f"prophet_model_store_{store}")
                    
                    logger.info(f"Logged model for store: {store}")
            
            except Exception as e:
                logger.error(f"Error training model for store {store}: {str(e)}")
                # Nested run will be automatically closed by context manager
                # Continue with next store instead of raising
                continue
        
        # Log summary metrics to parent run
        if prophet_models:
            mlflow.log_metric('successful_models', len(prophet_models))
            mlflow.log_metric('failed_models', len(available_stores) - len(prophet_models))
            logger.info(f"Parent run completed: {len(prophet_models)}/{len(available_stores)} models trained successfully")
    
    if not prophet_models:
        raise ValueError("No models were successfully trained")
    
    logger.info(f"Successfully trained {len(prophet_models)} models")
    return prophet_models


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE) - useful for sales forecasting
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
        logger.warning("Cannot calculate MAPE: all true values are zero")
    
    # Mean Absolute Scaled Error (MASE) - scale-independent metric
    # Using naive forecast as baseline (shift by 1 period)
    if len(y_true) > 1:
        naive_forecast = y_true.shift(1).dropna()
        naive_actual = y_true.iloc[1:]
        if len(naive_forecast) > 0 and naive_actual.std() > 0:
            mase = mae / np.mean(np.abs(naive_actual - naive_forecast))
        else:
            mase = np.nan
    else:
        mase = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'MASE': mase
    }

def evaluate_model(model: Prophet, future: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the model on test/validation data with comprehensive metrics.
    Assumes test data is clean and ready for evaluation.
    
    Args:
        model: Trained Prophet model
        future: Test data with 'ds' and 'Sales' columns
    """
    logger.info("=" * 50)
    logger.info("EVALUATING ON TEST DATA")
    logger.info("=" * 50)
    
    # Make predictions
    y_true = future['Sales']
    forecast = model.predict(future.drop(columns=['Sales']))
    y_pred = forecast['yhat']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    logger.info("\nTest Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics