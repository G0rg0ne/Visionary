"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

def train_model(df_prophet: pd.DataFrame, store_holidays: pd.DataFrame):
    """
    Train a Prophet model.
    Assumes data is clean and ready for training.
    
    Args:
        df_prophet: Training data in Prophet format (ds, y, and regressors)
        store_holidays: Holiday dataframe
    
    Returns:
        Trained Prophet model
    """
    logger.info("=" * 50)
    logger.info("TRAINING PROPHET MODEL")
    logger.info("=" * 50)
    
    # Initialize and configure model
    m = Prophet(
        holidays=store_holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    
    # Add regressors if they exist in the data
    if 'Promo' in df_prophet.columns:
        m.add_regressor('Promo')
        logger.info("Added regressor: Promo")
    
    if 'SchoolHoliday' in df_prophet.columns:
        m.add_regressor('SchoolHoliday')
        logger.info("Added regressor: SchoolHoliday")
    
    # Train the model
    logger.info("Fitting model...")
    m.fit(df_prophet)
    logger.info("✓ Model training completed successfully")
    
    return m


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


def analyze_model_components(model: Prophet, df_prophet: pd.DataFrame):
    """
    Analyze and log model components (trend, seasonality, regressors).
    
    Args:
        model: Trained Prophet model
        df_prophet: Training data
    """
    logger.info("=" * 50)
    logger.info("MODEL COMPONENT ANALYSIS")
    logger.info("=" * 50)
    
    # Get forecast to analyze components
    forecast = model.predict(df_prophet.drop(columns=['y']))
    
    # Analyze trend
    trend_change = forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]
    trend_pct_change = (trend_change / forecast['trend'].iloc[0]) * 100 if forecast['trend'].iloc[0] != 0 else 0
    
    logger.info("Trend Analysis:")
    logger.info(f"  Initial trend: {forecast['trend'].iloc[0]:.2f}")
    logger.info(f"  Final trend: {forecast['trend'].iloc[-1]:.2f}")
    logger.info(f"  Change: {trend_change:.2f} ({trend_pct_change:.2f}%)")
    
    # Analyze seasonality components
    if 'yearly' in forecast.columns:
        yearly_range = forecast['yearly'].max() - forecast['yearly'].min()
        logger.info(f"\nYearly Seasonality:")
        logger.info(f"  Range: {yearly_range:.2f}")
        logger.info(f"  Mean: {forecast['yearly'].mean():.2f}")
    
    if 'weekly' in forecast.columns:
        weekly_range = forecast['weekly'].max() - forecast['weekly'].min()
        logger.info(f"\nWeekly Seasonality:")
        logger.info(f"  Range: {weekly_range:.2f}")
        logger.info(f"  Mean: {forecast['weekly'].mean():.2f}")
    
    # Analyze regressors
    regressor_cols = [col for col in forecast.columns if col not in 
                     ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 
                      'yearly', 'weekly', 'daily', 'holidays', 'additive_terms', 
                      'multiplicative_terms', 'additive_terms_lower', 
                      'additive_terms_upper', 'multiplicative_terms_lower', 
                      'multiplicative_terms_upper']]
    
    if regressor_cols:
        logger.info(f"\nRegressor Analysis:")
        for reg in regressor_cols:
            reg_range = forecast[reg].max() - forecast[reg].min()
            reg_mean = forecast[reg].mean()
            logger.info(f"  {reg}:")
            logger.info(f"    Range: {reg_range:.2f}")
            logger.info(f"    Mean: {reg_mean:.2f}")


def evaluate_model(model: Prophet, future: pd.DataFrame, 
                  ) -> Dict[str, float]:
    """
    Evaluate the model on test/validation data with comprehensive metrics.
    Assumes test data is clean and ready for evaluation.
    
    Args:
        model: Trained Prophet model
        future: Test data with 'ds' and 'Sales' columns
        df_prophet: Training data (optional, for comparison)
    
    Returns:
        Dictionary of evaluation metrics
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
        if not np.isnan(metric_value):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: N/A")
    
    return metrics