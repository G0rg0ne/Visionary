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
from typing import Dict, Tuple, Any
import mlflow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def train_model(df_prophet: pd.DataFrame, store_holidays: pd.DataFrame, 
                mlflow_experiment_name: str, mlflow_run_name: str):
    """
    Train a Prophet model.
    Assumes data is clean and ready for training.
    
    Args:
        df_prophet: Training data in Prophet format (ds, y, and regressors)
        store_holidays: Holiday dataframe
        mlflow_experiment_name: Name of the MLflow experiment
        mlflow_run_name: Name of the MLflow run
    
    Returns:
        Trained Prophet model
    """
    logger.info("=" * 50)
    logger.info("TRAINING PROPHET MODEL")
    logger.info("=" * 50)
    
    # Set MLflow experiment and run
    mlflow.set_experiment(mlflow_experiment_name)
    mlflow.start_run(run_name=mlflow_run_name)
    logger.info(f"MLflow experiment: {mlflow_experiment_name}")
    logger.info(f"MLflow run: {mlflow_run_name}")
    
    # Initialize and configure model
    m = Prophet(
        holidays=store_holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    
    # Add regressors if they exist in the data
    regressors = []
    if 'Promo' in df_prophet.columns:
        m.add_regressor('Promo')
        regressors.append('Promo')
        logger.info("Added regressor: Promo")
    
    if 'SchoolHoliday' in df_prophet.columns:
        m.add_regressor('SchoolHoliday')
        regressors.append('SchoolHoliday')
        logger.info("Added regressor: SchoolHoliday")
    
    # Log model parameters to MLflow
    mlflow.log_params({
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'interval_width': 0.95,
        'regressors': ','.join(regressors) if regressors else 'none',
        'has_holidays': len(store_holidays) > 0 if store_holidays is not None else False
    })
    
    # Train the model
    logger.info("Fitting model...")
    m.fit(df_prophet)
    logger.info("✓ Model training completed successfully")
    
    # Log model to MLflow
    mlflow.prophet.log_model(m, "prophet_model")
    
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
        yearly_mean = forecast['yearly'].mean()
        logger.info(f"\nYearly Seasonality:")
        logger.info(f"  Range: {yearly_range:.2f}")
        logger.info(f"  Mean: {yearly_mean:.2f}")
    
    if 'weekly' in forecast.columns:
        weekly_range = forecast['weekly'].max() - forecast['weekly'].min()
        weekly_mean = forecast['weekly'].mean()
        logger.info(f"\nWeekly Seasonality:")
        logger.info(f"  Range: {weekly_range:.2f}")
        logger.info(f"  Mean: {weekly_mean:.2f}")
    
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


def plot_forecast_results(future: pd.DataFrame, forecast: pd.DataFrame, 
                          metrics: Dict[str, float]):
    """
    Create a visualization of forecasted vs actual values.
    Saves the figure as an HTML file and logs it to MLflow as an artifact.
    
    Args:
        future: Test data with 'ds' and 'Sales' columns
        forecast: Prophet forecast dataframe with predictions
        metrics: Dictionary of evaluation metrics (not displayed in plot)
    """
    # Prepare data
    dates = pd.to_datetime(future['ds'])
    y_true = future['Sales']
    y_pred = forecast['yhat']
    y_lower = forecast['yhat_lower']
    y_upper = forecast['yhat_upper']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Forecast vs Actual Sales', 'Residuals'),
        vertical_spacing=0.2
    )
    
    # Main forecast plot
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_true,
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_pred,
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_upper,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_lower,
            mode='lines',
            name='Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(width=0),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Residuals plot
    residuals = y_true - y_pred
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color='green', width=1),
            marker=dict(size=3)
        ),
        row=2, col=1
    )
    
    # Add zero line for residuals
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout with better spacing
    fig.update_layout(
        title=dict(
            text='Forecast Results',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=800,
        showlegend=True,
        hovermode='x unified',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Sales", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    
    # Save figure as HTML and log directly to MLflow
    html_content = fig.to_html(full_html=True)
    
    # Write to a file with the desired artifact name
    forecast_file = "forecast_graph.html"
    with open(forecast_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Log HTML to MLflow - file will be logged with its filename at root level
    mlflow.log_artifact(forecast_file)
    logger.info("Forecast graph logged to MLflow as artifact")
    
    # Clean up file
    os.remove(forecast_file)


def evaluate_model(model: Prophet, future: pd.DataFrame):
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
    # Prepare metrics for MLflow (only non-NaN values)
    mlflow_metrics = {}
    for metric_name, metric_value in metrics.items():
        if not np.isnan(metric_value):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
            mlflow_metrics[metric_name] = float(metric_value)
        else:
            logger.info(f"  {metric_name}: N/A")
    
    # Log metrics to MLflow
    mlflow.log_metrics(mlflow_metrics)
    
    # Create forecast visualization and log to MLflow
    logger.info("Creating forecast visualization...")
    plot_forecast_results(future, forecast, metrics)
    logger.info("✓ Forecast graph created and logged to MLflow")
    
    # End MLflow run
    mlflow.end_run()
    logger.info("MLflow run completed")