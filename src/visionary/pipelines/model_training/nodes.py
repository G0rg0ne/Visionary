"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

import pandas as pd
from prophet import Prophet
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(df_prophet: pd.DataFrame, store_holidays: pd.DataFrame):
    """
    Train a Prophet model.
    """
    m = Prophet(
        holidays=store_holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    m.add_regressor('Promo')
    m.add_regressor('SchoolHoliday')

    m.fit(df_prophet)
    
    return m

def evaluate_model(model: Prophet, future: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the model.
    also log the metrics for the model
    """
    #future alread contains the Sales column so keep it somewhere so you can use it for evaluation
    y_true = future['Sales']
    forecast = model.predict(future.drop(columns=['Sales']))
    y_pred = forecast['yhat']
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
    return y_true, y_pred, mse, mae, r2