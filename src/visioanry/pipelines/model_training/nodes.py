"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

import pandas as pd
from prophet import Prophet

def train_model(df_prophet: pd.DataFrame, future: pd.DataFrame, store_holidays: pd.DataFrame):
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