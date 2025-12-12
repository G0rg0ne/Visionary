"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""
import pandas as pd

def aggegate_sales_data(sales_data: pd.DataFrame, stores_info: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the sales data by date and store.

    Args:
        sales_data (pd.DataFrame): Training sales data.
        stores_info (pd.DataFrame): Stores information.
    """
    #format the date column to be a date
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    stores_info['Store'] = stores_info['Store'].astype(str)
    sales_data['Store'] = sales_data['Store'].astype(str)
    aggregated_sales_data = sales_data.merge(stores_info, on='Store', how='left')
    return aggregated_sales_data

def extract_holidays(sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract holidays from the sales data, including both state and school holidays.

    Args:
        sales_data (pd.DataFrame): Sales data.
    """
    holiday_data = sales_data[['Date', 'StateHoliday','SchoolHoliday']].copy()
    holiday_data = holiday_data.drop_duplicates()
    holiday_data['StateHoliday'] = holiday_data['StateHoliday'].astype(str)
    holiday_data['SchoolHoliday'] = holiday_data['SchoolHoliday'].astype(str)

    # Select rows where there is either a StateHoliday or a SchoolHoliday
    #holiday_data = holiday_data[(holiday_data['StateHoliday'] != '0') | (holiday_data['SchoolHoliday'] != '0')]
    holiday_data = holiday_data[(holiday_data['StateHoliday'] != '0')]

    holiday_map = {
        'a': 'PublicHoliday',
        'b': 'Easter',
        'c': 'Christmas'
    }

    # For state holidays, assign mapped name if present
    holiday_data['holiday'] = holiday_data['StateHoliday'].map(holiday_map)

    # Where there is a school holiday, add/overwrite to 'SchoolHoliday'
    #holiday_data.loc[holiday_data['SchoolHoliday'] != '0', 'holiday'] = 'SchoolHoliday'
    prophet_holidays = holiday_data[['Date', 'holiday']].rename(columns={'Date': 'ds'})
    return prophet_holidays

def split_data_dates(sales_data: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """
    Split the sales data into training and testing sets.
    """
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    sales_data = sales_data.sort_values(by='Date')
    split_date = pd.to_datetime(split_date)

    sales_data_training = sales_data[sales_data['Date'] < split_date]
    sales_data_testing = sales_data[sales_data['Date'] >= split_date]

    return sales_data_training, sales_data_testing


def feature_engineering(sales_data: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """
    Perform feature engineering on the aggregated sales data.

    Args:
        aggregated_sales_data (pd.DataFrame): Aggregated sales data.
    """
    
    store_holidays = extract_holidays(sales_data)

    #filter data for the store id to train
    sales_data = sales_data[sales_data['Open'] == 1].copy()
    sales_data = sales_data.drop(columns=['Customers'])


    sales_data_training, sales_data_testing = split_data_dates(sales_data, split_date)

    store_train = sales_data_training.copy()
    store_evaluation = sales_data_testing.copy()

    df_train_prophet = store_train.rename(columns={'Date': 'ds', 'Sales': 'y'})
    future_evaluation = store_evaluation.rename(columns={'Date': 'ds'})

    return df_train_prophet, future_evaluation, store_holidays