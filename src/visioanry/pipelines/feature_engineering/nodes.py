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


def feature_engineering(aggregated_sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the aggregated sales data.

    Args:
        aggregated_sales_data (pd.DataFrame): Aggregated sales data.
    """
    #format the date column to be a date
    aggregated_sales_data['Date'] = pd.to_datetime(aggregated_sales_data['Date'])
    #add a column for the day of the week
    aggregated_sales_data['DayOfWeek'] = aggregated_sales_data['Date'].dt.dayofweek
    #add a column for the month
    aggregated_sales_data['Month'] = aggregated_sales_data['Date'].dt.month
    #add a column for the year
    aggregated_sales_data['Year'] = aggregated_sales_data['Date'].dt.year
    #add a column for the day of the month
    aggregated_sales_data['DayOfMonth'] = aggregated_sales_data['Date'].dt.day
    #add a column for the day of the year
    aggregated_sales_data['DayOfYear'] = aggregated_sales_data['Date'].dt.dayofyear
    #one hot encode StoreType
    aggregated_sales_data = pd.get_dummies(aggregated_sales_data, columns=['StoreType'])
    #one hot encode Assortment
    aggregated_sales_data = pd.get_dummies(aggregated_sales_data, columns=['Assortment'])
    #one hot encode StateHoliday
    aggregated_sales_data = pd.get_dummies(aggregated_sales_data, columns=['StateHoliday'])

    import pdb; pdb.set_trace()
    return aggregated_sales_data