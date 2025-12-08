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
    import pdb; pdb.set_trace()
    return aggregated_sales_data