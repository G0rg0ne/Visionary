"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""
import pandas as pd
import plotly.express as px

def plot_sales_data(sales_data: pd.DataFrame) -> str:
    """
    Plot the sales data from both train and test sets and return the HTML representation of the graph.

    Args:
        sales_train_data (pd.DataFrame): Training sales data.
        sales_test_data (pd.DataFrame): Test sales data.

    Columns expected:
        - Store
        - DayOfWeek
        - Date
        - Sales
        - Customers
        - Open
        - Promo
        - StateHoliday
        - SchoolHoliday
    """
    #format the date column to be a date
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    #format the store column to be a string
    sales_data['Store'] = sales_data['Store'].astype(str)
    fig_train = px.line(
        sales_data,
        x='Date',
        y='Sales',
        color='Store',
        title='Train Sales Over Time by Store',
    )
    return fig_train.to_html(full_html=False)