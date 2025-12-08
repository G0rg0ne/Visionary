"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""
import pandas as pd
import plotly.express as px

def plot_sales_data(sales_data: pd.DataFrame) -> str:
    """
    Plot the sales data from the training set and return the HTML representation of the graph.

    Args:
        sales_data (pd.DataFrame): Training sales data.
    """
    #format the date column to be a date
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    #format the store column to be a string
    sales_data['Store'] = sales_data['Store'].astype(str)
    daily_sales = sales_data.groupby('Date')['Sales'].sum().reset_index()
    fig1 = px.line(
        daily_sales,
        x='Date',
        y='Sales',
        title='Total Daily Sales Over Time'
    )
    
    return fig1.to_html(full_html=False)

def plot_top_10_stores_by_total_sales(sales_data: pd.DataFrame) -> str:
    """
    Plot the top 10 stores by total sales and return the HTML representation of the graph.

    Args:
        sales_data (pd.DataFrame): Training sales data.
    """
    store_totals = sales_data.groupby('Store')['Sales'].sum().nlargest(10)
    fig_top_10_stores = px.bar(
        x=store_totals.index.astype(str),
        y=store_totals.values,
        title='Top 10 Stores by Total Sales',
        labels={'x': 'Store', 'y': 'Total Sales'}
    )
    return fig_top_10_stores.to_html(full_html=False)

def correlation_matrix(sales_data: pd.DataFrame) -> str:
    """
    Plot the correlation matrix of the sales data and return the HTML representation of the graph.

    Args:
        sales_data (pd.DataFrame): Training sales data.
    """
    correlation_columns = ['Sales', 'Customers', 'Open', 'Promo', 'SchoolHoliday']
    correlation_data = sales_data[correlation_columns]
    correlation_matrix = correlation_data.corr()
    fig_correlation_matrix = px.imshow(correlation_matrix, title='Correlation Matrix')
    return fig_correlation_matrix.to_html(full_html=False)
