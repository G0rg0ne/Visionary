"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    plot_sales_data,
    plot_top_10_stores_by_total_sales,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            plot_sales_data,
            inputs="sales_train_data",
            outputs="sales_graph_html",
            name="plot_sales_data"
            ),

        Node(
            plot_top_10_stores_by_total_sales,
            inputs="sales_train_data",
            outputs="top_10_stores_by_total_sales_graph_html",
            name="plot_top_10_stores_by_total_sales"),
    ])
