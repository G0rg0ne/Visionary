"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    plot_sales_data,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(plot_sales_data,
        inputs="sales_train_data",
        outputs="sales_graph_html",
        name="plot_sales_data"),
    ])
