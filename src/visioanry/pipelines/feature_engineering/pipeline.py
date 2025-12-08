"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    aggegate_sales_data,
    feature_engineering,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                aggegate_sales_data, 
                inputs=["sales_train_data", "stores_info"], 
                outputs="aggregated_sales_data", 
                name="aggegate_sales_data",
                ),
            Node(
                feature_engineering,
                inputs="aggregated_sales_data",
                outputs="feature_engineered_sales_data",
                name="feature_engineering",
                ),
        ]
    )
