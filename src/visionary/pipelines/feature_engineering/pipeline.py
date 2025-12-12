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
            # Node(   
            #     aggegate_sales_data, 
            #     inputs=["sales_train_data", "stores_info"], 
            #     outputs="aggregated_sales_data", 
            #     name="aggegate_sales_data",
            #     ),
            Node(
                feature_engineering,
                inputs=["sales_train_data","params:split_date"],
                outputs=["df_train_prophet", "future_evaluation", "store_holidays"],
                name="feature_engineering",
                ),
        ]
    )
