"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    feature_engineering,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                feature_engineering,
                inputs=["merged_data", "params:airport_country_mapping"],
                outputs="merged_data_with_holidays",
                name="feature_engineering",
                ),
        ]
    )
