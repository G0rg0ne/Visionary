"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    feature_engineering,
    split_data,
    data_augmentation,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                feature_engineering,
                inputs=["sampled_dataset", "params:airport_country_mapping"],
                outputs="merged_data_with_holidays",
                name="feature_engineering",
            ),
            Node(
                data_augmentation,
                inputs=["merged_atda_with_holidays", "airport_data"],
                outputs="full_feature_data",
                name="data_augmentation",
            ),
            Node(
                split_data,
                inputs="full_feature_data",
                outputs=["tickets_train_data", "tickets_test_data"],
                name="split_dataset",
            ),
        ]
    )
