"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    feature_engineering,
    split_data,
    data_augmentation,
    build_target_vector,
    clean_target_vector,
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
                inputs=["merged_data_with_holidays", "airport_data"],
                outputs="full_feature_data",
                name="data_augmentation",
            ),
            Node(
                build_target_vector,
                inputs=["full_feature_data", "params:horizon"],
                outputs="features_with_target_vector",
                name="build_target_vector",
            ),
            Node(
                clean_target_vector,
                inputs="features_with_target_vector",
                outputs="cleaned_horizon_features_with_target_vector",
                name="clean_target_vector",
            ),
            Node(
                split_data,
                inputs="cleaned_horizon_features_with_target_vector",
                outputs=["tickets_train_data", "tickets_test_data"],
                name="split_dataset",
            ),
        ]
    )
