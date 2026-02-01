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
    build_delta_targets,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                feature_engineering,
                inputs=["sampled_dataset", "params:airport_country_mapping", "params:n_holiday_days"],
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
                inputs=["features_with_target_vector", "params:horizon"],
                outputs="cleaned_horizon_features_with_target_vector",
                name="clean_target_vector",
            ),
            Node(
                build_delta_targets,
                inputs=["cleaned_horizon_features_with_target_vector", "params:horizon"],
                outputs="features_with_delta_targets",
                name="build_delta_targets",
            ),
            Node(
                split_data,
                inputs="features_with_delta_targets",
                outputs=["tickets_train_data", "tickets_test_data"],
                name="split_dataset",
            ),
        ]
    )
