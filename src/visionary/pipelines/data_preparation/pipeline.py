"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    merge_dataframes,
    filter_flight,
    prepare_timeseries_data,
    split_timeseries_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            merge_dataframes,
            inputs="raw_csv_data",
            outputs="merged_data",
            name="merge_dataframes",
        ),
        Node(
            filter_flight,
            inputs=["merged_data", "params:num_samples"],
            outputs="sampled_dataset",
            name="filter_flights",
        ),
        Node(
            prepare_timeseries_data,
            inputs=["sampled_dataset", "params:timeseries_preparation"],
            outputs="timeseries_prepared",
            name="prepare_timeseries",
        ),
        Node(
            split_timeseries_data,
            inputs=["timeseries_prepared", "params:train_test_split"],
            outputs=["timeseries_train", "timeseries_test"],
            name="split_timeseries",
        ),
    ])
