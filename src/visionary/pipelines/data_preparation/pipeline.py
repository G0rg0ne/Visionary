"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    merge_dataframes,
    filter_flight,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            merge_dataframes,
            inputs="raw_csv_data",
            outputs="merged_data",
            name="merge_dataframes"
        ),
        Node(
            filter_flight,
            inputs=["merged_data", "params:num_samples"],
            outputs="sampled_dataset",
            name="filter_flights"
        ),
    ])
