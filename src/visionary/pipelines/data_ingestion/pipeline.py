"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import (
    load_csv_files_from_minio_combined,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            load_csv_files_from_minio_combined,
            inputs=["params:minio_credentials"],
            outputs=None,
            name="load_csv_files_from_minio"
        ),
    ])
