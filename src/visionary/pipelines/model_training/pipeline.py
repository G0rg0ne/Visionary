"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import train_autogluon_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            train_autogluon_model,
            inputs=[
                "timeseries_train",
                "timeseries_test",
                "params:model_training_parameters",
            ],
            outputs="autogluon_predictor",
            name="train_autogluon",
        ),
    ])
