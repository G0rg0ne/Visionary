"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    train_model, 
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            train_model,
            inputs=[
                "tickets_train_data",
                "tickets_test_data",
                "params:model_training_parameters",
            ],
            name="train_model",
            outputs= None,
        ),
    ])
