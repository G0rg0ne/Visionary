"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    train_model, 
    evaluate_model
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            train_model,
            inputs=["df_prophet", "store_holidays"],
            name="train_model",
            outputs="prophet_model",
        ),
        Node(
            evaluate_model,
            inputs=["prophet_model", "future"],
            name="evaluate_model",
            outputs=None,
        )
    ])
