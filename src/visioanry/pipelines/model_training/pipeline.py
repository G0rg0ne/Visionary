"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            train_model,
            inputs=["df_prophet", "future", "store_holidays"],
            outputs="trained_prophet_model",
            name="train_model"
        )
    ])
