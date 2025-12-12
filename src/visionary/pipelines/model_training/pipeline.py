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
                "df_train_prophet",
                "store_holidays",
                "future_evaluation",
                "params:mlflow_experiment_name",
                "params:mlflow_run_name",
                "params:model_training_parameters",
            ],
            name="train_model",
            outputs="prophet_models",
        ),
    ])
