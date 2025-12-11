"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    train_model, 
    evaluate_model,
    analyze_model_components,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            train_model,
            inputs=["df_prophet", "store_holidays", "params:mlflow_experiment_name", "params:mlflow_run_name"],
            name="train_model",
            outputs="prophet_model",
        ),
        Node(
            analyze_model_components,
            inputs=["prophet_model", "df_prophet"],
            name="analyze_model_components",
            outputs=None,
        ),
        Node(
            evaluate_model,
            inputs=["prophet_model", "future"],
            name="evaluate_model",
            outputs=None,
        )
    ])
