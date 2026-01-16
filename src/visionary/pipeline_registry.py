"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines.data_ingestion import pipeline as data_ingestion_pipe
from .pipelines.feature_engineering import pipeline as feature_engineering_pipe
from .pipelines.model_training import pipeline as model_training_pipe
from .pipelines.data_preparation import pipeline as data_preparation_pipe
from .pipelines.data_viz import pipeline as data_viz_pipe

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    #create the data ingestion pipeline
    data_ingestion_pipeline = data_ingestion_pipe.create_pipeline()
    data_preparation_pipeline = data_preparation_pipe.create_pipeline()
    data_viz_pipeline = data_viz_pipe.create_pipeline()


    #create the ML pipeline
    #feature_engineering_pipeline = feature_engineering_pipe.create_pipeline()
    #model_training_pipeline = model_training_pipe.create_pipeline()
    #training_pipeline =    model_training_pipeline + feature_engineering_pipeline
    return {
        "__default__": sum(pipelines.values()),
        "ingestion_pipeline": data_ingestion_pipeline,
        "fusion_pipeline": data_preparation_pipeline,
        "viz_pipeline": data_viz_pipeline,
        #"training_pipeline": training_pipeline,
    }