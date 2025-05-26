"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline
from ml_regression_project1.pipelines import regression_pipeline

pipelines_package = "ml_regression_project1.pipelines"

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "__default__": regression_pipeline.create_pipeline()
    }