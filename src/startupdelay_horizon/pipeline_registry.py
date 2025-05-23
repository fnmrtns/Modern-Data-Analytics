
from kedro.pipeline import Pipeline

from startupdelay_horizon.pipelines.general_preprocessing import create_pipeline as create_general_preprocessing_pipeline
from startupdelay_horizon.pipelines.train_xgboost import create_pipeline as create_train_xgboost_pipeline
from startupdelay_horizon.pipelines.train_catboost import create_pipeline as create_train_catboost_pipeline
from startupdelay_horizon.pipelines.evaluate_model.pipeline import create_pipeline as create_evaluate_model_pipeline
from startupdelay_horizon.pipelines.compare_models import create_pipeline as create_compare_models_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "train_xgboost": create_train_xgboost_pipeline(),
        "train_catboost": create_train_catboost_pipeline(),
        "evaluate_model": create_evaluate_model_pipeline(),
        "compare_models": create_compare_models_pipeline(),
        "general_preprocessing": create_general_preprocessing_pipeline(),
        "__default__": (
            create_general_preprocessing_pipeline()
            + create_train_xgboost_pipeline()
            + create_train_catboost_pipeline()
            + create_evaluate_model_pipeline()
            + create_compare_models_pipeline()
        ),
    }

