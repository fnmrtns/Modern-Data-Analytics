from kedro.pipeline import Pipeline, node, pipeline
import mlflow
from .nodes import (
    preprocess_for_catboost,
    split_catboost_data,
    train_catboost_model,
)

def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=preprocess_for_catboost,
            inputs="model_input_table",
            outputs="cb_input_table",
            name="cb_preprocessing_node",
        ),
        node(
            func=split_catboost_data,
            inputs="cb_input_table",
            outputs=[
                "cb_X_train",
                "cb_X_test",
                "cb_y_train",
                "cb_y_test",
                "cb_cat_features",
            ],
            name="cb_split_data_node",
        ),
        node(
            func=train_catboost_model,
            inputs=["cb_X_train", "cb_y_train", "cb_X_test", "cb_y_test", "params:cb_params", "cb_cat_features"],
            outputs=["cb_model", "cb_model_mlflow"],
            name="cb_training_node"
        )
    ])
