from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_for_catboost,
    split_catboost_data,
    train_catboost_model,
    train_cb_quantile_model,
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
            inputs={
                "X_train": "cb_X_train",
                "y_train": "cb_y_train",
                "X_valid": "cb_X_test",
                "y_valid": "cb_y_test",
                "catboost_params": "params:cb_params",
                "cat_features": "cb_cat_features",
            },
            outputs="cb_model",
            name="train_cb_point_node",
        ),
        node(
            func=train_cb_quantile_model,
            inputs={
                "X_train": "cb_X_train",
                "y_train": "cb_y_train",
                "X_valid": "cb_X_test",
                "y_valid": "cb_y_test",
                "catboost_params": "params:cb_quantile_params",
                "cat_features": "cb_cat_features",
                "alpha": "params:cb_alpha_low",
            },
            outputs="cb_model_low",
            name="train_cb_quantile_low_node",
        ),
        node(
            func=train_cb_quantile_model,
            inputs={
                "X_train": "cb_X_train",
                "y_train": "cb_y_train",
                "X_valid": "cb_X_test",
                "y_valid": "cb_y_test",
                "catboost_params": "params:cb_quantile_params",
                "cat_features": "cb_cat_features",
                "alpha": "params:cb_alpha_median",
            },
            outputs="cb_model_median",
            name="train_cb_quantile_median_node",
        ),
        node(
            func=train_cb_quantile_model,
            inputs={
                "X_train": "cb_X_train",
                "y_train": "cb_y_train",
                "X_valid": "cb_X_test",
                "y_valid": "cb_y_test",
                "catboost_params": "params:cb_quantile_params",
                "cat_features": "cb_cat_features",
                "alpha": "params:cb_alpha_high",
            },
            outputs="cb_model_high",
            name="train_cb_quantile_high_node",
        ),
    ])
