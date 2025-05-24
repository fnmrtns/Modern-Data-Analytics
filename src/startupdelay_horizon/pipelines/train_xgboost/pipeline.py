from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_for_xgboost, split_xgboost_data, train_xgboost_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_for_xgboost,
            inputs="model_input_table",
            outputs="xgb_input_table",
            name="xgb_preprocessing_node"
        ),
        node(
            func=split_xgboost_data,
            inputs="xgb_input_table",
            outputs=["xgb_X_train", "xgb_X_test", "xgb_y_train", "xgb_y_test"],
            name="xgb_split_data_node"
        ),
        
        node(
            func=train_xgboost_model,
            inputs=["xgb_X_train", "xgb_y_train", "params:xgb_params"],
            outputs=["xgb_model", "xgb_model_mlflow"],
            name="xgb_training_node"
        )
    ])
