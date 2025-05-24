from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_xgb_model, evaluate_cb_point_model

def create_pipeline() -> Pipeline:
    return pipeline([
        node(
            func=evaluate_xgb_model,
            inputs=["xgb_model", "xgb_X_test", "xgb_y_test"],
            outputs="metrics_xgb",
            name="evaluate_xgb_node",
        ),
        node(
            func=evaluate_cb_point_model,
            inputs=["cb_model", "cb_X_test", "cb_y_test"],
            outputs="metrics_cb",
            name="evaluate_cb_point_node",
        )
    ])
