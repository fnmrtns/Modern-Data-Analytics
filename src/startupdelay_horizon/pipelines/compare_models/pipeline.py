from kedro.pipeline import Pipeline, node, pipeline
from .nodes import compare_model_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compare_model_metrics,
            inputs=["metrics_xgb", "metrics_cb"],
            outputs="model_comparison_table",
            name="compare_model_metrics_node"
        )
    ])
