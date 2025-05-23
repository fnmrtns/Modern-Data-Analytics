

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import preprocess

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess,
            inputs=["project_raw", "programme_raw", "organization_raw"],
            outputs="model_input_table",
            name="preprocess_node"
        )
    ])