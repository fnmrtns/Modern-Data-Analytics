# src/ml_regression_project/pipelines/regression_pipeline/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import report_pca_variance,evaluate_model_r2,fit_transform_features,enrich_project_data,remove_outliers_isolation_forest,split_data, apply_pca,train_model, evaluate_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=enrich_project_data,
            inputs=["df", "organization"],
            outputs="enriched_data",
            name="enrich_project_data_node"
        ),
        node(
            func=remove_outliers_isolation_forest,
            inputs=dict(data="enriched_data", contamination="params:contamination"),
            outputs="cleaned_data",
            name="remove_outliers_node"
        ),
        node(
            func=split_data,
            inputs=dict(data="cleaned_data", test_size="params:test_size"),
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),
        node(
            func=fit_transform_features,
            inputs=dict(
                X_train="X_train",
                X_test="X_test",
                columns_to_scale="params:columns_to_scale",
                columns_to_encode="params:columns_to_encode"
            ),
            outputs=["X_train_trans", "X_test_trans", "transformer_model"],
            name="fit_transform_features_node"
        ),
        node(
            func=apply_pca,
            inputs=dict(X_train="X_train_trans", X_test="X_test_trans", n_components="params:n_components"),
            outputs=["X_train_pca", "X_test_pca", "pca_model"],
            name="apply_pca_node"
        ),
    node(
        func=report_pca_variance,
        inputs="pca_model",
        outputs="explained_pca_variance",
        name="report_pca_variance_node"
        ),        
        node(
            func=train_model,
            inputs=["X_train_pca", "y_train"],
            outputs="model",
            name="train_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["model", "X_test_pca", "y_test"],
            outputs="mse",
            name="evaluate_model_node"
        ),
    node(
        func=evaluate_model_r2,
        inputs=["model", "X_test_pca", "y_test"],
        outputs="r2_score",
        name="evaluate_model_r2_node"
    ),        
    ])
