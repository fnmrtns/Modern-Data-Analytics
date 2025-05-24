import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from startupdelay_horizon.metrics import regression_metrics
import mlflow

def evaluate_xgb_model(
    xgb_model: XGBRegressor,
    xgb_X_test: pd.DataFrame,
    xgb_y_test: pd.Series
) -> pd.DataFrame:
    y_pred = xgb_model.predict(xgb_X_test)
    
    # Compute metrics (assuming this is a custom util that returns a dict)
    metrics = regression_metrics(xgb_y_test, y_pred)

    # Log each metric to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(f"xgb_{key}", value)

    return pd.DataFrame([metrics])


def evaluate_cb_point_model(
    cb_model: CatBoostRegressor,
    cb_X_test: pd.DataFrame,
    cb_y_test: pd.Series
) -> pd.DataFrame:
    y_pred = cb_model.predict(cb_X_test)

    # Compute regression metrics using your custom utility
    metrics = regression_metrics(cb_y_test, y_pred)

    # Log each metric to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(f"cb_{key}", value)

    return pd.DataFrame([metrics])




