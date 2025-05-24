import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from startupdelay_horizon.metrics import regression_metrics


def evaluate_xgb_model(xgb_model: XGBRegressor, xgb_X_test: pd.DataFrame, xgb_y_test: pd.Series) -> pd.DataFrame:
    y_pred = xgb_model.predict(xgb_X_test)
    metrics = regression_metrics(xgb_y_test, y_pred)
    return pd.DataFrame([metrics])


def evaluate_cb_point_model(cb_model: CatBoostRegressor, cb_X_test: pd.DataFrame, cb_y_test: pd.Series) -> pd.DataFrame:
    y_pred = cb_model.predict(cb_X_test)
    metrics = regression_metrics(cb_y_test, y_pred)
    return pd.DataFrame([metrics])



