import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from startupdelay_horizon.metrics import regression_metrics, quantile_coverage


def evaluate_xgb_model(xgb_model: XGBRegressor, xgb_X_test: pd.DataFrame, xgb_y_test: pd.Series) -> pd.DataFrame:
    y_pred = xgb_model.predict(xgb_X_test)
    metrics = regression_metrics(xgb_y_test, y_pred)
    return pd.DataFrame([metrics])


def evaluate_cb_point_model(cb_model: CatBoostRegressor, cb_X_test: pd.DataFrame, cb_y_test: pd.Series) -> pd.DataFrame:
    y_pred = cb_model.predict(cb_X_test)
    metrics = regression_metrics(cb_y_test, y_pred)
    return pd.DataFrame([metrics])


def evaluate_cb_quantile_model(
    cb_model_low: CatBoostRegressor,
    cb_model_median: CatBoostRegressor,
    cb_model_high: CatBoostRegressor,
    cb_X_test: pd.DataFrame,
    cb_y_test: pd.Series
) -> pd.DataFrame:
    preds = np.vstack([
        cb_model_low.predict(cb_X_test),
        cb_model_median.predict(cb_X_test),
        cb_model_high.predict(cb_X_test),
    ]).T
    metrics = quantile_coverage(cb_y_test.values, preds)
    return pd.DataFrame([metrics])


