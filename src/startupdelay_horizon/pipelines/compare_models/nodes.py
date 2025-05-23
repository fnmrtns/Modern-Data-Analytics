import pandas as pd

def compare_model_metrics(metrics_xgb, metrics_cb_point, metrics_cb_quantile):
    df_xgb = metrics_xgb.copy()
    df_cb_point = metrics_cb_point.copy()
    df_cb_quantile = metrics_cb_quantile.copy()

    df_xgb["model"] = "xgboost"
    df_cb_point["model"] = "catboost_point"
    df_cb_quantile["model"] = "catboost_quantile"

    result = pd.concat([df_xgb, df_cb_point, df_cb_quantile], ignore_index=True)
    return result  # guaranteed to be a single DataFrame
