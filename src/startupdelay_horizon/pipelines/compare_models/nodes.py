import pandas as pd

def compare_model_metrics(metrics_xgb, metrics_cb):
    df_xgb = metrics_xgb.copy()
    df_cb = metrics_cb.copy()

    df_xgb["model"] = "xgboost"
    df_cb["model"] = "catboost_point"
    

    result = pd.concat([df_xgb, df_cb], ignore_index=True)
    return result  # guaranteed to be a single DataFrame
