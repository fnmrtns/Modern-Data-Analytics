from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

def preprocess_for_catboost(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pillar"] = df["pillar"].astype(str)
    df["countryCoor"] = df["countryCoor"].astype(str)
    return df

def split_catboost_data(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]
    cat_features = ["pillar", "countryCoor"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wrap y Series as DataFrames
    return (
        X_train,
        X_test,
        y_train.to_frame(name="startupDelay"),
        y_test.to_frame(name="startupDelay"),
        cat_features,
    )

def train_catboost_model(X_train, y_train, X_valid, y_valid, catboost_params, cat_features):
    model = CatBoostRegressor(**catboost_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        verbose=0
    )
    return model

def train_cb_quantile_model(X_train, y_train, X_valid, y_valid, catboost_params, cat_features, alpha):
    params = catboost_params.copy()
    params["loss_function"] = f"Quantile:alpha={alpha}"
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        verbose=0
    )
    return model
