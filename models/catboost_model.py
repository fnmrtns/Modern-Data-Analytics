import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from config.paths import data_path

# ------------------ Preprocessing ------------------ #

def preprocess_for_catboost(input_file="selected_features.json", output_file="selected_features_CB.json"):
    """
    Preprocess data for CatBoost:
    - Keep categorical features ('pillar', 'countryCoor') as strings
    - Keep numeric features as-is
    - Save as JSON to the Data/ folder
    """
    df = pd.read_json(data_path(input_file))

    # Ensure correct dtype
    df["pillar"] = df["pillar"].astype(str)
    df["countryCoor"] = df["countryCoor"].astype(str)

    output_path = data_path(output_file)
    df.to_json(output_path, orient="records", indent=2)

    print(f"CatBoost preprocessing complete. Saved to {output_path}")


# ------------------ Model Training ------------------ #

def train_catboost(X_train, y_train, X_valid=None, y_valid=None, catboost_params=None, cat_features=None):
    """
    Trains a CatBoostRegressor on the given data.
    `cat_features` should be a list of column names (not indices).
    """
    if cat_features is None:
        raise ValueError("You must specify `cat_features` when using raw categorical columns.")

    model = CatBoostRegressor(**(catboost_params or {}))

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid) if X_valid is not None else None,
        cat_features=cat_features,
        verbose=0
    )

    return model


def train_catboost_quantiles(X_train, y_train, alpha, X_valid=None, y_valid=None, catboost_quantile_params=None, cat_features=None):
    """
    Trains CatBoost for quantile regression with the given alpha level.
    """
    if cat_features is None:
        raise ValueError("You must specify `cat_features` when using raw categorical columns.")

    params = (catboost_quantile_params or {}).copy()
    params["loss_function"] = f"Quantile:alpha={alpha}"

    model = CatBoostRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid) if X_valid is not None else None,
        cat_features=cat_features,
        verbose=0
    )

    return model


# ------------------ Prediction Utilities ------------------ #

def predict(model, X_test):
    return model.predict(X_test)


def predict_interval(m_low, m_median, m_high, X_test):
    """
    Returns a matrix of [low, median, high] predictions.
    """
    return np.vstack([
        m_low.predict(X_test),
        m_median.predict(X_test),
        m_high.predict(X_test)
    ]).T


if __name__ == "__main__":
    preprocess_for_catboost()