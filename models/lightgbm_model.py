import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from config.paths import data_path

# ------------------ Preprocessing ------------------ #

def preprocess_for_lightgbm(input_file="selected_features.json", output_file="selected_features_LGBM.json"):
    """
    Preprocess for LightGBM:
    - Converts 'pillar' and 'countryCoor' to categorical dtype
    - Saves to Data folder
    """
    df = pd.read_json(data_path(input_file))
    df["pillar"] = df["pillar"].astype("category")
    df["countryCoor"] = df["countryCoor"].astype("category")
    df.to_json(data_path(output_file), orient="records", indent=2)
    print(f"LightGBM preprocessing complete. Saved to {output_file}")


# ------------------ Model Training ------------------ #

def load_lgbm_data(file="selected_features_LGBM.json"):
    df = pd.read_json(data_path(file))
    df["pillar"] = df["pillar"].astype("category")
    df["countryCoor"] = df["countryCoor"].astype("category")
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_lightgbm(X_train, y_train, lgbm_params=None):
    model = LGBMRegressor(**(lgbm_params or {}))
    model.fit(X_train, y_train)
    return model


def train_lightgbm_quantiles(X_train, y_train, alphas=[0.1, 0.5, 0.9], base_params=None):
    """
    Trains LightGBM models for each quantile (e.g., p10, median, p90).
    Returns a list of trained models.
    """
    models = []
    for alpha in alphas:
        params = dict(base_params or {})
        params["objective"] = "quantile"
        params["alpha"] = alpha
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models.append(model)
    return models


def predict_interval(models, X_test):
    """
    Returns np.array of shape (n, 3): [lower, median, upper]
    """
    preds = [m.predict(X_test) for m in models]
    return np.vstack(preds).T


# ------------------ Manual Test ------------------ #

if __name__ == "__main__":
    preprocess_for_lightgbm()
