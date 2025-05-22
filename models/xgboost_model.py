import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from config.paths import data_path

# ------------------ Preprocessing ------------------ #

def preprocess_for_xgboost(input_file="selected_features.json", output_file="selected_features_XGB.json", top_n=10):
    """
    Preprocess data for XGBoost:
    - Label encode 'pillar'
    - One-hot encode 'countryCoor', grouping rare categories under 'Other'
    - Save as JSON to the Data/ folder
    """
    # Load data
    df = pd.read_json(data_path(input_file))

    # Label encode 'pillar'
    le_pillar = LabelEncoder()
    df["pillar_encoded"] = le_pillar.fit_transform(df["pillar"].astype(str))

    # One-hot encode 'countryCoor', grouping rare values
    top_countries = df["countryCoor"].value_counts().nlargest(top_n).index
    df["country_clean"] = df["countryCoor"].where(df["countryCoor"].isin(top_countries), "Other")
    country_dummies = pd.get_dummies(df["country_clean"], prefix="country")

    # Combine into final dataframe
    df_xgb = pd.concat([
        df.drop(columns=["pillar", "countryCoor", "country_clean"]),
        country_dummies
    ], axis=1)

    # Save to file
    df_xgb.to_json(data_path(output_file), orient="records", indent=2)
    print(f"XGBoost preprocessing complete. Saved to {output_file}")


# ------------------ Model Training ------------------ #

def load_xgb_data(file="selected_features_XGB.json"):
    """
    Load preprocessed data for XGBoost and return train/test split.
    """
    df = pd.read_json(data_path(file))
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_xgboost(X_train, y_train, xgb_params=None):
    """
    Train an XGBoost regressor.
    """
    model = XGBRegressor(**(xgb_params or {}))
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    preprocess_for_xgboost()

