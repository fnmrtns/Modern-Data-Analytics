from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config.paths import data_path

def preprocess_for_random_forest(input_file="selected_features.json", output_file="selected_features_RF.json", top_n=10):
    # Load data
    df = pd.read_json(data_path(input_file))

    # Label encode 'pillar'
    le = LabelEncoder()
    df['pillar_encoded'] = le.fit_transform(df['pillar'].astype(str))

    # Group and one-hot encode 'countryCoor'
    top_countries = df['countryCoor'].value_counts().nlargest(top_n).index
    df['country_clean'] = df['countryCoor'].where(df['countryCoor'].isin(top_countries), 'Other')
    country_dummies = pd.get_dummies(df['country_clean'], prefix='country')

    # Drop and combine
    df_encoded = pd.concat([
        df.drop(columns=['pillar', 'countryCoor', 'country_clean']),
        country_dummies
    ], axis=1)

    # Save to file
    df_encoded.to_json(data_path(output_file), orient="records", indent=2)

    print(f"Random Forest preprocessing complete. Saved to {output_file}")

if __name__ == "__main__":
    preprocess_for_random_forest()

def load_rf_data():
    """Load preprocessed RF features and return X, y, and train/test split."""
    df = pd.read_json(data_path("selected_features_RF.json"))
    X = df.drop(columns=["startupDelay", "id"])
    y = df["startupDelay"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train, rf_params):
    """Train a RandomForestRegressor with the given parameters."""
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    return model