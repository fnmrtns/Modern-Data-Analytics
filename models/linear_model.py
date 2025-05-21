import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 📦 Shared project paths
from config.paths import data_path


def preprocess_and_save(raw_path, processed_path):
    """Preprocess raw JSON file and save linear-ready feature set."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} does not exist.")

    df = pd.read_json(raw_path).dropna()
    numeric_features = ["ecMaxContribution", "duration", "contRatio", "numberOrg"]

    # Normalize totalCost
    scaler = StandardScaler()
    nonzero_mask = df["totalCost"] != 0
    df.loc[nonzero_mask, "totalCost"] = scaler.fit_transform(df.loc[nonzero_mask, ["totalCost"]])
    df.loc[~nonzero_mask, "totalCost"] = 0

    # Normalize other features
    scaler_other = StandardScaler()
    df[numeric_features] = scaler_other.fit_transform(df[numeric_features])

    # One-hot encode top 10 pillar categories
    top_pillars = df["pillar"].value_counts().head(10).index
    df["pillar_simplified"] = df["pillar"].where(df["pillar"].isin(top_pillars), "pillar_Other")
    pillar_dummies = pd.get_dummies(df["pillar_simplified"], prefix="pillar", drop_first=True)

    # Frequency encode countryCoor
    country_freq = df["countryCoor"].value_counts(normalize=True)
    df["countryCoor_freq"] = df["countryCoor"].map(country_freq)

    # Final DataFrame
    final_df = pd.concat([
        df[["id", "totalCost", "totalCostzero"] + numeric_features + ["countryCoor_freq", "startupDelay"]],
        pillar_dummies
    ], axis=1)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    final_df.to_json(processed_path, orient="records", indent=2)
    print(f"✅ Preprocessed data saved to: {processed_path}")


def plot_correlation_heatmap(processed_path):
    """Plot correlation matrix of numeric features in processed data."""
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"{processed_path} not found.")

    df = pd.read_json(processed_path)
    cols = ["totalCost", "ecMaxContribution", "duration", "contRatio", "numberOrg", "countryCoor_freq", "totalCostzero"]
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    print("✅ Columns included:", X.columns.tolist())
    print("✅ Shape:", X.shape)

    corr = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Minimal Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


def run_vif_check(processed_path):
    """Compute and print VIF scores for selected numerical features."""
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"{processed_path} not found.")

    df = pd.read_json(processed_path)
    cols = ["totalCost", "ecMaxContribution", "duration", "contRatio", "numberOrg", "countryCoor_freq", "totalCostzero"]
    X = df[cols].apply(pd.to_numeric, errors="coerce").dropna().astype(float)

    vif_data = pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    print("\n📊 Variance Inflation Factors:")
    print(vif_data.sort_values("VIF", ascending=False))


def load_features_and_split(processed_path, test_size=0.2, random_state=42):
    """Load processed data and return train/test splits."""
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"{processed_path} not found.")

    df = pd.read_json(processed_path)
    X = df.drop(columns=["id", "startupDelay"])
    y = df["startupDelay"]
    X = X.astype({col: int for col in X.columns if X[col].dtype == "bool"})

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_linear_regression(X_train, y_train, linreg_params=None):
    """Train a LinearRegression model."""
    linreg_params = linreg_params or {}
    model = LinearRegression(**linreg_params)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # Example: test script manually
    raw_file = data_path("selected_features.json")
    processed_file = data_path("selected_features_linear.json")
    # preprocess_and_save(raw_file, processed_file)
    # plot_correlation_heatmap(processed_file)
    run_vif_check(processed_file)
