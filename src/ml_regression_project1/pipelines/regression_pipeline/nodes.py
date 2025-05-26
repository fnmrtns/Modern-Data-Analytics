# src/ml_regression_project/pipelines/regression_pipeline/nodes.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from typing import List
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def coma_to_point(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Replaces commas with points and converts to numeric.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df[column] = (
        df[column].astype(str).str.replace(",", ".", regex=False)
    )
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def enrich_project_data(df: pd.DataFrame, organization: pd.DataFrame) -> pd.DataFrame:
    
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if not isinstance(organization, pd.DataFrame):
        organization = pd.DataFrame(organization)


    # Número de organizaciones por proyecto
    organization_num = organization.groupby("projectID").size().reset_index(name="num_org")

    # Coordinadores por país
    organization_country = organization.query("role == 'coordinator'")[["projectID", "country"]]
    aux = organization_country.groupby("country").size().reset_index(name="n").sort_values(by="n", ascending=False)
    aux["cumsum"] = aux["n"].cumsum()
    aux["cumsum_per"] = aux["cumsum"] / aux["n"].sum()
    aux1 = aux.query("cumsum_per < 0.9")

    # Merge con número de organizaciones
    df1 = df.merge(organization_num, left_on="id", right_on="projectID", how="left")

    # Merge con país del coordinador
    df1 = df1.merge(organization_country, left_on="id", right_on="projectID", how="left")
    df1["country1"] = np.where(df1["country"].isin(aux1["country"]), df1["country"], "Other")

    # Fechas
    df1["startDate1"] = pd.to_datetime(df1["startDate"], errors="coerce")
    df1["ecSignatureDate1"] = pd.to_datetime(df1["ecSignatureDate"], errors="coerce")
    df1["endDate1"] = pd.to_datetime(df1["endDate"], errors="coerce")

    df1["delay_d"] = np.where(df1["startDate1"] < df1["ecSignatureDate1"], 0,
                              df1["startDate1"] - df1["ecSignatureDate1"])
    df1["delay_d1"] = pd.to_timedelta(df1["delay_d"]).dt.days
    df1["delay_m"] = (df1["delay_d1"] / 30.44).round(2)

    df1["pry_duration_d"] = np.where(df1["endDate1"] < df1["startDate1"], 0,
                                     df1["endDate1"] - df1["startDate1"])
    df1["pry_duration_d1"] = pd.to_timedelta(df1["pry_duration_d"]).dt.days
    df1["pry_duration_m"] = (df1["pry_duration_d1"] / 30.44).round(2)

    df1 = coma_to_point(df1, "totalCost")
    df1 = coma_to_point(df1, "ecMaxContribution")

    df1["ratio"] = np.where(df1["totalCost"] == 0, 1, df1["ecMaxContribution"] / df1["totalCost"])
    df1["cost0"] = np.where(df1["totalCost"] == 0, 1, 0)

    # Selección final de columnas
    final_cols = [
        "id", "totalCost", "ecMaxContribution", "pry_duration_m", "delay_m",
        "ratio", "num_org", "cost0", "country1", "legalBasis"
    ]
    return df1[final_cols].copy()

def remove_outliers_isolation_forest(data: pd.DataFrame, contamination: float = 0.05):
    """
    Remove outliers using Isolation Forest.
    Assumes all columns are numeric and includes target column in filtering.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    features = data.select_dtypes(include="number")  # puedes ajustar esto si es necesario
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(features)
    
    # -1 = outlier, 1 = inlier
    mask = preds == 1
    cleaned_data = data[mask].reset_index(drop=True)

    return cleaned_data

def split_data(data, test_size: float):

    data = data.dropna()
    X = data.drop(columns=["delay_m"])
    y = data["delay_m"]
    
    return train_test_split(X, y, test_size=test_size, random_state=42)

def fit_transform_features(X_train: pd.DataFrame, X_test: pd.DataFrame, columns_to_scale: list, columns_to_encode: list):
    """
    Entrena un ColumnTransformer en X_train y aplica la transformación en X_train y X_test.
    Devuelve X_train_trans, X_test_trans, y el transformer entrenado.
    """
    transformer = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), columns_to_scale),
            ("dummies", OneHotEncoder(handle_unknown="ignore"), columns_to_encode)
        ],
        remainder="drop"
    )
    
    X_train_trans = transformer.fit_transform(X_train)
    X_test_trans = transformer.transform(X_test)

    return X_train_trans, X_test_trans, transformer

def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.4f}")
    return mse
