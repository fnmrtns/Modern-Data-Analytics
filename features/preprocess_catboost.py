# preprocess_catboost.py

import pandas as pd
import numpy as np
import os
from datetime import datetime


def parse_date(d):
    return pd.to_datetime(d, format="%Y-%m-%d", errors="coerce")

def preprocess_project_data(filepath):
    # Load data

    df = pd.read_json(filepath)
    
    # Date parsing
    df["ecSignatureDate"] = parse_date(df["ecSignatureDate"])
    df["startDate"] = parse_date(df["startDate"])
    df["endDate"] = parse_date(df["endDate"])
    
    # Numerical fields
    df["totalCost"] = pd.to_numeric(df["totalCost"], errors="coerce")
    df["ecMaxContribution"] = pd.to_numeric(df["ecMaxContribution"], errors="coerce")
    
    # Derived features
    df["startupDelay"] = (df["startDate"] - df["ecSignatureDate"]).dt.days
    df["percEUcont"] = df["ecMaxContribution"] / df["totalCost"]
    df["duration"] = (df["endDate"] - df["startDate"]).dt.days

    # Features to keep
    features = ["totalCost", "percEUcont", "fundingScheme", "duration", "masterCall"]
    target = "startupDelay"
    id_col = "id"

    # Drop rows with missing values in required columns
    df = df[[id_col] + features + [target]].dropna()

    # Separate X and y
    X = df[features]
    y = df[target]
    meta = df[[id_col]]  # To track project identifiers

    # Identify categorical features for CatBoost
    cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]

    return X, y, cat_features, meta
