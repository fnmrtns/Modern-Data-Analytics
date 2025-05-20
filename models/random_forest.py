from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_random_forest(X_train, y_train, X_valid=None, y_valid=None, rf_params=None):
    model = RandomForestRegressor(**(rf_params or {}))

    model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    return model.predict(X_test)
