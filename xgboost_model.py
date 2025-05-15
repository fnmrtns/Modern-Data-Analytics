from xgboost import XGBRegressor
import numpy as np

def train_xgboost(X_train, y_train, X_valid=None, y_valid=None, xgb_params=None):

    model = XGBRegressor(**(xgb_params or {}))

    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return model

def predict(model, X_test):

    return model.predict(X_test)
