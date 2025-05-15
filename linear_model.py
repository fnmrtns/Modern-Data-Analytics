from sklearn.linear_model import LinearRegression
import numpy as np

def train_linear_regression(X_train, y_train, linreg_params=None):

    model = LinearRegression(**(linreg_params or {}))
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
 
    return model.predict(X_test)
