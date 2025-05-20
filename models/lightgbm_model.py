from lightgbm import LGBMRegressor
import numpy as np

def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, lgbm_params=None):

    model = LGBMRegressor(**(lgbm_params or {}))

    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return model

def train_lightgbm_quantiles(X_train, y_train, alpha, X_valid=None, y_valid=None, lgbm_quantile_params=None):

    params = lgbm_quantile_params.copy() if lgbm_quantile_params else {}
    params["objective"] = "quantile"
    params["alpha"] = alpha

    model = LGBMRegressor(**params)

    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    else:
        model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    return model.predict(X_test)

def predict_interval(m_low, m_median, m_high, X_test):
    p_low = m_low.predict(X_test)
    p_med = m_median.predict(X_test)
    p_high = m_high.predict(X_test)

    return np.vstack([p_low, p_med, p_high]).T
