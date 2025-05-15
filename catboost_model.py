from catboost import CatBoostRegressor, Pool
import numpy as np

# Optional: Set default params centrally
DEFAULT_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "verbose": 0,
    "random_seed": 42
}

def train_catboost(X_train, y_train, X_valid=None, y_valid=None, params=None):
    """Train a CatBoost regressor on point prediction."""
    p = DEFAULT_PARAMS.copy()
    if params:
        p.update(params)
    
    model = CatBoostRegressor(**p)

    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    else:
        model.fit(X_train, y_train)

    return model

def train_catboost_quantiles(X_train, y_train, alpha, X_valid=None, y_valid=None):
    """
    Train a quantile regression CatBoost model.
    alpha = 0.5 → median
    alpha = 0.05 → 5th percentile
    alpha = 0.95 → 95th percentile
    """
    model = CatBoostRegressor(
        loss_function='Quantile:alpha={}'.format(alpha),
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=42
    )

    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    else:
        model.fit(X_train, y_train)

    return model

def predict(model, X_test):
    """Standard prediction function."""
    return model.predict(X_test)

def predict_interval(m_low, m_median, m_high, X_test):
    """
    Predict lower, median, and upper quantiles using three models.
    Returns array of shape (n_samples, 3).
    """
    p_low = m_low.predict(X_test)
    p_med = m_median.predict(X_test)
    p_high = m_high.predict(X_test)

    return np.vstack([p_low, p_med, p_high]).T
