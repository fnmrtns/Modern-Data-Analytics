from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def regression_metrics(y_true, y_pred):

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def quantile_coverage(y_true, intervals, lower_idx=0, upper_idx=2):
    """
    Computes the percentage of true values that fall inside predicted intervals.
    Coverage — "Do our prediction intervals actually contain the real startup delay?"
    Efficiency — "How wide are those intervals?"
    intervals: shape (n_samples, 3), columns = [lower, median, upper]
    """
    lower = intervals[:, lower_idx]
    upper = intervals[:, upper_idx]
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    avg_interval_width = np.mean(upper - lower)

    return {
        "Interval Coverage": coverage,
        "Avg Interval Width": avg_interval_width
    }

def print_metrics(name, metrics_dict):

    print(f"\n{name} Metrics:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
