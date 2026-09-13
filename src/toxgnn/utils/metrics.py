"""Regression metrics for model evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics: R2, RMSE, MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Filter out NaN values
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation coefficient."""
    from scipy.stats import pearsonr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    return float(pearsonr(x[valid], y[valid])[0])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    return float(spearmanr(x[valid], y[valid])[0])
