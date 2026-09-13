"""Baseline toxicity models (pLC50 ~ LogP regression)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class _UnivariateOLS:
    """Closed-form one-feature OLS, avoiding a heavyweight LAPACK call."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_UnivariateOLS":
        x = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        x_centered = x - x.mean()
        denominator = float(np.dot(x_centered, x_centered))
        if denominator == 0.0:
            raise ValueError("Cannot fit baseline: predictor has zero variance")
        self.coef_ = np.array([float(np.dot(x_centered, y - y.mean()) / denominator)])
        self.intercept_ = float(y.mean() - self.coef_[0] * x.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float).reshape(-1)
        return self.intercept_ + self.coef_[0] * x


class BaselineToxicityModel:
    """Fit pLC50 = a * LogP + b on train data only."""

    def __init__(self, feature: str = "exp_logp"):
        self.feature = feature
        self.regressor = _UnivariateOLS()
        self.is_fitted = False
        self.coefficients = None

    def fit(self, train_df: pd.DataFrame) -> "BaselineToxicityModel":
        train = train_df.dropna(subset=[self.feature, "pLC50"]).copy()
        if len(train) < 5:
            raise ValueError(
                f"Too few rows to fit baseline with feature={self.feature}"
            )
        X = train[[self.feature]].astype(float).values
        y = train["pLC50"].astype(float).values
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.coefficients = {
            "intercept": float(self.regressor.intercept_),
            "slope": float(self.regressor.coef_[0]),
        }
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("BaselineToxicityModel must be fitted before predict")
        X = df[[self.feature]].astype(float).values
        return self.regressor.predict(X)

    def add_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["baseline_pLC50"] = self.predict(out)
        if "pLC50" in out.columns:
            out["residual_pLC50"] = out["pLC50"].astype(float) - out["baseline_pLC50"]
        return out


def regression_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}
    mse = mean_squared_error(y_true[valid], y_pred[valid])
    return {
        "r2": float(r2_score(y_true[valid], y_pred[valid])),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true[valid], y_pred[valid])),
    }
