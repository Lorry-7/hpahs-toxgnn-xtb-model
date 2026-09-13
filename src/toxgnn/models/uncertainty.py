"""Uncertainty quantification via ensemble predictions."""

from __future__ import annotations

import math

import numpy as np


def summarize_ensemble(pred_matrix: np.ndarray) -> dict:
    """Summarize ensemble predictions.

    Args:
        pred_matrix: shape [n_models, n_targets]

    Returns:
        dict with mean, sd, ci95_low, ci95_high arrays.
    """
    pred = np.asarray(pred_matrix, dtype=float)
    return {
        "mean": pred.mean(axis=0),
        "sd": pred.std(axis=0, ddof=1) if pred.shape[0] > 1 else np.zeros(pred.shape[1]),
        "ci95_low": np.percentile(pred, 2.5, axis=0),
        "ci95_high": np.percentile(pred, 97.5, axis=0),
    }


def pLC50_to_lc50_mgL(pLC50: np.ndarray, mw: np.ndarray) -> np.ndarray:
    """Convert pLC50 to LC50 in mg/L."""
    mol_l = np.power(10.0, -np.asarray(pLC50, dtype=float))
    return mol_l * np.asarray(mw, dtype=float) * 1000.0


def bootstrap_ensemble(
    X_graph: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_bootstrap: int = 40,
    fraction: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate bootstrap ensemble predictions.

    Returns array of shape [n_bootstrap, n_samples].
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    sample_size = int(n * fraction)
    preds = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        # Bootstrap sample
        idx = rng.choice(n, size=sample_size, replace=True)
        unique_frac = len(set(idx)) / n

        # Fit model
        model = model_factory()
        model.fit(X_graph[idx], X_xtb[idx], y[idx])

        # Predict on all samples
        preds[b] = model.predict(X_graph, X_xtb)

    return preds
