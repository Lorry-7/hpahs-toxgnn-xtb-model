"""Compare model predictions with TEST software predictions."""

from __future__ import annotations

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compare_with_test(
    final_pred: pd.DataFrame,
    test_df: pd.DataFrame,
    large_diff_threshold: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Compare model predictions with TEST predictions.

    Args:
        final_pred: DataFrame with hpah_id and pred_pLC50_mean columns
        test_df: DataFrame with hpah_id and TEST_pLC50 columns
        large_diff_threshold: Threshold for large difference (log units)

    Returns:
        Tuple of (merged DataFrame, comparison metrics dict)
    """
    merged = final_pred.merge(
        test_df[["hpah_id", "TEST_pLC50"]], on="hpah_id", how="left"
    )
    merged["delta_vs_TEST"] = merged["pred_pLC50_mean"] - merged["TEST_pLC50"]
    merged["abs_delta_vs_TEST"] = merged["delta_vs_TEST"].abs()
    merged["agreement_flag"] = pd.cut(
        merged["abs_delta_vs_TEST"],
        bins=[-0.001, 0.5, large_diff_threshold, 999],
        labels=["consistent", "moderate_difference", "large_difference"],
    ).astype(str)

    # Calculate metrics
    valid = merged.dropna(subset=["pred_pLC50_mean", "TEST_pLC50"])
    metrics = {
        "n": int(len(valid)),
        "pearson_r": (
            float(pearsonr(valid["pred_pLC50_mean"], valid["TEST_pLC50"])[0])
            if len(valid) >= 3
            else None
        ),
        "spearman_rho": (
            float(spearmanr(valid["pred_pLC50_mean"], valid["TEST_pLC50"])[0])
            if len(valid) >= 3
            else None
        ),
        "mean_delta": float(valid["delta_vs_TEST"].mean()) if len(valid) else None,
        "median_abs_delta": (
            float(valid["abs_delta_vs_TEST"].median()) if len(valid) else None
        ),
        "n_consistent": int((merged["agreement_flag"] == "consistent").sum()),
        "n_moderate": int((merged["agreement_flag"] == "moderate_difference").sum()),
        "n_large": int((merged["agreement_flag"] == "large_difference").sum()),
    }

    return merged, metrics


def generate_priority_class(
    df: pd.DataFrame,
    toxicity_quantile: float = 0.75,
    uncertainty_quantile: float = 0.75,
) -> pd.DataFrame:
    """Assign priority classes A/B/C/D based on toxicity and uncertainty.

    Priority A: High toxicity, low uncertainty (most urgent)
    Priority B: High toxicity, high uncertainty
    Priority C: Low toxicity, low uncertainty
    Priority D: Low toxicity, high uncertainty
    """
    out = df.copy()

    tox_threshold = out["pred_pLC50_mean"].quantile(toxicity_quantile)
    unc_threshold = out["pred_pLC50_sd"].quantile(uncertainty_quantile)

    def _classify(row):
        high_tox = row["pred_pLC50_mean"] >= tox_threshold
        high_unc = row["pred_pLC50_sd"] >= unc_threshold
        if high_tox and not high_unc:
            return "A"
        elif high_tox and high_unc:
            return "B"
        elif not high_tox and not high_unc:
            return "C"
        else:
            return "D"

    out["priority_class"] = out.apply(_classify, axis=1)
    return out
