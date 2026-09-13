"""Generate publication-quality tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _check_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Check that DataFrame has required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def generate_dataset_statistics(
    df: pd.DataFrame,
    split_col: str,
    target_col: str,
    output_path: str | Path,
) -> pd.DataFrame:
    """Generate dataset statistics table."""
    stats = []
    for split in df[split_col].unique():
        subset = df[df[split_col] == split]
        stats.append({
            "Split": split,
            "N": len(subset),
            "Mean": subset[target_col].mean(),
            "Std": subset[target_col].std(),
            "Min": subset[target_col].min(),
            "Max": subset[target_col].max(),
        })

    result = pd.DataFrame(stats)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def generate_model_performance_table(
    metrics: dict[str, dict],
    output_path: str | Path,
) -> pd.DataFrame:
    """Generate model performance comparison table."""
    rows = []
    for model_name, model_metrics in metrics.items():
        row = {"Model": model_name}
        row.update(model_metrics)
        rows.append(row)

    result = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def generate_final_predictions_table(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Generate final HPAH predictions table."""
    required_cols = ["hpah_id", "canonical_smiles", "pred_pLC50_mean", "pred_pLC50_sd"]
    _check_required_columns(predictions, required_cols, "final predictions")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions


def format_metrics_for_display(metrics: dict) -> dict:
    """Format metrics for display with appropriate precision."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if "r2" in key.lower():
                formatted[key] = f"{value:.3f}"
            elif "mae" in key.lower() or "rmse" in key.lower():
                formatted[key] = f"{value:.3f}"
            else:
                formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = value
    return formatted
