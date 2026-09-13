"""Generate publication-quality figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _check_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Check that DataFrame has required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def plot_parity(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: str | Path,
    xlabel: str = "Experimental",
    ylabel: str = "Predicted",
    figsize: tuple = (6, 6),
) -> None:
    """Create parity plot (predicted vs experimental)."""
    _check_required_columns(df, [x_col, y_col], "parity plot")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], alpha=0.6, edgecolors="k", linewidth=0.5)

    # Perfect prediction line
    lims = [
        min(df[x_col].min(), df[y_col].min()) - 0.5,
        max(df[x_col].max(), df[y_col].max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_with_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_low_col: str,
    ci_high_col: str,
    title: str,
    output_path: str | Path,
) -> None:
    """Create scatter plot with confidence intervals."""
    _check_required_columns(df, [x_col, y_col, ci_low_col, ci_high_col], "CI scatter")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot CI as error bars
    ax.errorbar(
        df[x_col], df[y_col],
        yerr=[df[y_col] - df[ci_low_col], df[ci_high_col] - df[y_col]],
        fmt="o", alpha=0.6, capsize=3, capthick=1,
    )

    lims = [
        min(df[x_col].min(), df[y_col].min()) - 0.5,
        max(df[x_col].max(), df[y_col].max()) + 0.5,
    ]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Experimental pLC50")
    ax.set_ylabel("Predicted pLC50")
    ax.set_title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar_comparison(
    metrics_dict: dict[str, float],
    title: str,
    output_path: str | Path,
    ylabel: str = "Metric Value",
) -> None:
    """Create bar chart comparing metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    bars = ax.bar(names, values, alpha=0.7, edgecolor="k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom",
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_path: str | Path,
    figsize: tuple = (8, 6),
) -> None:
    """Create heatmap visualization."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdBu_r",
        xticklabels=col_labels, yticklabels=row_labels,
        ax=ax, center=0,
    )
    ax.set_title(title)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curve(
    history: list[dict],
    output_path: str | Path,
    title: str = "Learning Curve",
) -> None:
    """Plot training and validation loss curves."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
