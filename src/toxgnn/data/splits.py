"""Data splitting utilities with group-aware strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_splits(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
    split_col: str,
) -> pd.DataFrame:
    """Assign group-safe train/val/test splits."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8, (
        f"Splits must sum to 1.0, got {train_size + val_size + test_size}"
    )

    out = df.copy()
    groups = out[group_col].fillna(out.get("mol_id", pd.Series(range(len(out))))).astype(str).values

    if test_size > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss.split(out, groups=groups))
        tmp = out.iloc[train_val_idx].copy()
        tmp_groups = groups[train_val_idx]
    else:
        train_val_idx = np.arange(len(out))
        test_idx = np.array([], dtype=int)
        tmp = out.copy()
        tmp_groups = groups

    rel_val = val_size / (train_size + val_size) if val_size > 0 else 0.0
    if rel_val > 0:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 17)
        train_rel, val_rel = next(gss2.split(tmp, groups=tmp_groups))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    out[split_col] = "unused"
    out.loc[out.index[train_idx], split_col] = "train"
    out.loc[out.index[val_idx], split_col] = "val"
    if len(test_idx) > 0:
        out.loc[out.index[test_idx], split_col] = "test"

    validate_no_leakage(out, [group_col], split_col)
    return out


def make_loocv_folds(
    df: pd.DataFrame,
    mol_id_col: str = "mol_id",
) -> pd.DataFrame:
    """Assign LOOCV fold indices. Each sample gets its own fold as validation."""
    out = df.copy()
    n = len(out)
    out["loocv_fold"] = list(range(n))
    out["loocv_split"] = "train"
    return out


def validate_no_leakage(
    df: pd.DataFrame,
    group_cols: list[str],
    split_col: str,
) -> None:
    """Validate no data leakage across splits."""
    for col in group_cols:
        if col not in df.columns:
            continue
        cross = df.dropna(subset=[col]).groupby(col)[split_col].nunique()
        bad = cross[cross > 1]
        if len(bad) > 0:
            examples = bad.index.astype(str).tolist()[:10]
            raise AssertionError(
                f"Data leakage: {col} crosses splits: {examples}"
            )


def get_split_sizes(df: pd.DataFrame, split_col: str) -> dict[str, int]:
    """Get counts for each split."""
    return df[split_col].value_counts().to_dict()


def filter_by_split(
    df: pd.DataFrame,
    split_col: str,
    allowed_splits: list[str],
) -> pd.DataFrame:
    """Filter DataFrame to only include rows in allowed splits."""
    return df[df[split_col].isin(allowed_splits)].copy()
