"""Deduplication utilities for molecular data."""

from __future__ import annotations

import pandas as pd


def find_duplicates(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    target_col: str | None = None,
) -> pd.DataFrame:
    """Find duplicate SMILES in DataFrame.

    Returns DataFrame with duplicate groups and their counts.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col} not found in DataFrame")

    dup_mask = df.duplicated(subset=[smiles_col], keep=False)
    dups = df[dup_mask].copy()

    if len(dups) == 0:
        return pd.DataFrame(columns=["canonical_smiles", "count", "duplicate_group"])

    # Assign duplicate group IDs
    dup_groups = dups.groupby(smiles_col).ngroup()
    dups["duplicate_group"] = dup_groups.values

    # Add count information
    counts = dups.groupby(smiles_col).size().reset_index(name="count")
    dups = dups.merge(counts, on=smiles_col)

    return dups


def deduplicate_by_mean(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    target_col: str = "pLC50",
    group_col: str = "duplicate_group",
) -> pd.DataFrame:
    """Deduplicate by averaging target values within each SMILES group.

    Returns DataFrame with one row per unique SMILES.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column {smiles_col} not found")

    # Keep first occurrence for non-target columns
    agg_dict = {}
    for col in df.columns:
        if col == smiles_col:
            continue
        elif col == target_col:
            agg_dict[col] = "mean"
        elif col == group_col:
            agg_dict[col] = "first"
        else:
            agg_dict[col] = "first"

    deduped = df.groupby(smiles_col, as_index=False).agg(agg_dict)
    return deduped


def assign_duplicate_groups(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
) -> pd.DataFrame:
    """Assign duplicate_group column based on SMILES identity."""
    out = df.copy()
    out["duplicate_group"] = out.groupby(smiles_col).ngroup()
    return out


def validate_no_cross_group_leakage(
    df: pd.DataFrame,
    group_col: str,
    split_col: str,
) -> None:
    """Validate that no group appears in multiple splits."""
    if group_col not in df.columns:
        return
    if split_col not in df.columns:
        return

    cross = df.dropna(subset=[group_col]).groupby(group_col)[split_col].nunique()
    bad = cross[cross > 1]
    if len(bad) > 0:
        examples = bad.index.astype(str).tolist()[:10]
        raise AssertionError(
            f"Data leakage: {group_col} crosses splits: {examples}"
        )
