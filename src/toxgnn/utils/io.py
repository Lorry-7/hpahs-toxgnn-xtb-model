"""I/O utilities for data loading and saving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load CSV file with UTF-8 encoding."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, encoding="utf-8", **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save DataFrame to CSV with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        return pd.read_parquet(path, **kwargs)
    except OSError:
        if "engine" in kwargs:
            raise
        return pd.read_parquet(path, engine="fastparquet", **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save DataFrame to Parquet format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
