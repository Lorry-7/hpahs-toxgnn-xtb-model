"""PyTorch Dataset for molecular graph data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from toxgnn.features.graph import mol_to_pyg_data


class MolGraphDataset(Dataset):
    """Dataset that converts SMILES to PyG Data objects on-the-fly."""

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str = "canonical_smiles",
        target_col: str | None = None,
        mol_id_col: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.mol_id_col = mol_id_col
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = str(row[self.smiles_col])
        target = float(row[self.target_col]) if self.target_col and self.target_col in row.index else None
        mol_id = str(row[self.mol_id_col]) if self.mol_id_col and self.mol_id_col in row.index else f"mol_{idx}"

        data = mol_to_pyg_data(smiles, y=target, mol_id=mol_id)
        return data


class CachedMolGraphDataset(Dataset):
    """Dataset that loads pre-computed PyG Data from disk cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], weights_only=False)
