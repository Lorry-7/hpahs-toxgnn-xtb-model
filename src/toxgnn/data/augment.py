"""Tautomer augmentation for molecular data."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def generate_tautomers(
    smiles: str,
    max_tautomers: int = 5,
) -> list[str]:
    """Generate canonical tautomers for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)

    canonical_set = set()
    for taut in tautomers:
        can = Chem.MolToSmiles(taut, canonical=True)
        canonical_set.add(can)

    result = sorted(canonical_set)[:max_tautomers]
    return result


def augment_with_tautomers(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    target_col: str = "exp_logp",
    tautomer_key_col: str = "tautomer_family_key",
    max_tautomers: int = 5,
    train_only: bool = True,
    split_col: str | None = None,
) -> pd.DataFrame:
    """Augment training data with tautomers.

    If train_only=True, only augment rows where split_col == 'train'.
    Tautomers inherit parent's split assignment.
    """
    augmented_rows = []

    for _, row in df.iterrows():
        if train_only and split_col and row.get(split_col) != "train":
            augmented_rows.append(row.to_dict())
            continue

        smiles = str(row[smiles_col])
        tautomers = generate_tautomers(smiles, max_tautomers)

        # Add original
        augmented_rows.append(row.to_dict())

        # Add tautomers (excluding original)
        original_canonical = Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles), canonical=True
        ) if Chem.MolFromSmiles(smiles) else smiles

        for taut_smi in tautomers:
            if taut_smi == original_canonical:
                continue
            new_row = row.to_dict()
            new_row[smiles_col] = taut_smi
            new_row["is_tautomer_augmented"] = True
            augmented_rows.append(new_row)

    result = pd.DataFrame(augmented_rows)
    if "is_tautomer_augmented" not in result.columns:
        result["is_tautomer_augmented"] = False
    else:
        result["is_tautomer_augmented"] = result["is_tautomer_augmented"].fillna(False).infer_objects(copy=False)

    return result
