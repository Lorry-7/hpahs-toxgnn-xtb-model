"""Molecule standardization using RDKit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

Status = Literal["success", "invalid_smiles", "salt_removed", "manual_check"]


@dataclass(frozen=True)
class StandardizedMol:
    """Result of molecule standardization."""

    raw_smiles: str
    canonical_smiles: str | None
    isomeric_smiles: str | None
    inchi_key: str | None
    skeleton_key: str | None
    tautomer_family_key: str | None
    molecular_weight: float | None
    formal_charge: int | None
    heavy_atom_count: int | None
    status: Status
    message: str


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment from a molecule."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def _tautomer_key(mol: Chem.Mol) -> str:
    """Generate canonical tautomer InChIKey for grouping."""
    enumerator = rdMolStandardize.TautomerEnumerator()
    can = enumerator.Canonicalize(mol)
    return Chem.MolToInchiKey(can)


def standardize_molecule(
    smiles: str,
    keep_largest_fragment: bool = True,
    remove_isotopes: bool = True,
) -> StandardizedMol:
    """Parse and standardize a molecule. Never return None silently."""
    raw = "" if smiles is None else str(smiles).strip()
    if not raw:
        return StandardizedMol(
            raw, None, None, None, None, None, None, None, None,
            "invalid_smiles", "Empty SMILES string",
        )
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return StandardizedMol(
            raw, None, None, None, None, None, None, None, None,
            "invalid_smiles", "RDKit MolFromSmiles failed",
        )

    status: Status = "success"
    if keep_largest_fragment and len(Chem.GetMolFrags(mol)) > 1:
        mol = _largest_fragment(mol)
        status = "salt_removed"

    if remove_isotopes:
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)

    Chem.SanitizeMol(mol)
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    ik = Chem.MolToInchiKey(mol)
    mw = float(Descriptors.MolWt(mol))
    charge = int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    heavy = int(mol.GetNumHeavyAtoms())
    taut_key = _tautomer_key(mol)
    skeleton = ik.split("-")[0] if ik else None

    return StandardizedMol(
        raw, can, iso, ik, skeleton, taut_key, mw, charge, heavy,
        status, "ok",
    )


def standardize_dataframe(
    df,
    smiles_col: str = "SMILES",
    keep_largest_fragment: bool = True,
    remove_isotopes: bool = True,
) -> tuple:
    """Standardize all SMILES in a DataFrame. Returns (results_df, invalid_report)."""
    import pandas as pd

    results = []
    invalid_rows = []

    for idx, row in df.iterrows():
        smi = str(row.get(smiles_col, "")).strip()
        mol_info = standardize_molecule(smi, keep_largest_fragment, remove_isotopes)

        result = {
            "mol_id": row.get("mol_id", f"mol_{idx}"),
            "raw_smiles": mol_info.raw_smiles,
            "canonical_smiles": mol_info.canonical_smiles,
            "isomeric_smiles": mol_info.isomeric_smiles,
            "inchi_key": mol_info.inchi_key,
            "skeleton_key": mol_info.skeleton_key,
            "tautomer_family_key": mol_info.tautomer_family_key,
            "molecular_weight": mol_info.molecular_weight,
            "formal_charge": mol_info.formal_charge,
            "heavy_atom_count": mol_info.heavy_atom_count,
            "status": mol_info.status,
            "message": mol_info.message,
        }

        # Copy other columns from original row
        for col in df.columns:
            if col != smiles_col and col not in result:
                result[col] = row[col]

        results.append(result)

        if mol_info.status == "invalid_smiles":
            invalid_rows.append({
                "mol_id": result["mol_id"],
                "raw_smiles": mol_info.raw_smiles,
                "reason": mol_info.message,
            })

    results_df = pd.DataFrame(results)
    invalid_df = pd.DataFrame(invalid_rows) if invalid_rows else pd.DataFrame(
        columns=["mol_id", "raw_smiles", "reason"]
    )

    return results_df, invalid_df
