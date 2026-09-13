"""Molecular graph construction for PyTorch Geometric."""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, BondType, HybridizationType
from torch_geometric.data import Data

ATOM_TYPES = ["C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "Other"]
DEGREES = [0, 1, 2, 3, 4]  # 4 means 4 or higher
CHARGES = [-1, 0, 1]  # 1 also catches other positive charge
HYBRIDS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
]
NUM_HS = [0, 1, 2, 3]  # 3 means 3 or higher
STEREO = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
]


def one_hot(value, choices: list, other_last: bool = True) -> list[float]:
    """One-hot encode a value given a list of choices."""
    if other_last and value not in choices:
        value = choices[-1]
    return [1.0 if value == x else 0.0 for x in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    """Extract 30-dimensional atom features."""
    symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_TYPES[:-1] else "Other"
    degree = atom.GetTotalDegree()
    degree = degree if degree in DEGREES[:-1] else 4
    chg = atom.GetFormalCharge()
    if chg not in [-1, 0, 1]:
        chg = 1 if chg > 0 else -1
    hybrid = atom.GetHybridization()
    hybrid = hybrid if hybrid in HYBRIDS[:-1] else HybridizationType.SP3D
    hs = atom.GetTotalNumHs()
    hs = hs if hs in NUM_HS[:-1] else 3

    feats = []
    feats += one_hot(symbol, ATOM_TYPES)  # 10
    feats += one_hot(degree, DEGREES)  # 5 -> 15
    feats += one_hot(chg, CHARGES)  # 3 -> 18
    feats += one_hot(hybrid, HYBRIDS)  # 4 -> 22
    feats.append(float(atom.GetIsAromatic()))  # 1 -> 23
    feats += one_hot(hs, NUM_HS)  # 4 -> 27
    feats.append(float(atom.IsInRing()))  # 1 -> 28
    feats.append(float(symbol in ["F", "Cl", "Br", "I"]))  # 1 -> 29
    feats.append(float(atom.HasProp("_CIPCode")))  # 1 -> 30

    assert len(feats) == 30, f"Expected 30 atom features, got {len(feats)}"
    return feats


def bond_features(bond: Chem.Bond) -> list[float]:
    """Extract 11-dimensional bond features."""
    btype = bond.GetBondType()
    feats = [
        float(btype == BondType.SINGLE),
        float(btype == BondType.DOUBLE),
        float(btype == BondType.TRIPLE),
        float(btype == BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    stereo = bond.GetStereo()
    stereo = stereo if stereo in STEREO[:-1] else BondStereo.STEREOE
    feats += one_hot(stereo, STEREO)  # 4 -> total 10
    feats.append(float(bond.GetIsAromatic()))  # 1 -> total 11

    assert len(feats) == 11, f"Expected 11 bond features, got {len(feats)}"
    return feats


def mol_to_pyg_data(
    smiles: str,
    y: float | None = None,
    mol_id: str | None = None,
) -> Data:
    """Convert SMILES to PyTorch Geometric Data object.

    Creates bidirectional edges for each bond.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for graph conversion: {smiles}")

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float32,
    )

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Bidirectional edges
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bf, bf]

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 11), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    data.mol_id = mol_id or "unknown"
    data.smiles = smiles

    return data


def batch_smiles_to_pyg(
    smiles_list: list[str],
    targets: list[float] | None = None,
    mol_ids: list[str] | None = None,
) -> list[Data]:
    """Convert batch of SMILES to list of PyG Data objects."""
    if targets is None:
        targets = [None] * len(smiles_list)
    if mol_ids is None:
        mol_ids = [f"mol_{i}" for i in range(len(smiles_list))]

    data_list = []
    for smi, y, mid in zip(smiles_list, targets, mol_ids):
        try:
            data = mol_to_pyg_data(smi, y, mid)
            data_list.append(data)
        except ValueError:
            continue  # Skip invalid SMILES

    return data_list
