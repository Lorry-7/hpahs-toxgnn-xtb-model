"""Atom and substructure masking for explainability."""

from __future__ import annotations

import copy

import torch
import pandas as pd


@torch.no_grad()
def atom_feature_zero_masking(
    model,
    data,
    device: str = "cpu",
) -> pd.DataFrame:
    """Approximate atom contribution by zeroing each atom feature vector."""
    model.eval()
    data = data.to(device)
    base = model(data).detach().cpu().item()

    rows = []
    for idx in range(data.x.size(0)):
        masked = copy.deepcopy(data)
        masked.x[idx, :] = 0.0
        pred = model(masked.to(device)).detach().cpu().item()
        rows.append({
            "atom_idx": idx,
            "base_pred": base,
            "masked_pred": pred,
            "delta_pLC50": base - pred,
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def edge_masking(
    model,
    data,
    device: str = "cpu",
) -> pd.DataFrame:
    """Approximate edge contribution by removing each edge pair."""
    model.eval()
    data = data.to(device)
    base = model(data).detach().cpu().item()

    n_edges = data.edge_index.size(1)
    rows = []

    # Process edges in pairs (bidirectional)
    processed = set()
    for i in range(n_edges):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        edge_pair = (min(src, dst), max(src, dst))
        if edge_pair in processed:
            continue
        processed.add(edge_pair)

        # Mask this edge pair
        mask = torch.ones(n_edges, dtype=torch.bool, device=device)
        for j in range(n_edges):
            s, d = data.edge_index[0, j].item(), data.edge_index[1, j].item()
            if (min(s, d), max(s, d)) == edge_pair:
                mask[j] = False

        masked = copy.deepcopy(data)
        masked.edge_index = data.edge_index[:, mask]
        masked.edge_attr = data.edge_attr[mask]

        pred = model(masked.to(device)).detach().cpu().item()
        rows.append({
            "src_atom": src,
            "dst_atom": dst,
            "base_pred": base,
            "masked_pred": pred,
            "delta_pLC50": base - pred,
        })

    return pd.DataFrame(rows)


def compute_shapley_approximation(
    model,
    data,
    device: str = "cpu",
    n_samples: int = 100,
) -> pd.DataFrame:
    """Approximate Shapley values for atoms using sampling."""
    import numpy as np

    model.eval()
    data = data.to(device)
    n_atoms = data.x.size(0)
    base = model(data).detach().cpu().item()

    shapley_values = np.zeros(n_atoms)

    for _ in range(n_samples):
        # Random permutation
        perm = np.random.permutation(n_atoms)

        for idx, atom_idx in enumerate(perm):
            # With atom
            mask_with = torch.zeros(n_atoms, dtype=torch.bool, device=device)
            mask_with[perm[:idx + 1]] = True

            # Without atom
            mask_without = torch.zeros(n_atoms, dtype=torch.bool, device=device)
            if idx > 0:
                mask_without[perm[:idx]] = True

            # Compute marginal contribution
            data_with = copy.deepcopy(data)
            data_with.x[~mask_with] = 0.0
            pred_with = model(data_with.to(device)).detach().cpu().item()

            data_without = copy.deepcopy(data)
            data_without.x[~mask_without] = 0.0
            pred_without = model(data_without.to(device)).detach().cpu().item()

            shapley_values[atom_idx] += (pred_with - pred_without)

    shapley_values /= n_samples

    mol = Chem.MolFromSmiles(data.smiles)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()] if mol else ["?"] * n_atoms

    return pd.DataFrame({
        "atom_idx": range(n_atoms),
        "atom_symbol": symbols,
        "shapley_value": shapley_values,
    })
