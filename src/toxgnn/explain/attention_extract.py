"""Extract attention weights from GATv2 layers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


@torch.no_grad()
def extract_attention_weights(
    model,
    data,
    device: str = "cpu",
) -> pd.DataFrame:
    """Extract attention weights from the last GATv2 layer.

    Returns DataFrame with atom_idx, atom_symbol, attention_score columns.
    """
    model.eval()
    data = data.to(device)

    # Forward pass with attention
    _ = model.backbone.encode(data, return_attention=True)
    att = model.backbone.last_attention

    if att is None:
        raise ValueError("No attention weights available")

    # att is (edge_index, attention_weights)
    edge_index, att_weights = att

    # Aggregate attention per atom (mean of incoming edges)
    n_atoms = data.x.size(0)
    atom_attention = np.zeros(n_atoms)

    for i in range(n_atoms):
        mask = edge_index[1] == i
        if mask.any():
            atom_attention[i] = att_weights[mask].mean().cpu().item()

    # Get atom symbols
    mol = Chem.MolFromSmiles(data.smiles)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()] if mol else ["?"] * n_atoms

    return pd.DataFrame({
        "atom_idx": range(n_atoms),
        "atom_symbol": symbols,
        "attention_score": atom_attention,
    })


@torch.no_grad()
def extract_node_embeddings(
    model,
    data,
    device: str = "cpu",
) -> np.ndarray:
    """Extract node-level embeddings from the backbone."""
    model.eval()
    data = data.to(device)
    node_emb = model.backbone.encode_nodes(data)
    return node_emb.cpu().numpy()
