"""Residual adapter for conservative toxicity adaptation.

Reference: ToxGNN_Project_2 - q6-true/models/predictor.py
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualAdapter(nn.Module):
    """Residual adapter: z2 = z + alpha * Adapter(z).

    Reference architecture: bottleneck=32, dropout=0.1
    """

    def __init__(self, dim: int = 128, bottleneck: int = 32, alpha: float = 0.10, dropout: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.alpha * self.adapter(z)


class LC50AdapterModel(nn.Module):
    """Frozen backbone + residual adapter + MLP head for LC50 prediction.

    Reference architecture: head_hidden=128 (same as embedding dim)
    """

    def __init__(
        self,
        frozen_backbone,
        adapter: ResidualAdapter,
        head_hidden: int = 128,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.backbone = frozen_backbone
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.adapter = adapter
        embed_dim = frozen_backbone.output_dim
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        batch,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple:
        with torch.no_grad():
            z0 = self.backbone.encode(batch)

        z = self.adapter(z0)
        y = self.head(z).squeeze(-1)

        if return_embeddings:
            return y, z0, z
        return y

    def get_adapted_embeddings(self, batch) -> torch.Tensor:
        """Get adapted embeddings without prediction."""
        with torch.no_grad():
            z0 = self.backbone.encode(batch)
        return self.adapter(z0)


class DirectFinetuneModel(nn.Module):
    """Direct fine-tuning model for ablation (no adapter)."""

    def __init__(self, backbone, head_hidden: int = 128, dropout: float = 0.20):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone.output_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        z = self.backbone.encode(batch)
        return self.head(z).squeeze(-1)
