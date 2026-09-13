"""GATv2 backbone and GNN regressor models.

Reference: ToxGNN_Project_2 - q6-true/models/gnn_backbone.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2Backbone(nn.Module):
    """GATv2 backbone with edge features and attention.

    Reference architecture: 3-layer GATv2, no BatchNorm, ReLU after each layer.
    """

    def __init__(
        self,
        node_dim: int = 30,
        edge_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        drop_edge_rate: float = 0.0,
    ):
        super().__init__()
        self.drop_edge_rate = drop_edge_rate
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )

        self.output_dim = hidden_dim
        self.last_attention = None

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        """Forward pass matching reference architecture.

        Args:
            x: (sum_nodes, node_dim) node features
            edge_index: (2, sum_edges) edge indices
            edge_attr: (sum_edges, edge_dim) edge features
            batch: (sum_nodes,) batch assignment vector
            return_attention: whether to extract attention weights from last layer

        Returns:
            graph_embedding: (batch_size, hidden_dim)
            edge_att: (optional) attention weights tuple
        """
        # DropEdge: randomly drop edges during training
        if self.training and self.drop_edge_rate > 0:
            num_edges = edge_index.size(1)
            keep_mask = (
                torch.rand(num_edges, device=edge_index.device) > self.drop_edge_rate
            )
            edge_index = edge_index[:, keep_mask]
            edge_attr = edge_attr[keep_mask]

        self.last_attention = None
        edge_att = None

        # Forward through layers
        for i, conv in enumerate(self.convs):
            is_last = i == len(self.convs) - 1
            if is_last and return_attention:
                x, edge_att = conv(
                    x, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True,
                )
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            if not (is_last and return_attention):
                x = torch.relu(x)

        # Global mean pooling
        graph_embedding = global_mean_pool(x, batch)

        if return_attention:
            return graph_embedding, edge_att
        return graph_embedding

    def encode_nodes(self, batch, return_attention: bool = False) -> torch.Tensor:
        """Encode nodes through GATv2 layers (legacy interface)."""
        if return_attention:
            emb, att = self.forward(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                return_attention=True,
            )
            self.last_attention = att
            return emb
        return self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def encode(self, batch, return_attention: bool = False) -> torch.Tensor:
        """Encode graph-level representation via mean pooling."""
        return self.encode_nodes(batch, return_attention=return_attention)


class GNNRegressor(nn.Module):
    """GNN backbone with MLP prediction head."""

    def __init__(
        self,
        backbone: GATv2Backbone,
        head_hidden: int = 128,
        dropout: float = 0.20,
    ):
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

    def encode(self, batch) -> torch.Tensor:
        """Get backbone embeddings without prediction head."""
        return self.backbone.encode(batch)
