"""
Autoresearch: Protein Inverse Folding.
AGENT MODIFIES THIS FILE. Everything is fair game.
Usage: uv run train.py
"""

# YOUR MODEL MUST IMPLEMENT:
#   model.predict_logits(batch) -> (total_residues, 20) logits
# This method is called by evaluate_recovery() in prepare.py.
# Do not rename it or change its signature.

import os
import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    TIME_BUDGET, SEED, NUM_AMINO_ACIDS, MAX_NEIGHBORS, NUM_RBF,
    compute_edge_features, compute_node_features,
)

# Hyperparameters (edit freely)
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 3
BACKBONE_NOISE = 0.02  # Angstroms
DROPOUT = 0.1
LR = 1e-3
WARMUP_EPOCHS = 3
BATCH_SIZE_TOKENS = 10000

# Derived constants
EDGE_FEAT_DIM = 15 * NUM_RBF  # 240
NODE_FEAT_DIM = 6  # sin/cos of phi, psi, omega


class EncoderLayer(nn.Module):
    """Message-passing layer with edge updates using dense k-NN operations."""

    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
            nn.Dropout(dropout),
        )
        self.norm_node = nn.LayerNorm(hidden_dim)
        self.norm_edge = nn.LayerNorm(edge_dim)

    def forward(self, nodes, edges, knn_indices):
        """
        Args:
            nodes: (N, hidden_dim)
            edges: (N, k, edge_dim)
            knn_indices: (N, k) — neighbor indices
        Returns:
            updated nodes (N, hidden_dim), updated edges (N, k, edge_dim)
        """
        k = knn_indices.shape[1]

        # Gather neighbor node features
        neighbor_nodes = nodes[knn_indices]  # (N, k, hidden)
        src_nodes = nodes.unsqueeze(1).expand(-1, k, -1)  # (N, k, hidden)

        # Messages: (N, k, hidden*2 + edge_dim) -> (N, k, hidden)
        msg_input = torch.cat([src_nodes, neighbor_nodes, edges], dim=-1)
        messages = self.message_mlp(msg_input)

        # Aggregate: sum over neighbors
        agg = messages.sum(dim=1)  # (N, hidden)

        # Update nodes (residual + LayerNorm)
        nodes = self.norm_node(nodes + self.node_mlp(torch.cat([nodes, agg], dim=-1)))

        # Update edges (residual + LayerNorm)
        # Recompute src_nodes after node update
        src_updated = nodes.unsqueeze(1).expand(-1, k, -1)
        neighbor_updated = nodes[knn_indices]
        edge_input = torch.cat([src_updated, neighbor_updated, edges], dim=-1)
        edges = self.norm_edge(edges + self.edge_mlp(edge_input))

        return nodes, edges


class InverseFoldingModel(nn.Module):
    """One-shot encoder for protein inverse folding.

    Encodes backbone structure via message-passing GNN, then predicts
    amino acid identity at all positions simultaneously.
    """

    def __init__(self):
        super().__init__()
        self.node_proj = nn.Linear(NODE_FEAT_DIM, HIDDEN_DIM)
        self.edge_proj = nn.Linear(EDGE_FEAT_DIM, HIDDEN_DIM)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(HIDDEN_DIM, HIDDEN_DIM, DROPOUT)
            for _ in range(NUM_ENCODER_LAYERS)
        ])

        self.output_head = nn.Linear(HIDDEN_DIM, NUM_AMINO_ACIDS)

    def _featurize(self, batch, noise=0.0):
        """Compute features from (possibly noised) coordinates."""
        coords = batch['coords']  # (N, 5, 3)
        knn_indices = batch['knn_indices']  # (N, k)
        batch_idx = batch['batch_idx']
        lengths = batch['lengths']

        if noise > 0:
            coords = coords + torch.randn_like(coords) * noise

        edge_features = compute_edge_features(coords, knn_indices)  # (N, k, 240)
        node_features = compute_node_features(coords[:, :4], batch_idx, lengths)  # (N, 6)

        return node_features, edge_features

    def encode(self, node_features, edge_features, knn_indices):
        """Run encoder on features."""
        nodes = self.node_proj(node_features)   # (N, hidden)
        edges = self.edge_proj(edge_features)   # (N, k, hidden)

        for layer in self.encoder_layers:
            nodes, edges = layer(nodes, edges, knn_indices)

        return nodes

    def forward(self, batch):
        """Training forward pass. Returns cross-entropy loss."""
        node_features, edge_features = self._featurize(batch, noise=BACKBONE_NOISE)
        nodes = self.encode(node_features, edge_features, batch['knn_indices'])

        logits = self.output_head(nodes)  # (N, 20)
        targets = batch['seq']
        mask = batch['mask']

        loss = F.cross_entropy(logits[mask], targets[mask])
        return loss

    def predict_logits(self, batch):
        """Inference: predict logits for all residues. No noise."""
        node_features, edge_features = self._featurize(batch, noise=0.0)
        nodes = self.encode(node_features, edge_features, batch['knn_indices'])
        logits = self.output_head(nodes)  # (N, 20)
        return logits
