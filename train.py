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
    TIME_BUDGET as _TIME_BUDGET, SEED, NUM_AMINO_ACIDS, MAX_NEIGHBORS, NUM_RBF,
    compute_edge_features, compute_node_features,
    make_dataloader, evaluate_recovery,
)

# Override time budget for demo (set to 300 for full runs)
TIME_BUDGET = 60

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


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloaders
# ---------------------------------------------------------------------------

t_start = time.time()

train_loader = make_dataloader('train', BATCH_SIZE_TOKENS)
val_loader = make_dataloader('val', BATCH_SIZE_TOKENS)
test_loader = make_dataloader('test', BATCH_SIZE_TOKENS)

model = InverseFoldingModel().to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print(f"Device: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

total_training_time = 0
step = 0
epoch = 0

while True:
    model.train()
    epoch_loss = 0
    epoch_residues = 0

    for batch in train_loader:
        t0 = time.time()

        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        total_training_time += dt

        epoch_loss += loss.item() * batch['mask'].sum().item()
        epoch_residues += batch['mask'].sum().item()

        if total_training_time >= TIME_BUDGET:
            break

    epoch += 1

    # LR warmup
    if epoch <= WARMUP_EPOCHS:
        lr_scale = epoch / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg['lr'] = LR * lr_scale

    avg_loss = epoch_loss / max(epoch_residues, 1)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {total_training_time:.0f}s")

    if total_training_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    recovery, perplexity = evaluate_recovery(model, test_loader, device)

t_end = time.time()

# Memory reporting
if device.type == "mps":
    peak_vram_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
elif device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_metric:       {recovery:.6f}")
print(f"val_perplexity:   {perplexity:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_params:       {num_params}")
