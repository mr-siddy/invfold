"""Visualize the protein inverse folding dataset."""

import os
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

CACHE_DIR = os.path.expanduser("~/.cache/auto-bio/invfold")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
PROC_DIR = os.path.join(CACHE_DIR, "processed")

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_NAMES = {
    'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu', 'F': 'Phe',
    'G': 'Gly', 'H': 'His', 'I': 'Ile', 'K': 'Lys', 'L': 'Leu',
    'M': 'Met', 'N': 'Asn', 'P': 'Pro', 'Q': 'Gln', 'R': 'Arg',
    'S': 'Ser', 'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr',
}
AA_CATEGORIES = {
    'Hydrophobic': list('AILMFWVP'),
    'Polar': list('STNQYC'),
    'Positive': list('RHK'),
    'Negative': list('DE'),
    'Special': list('G'),
}
CAT_COLORS = {
    'Hydrophobic': '#e74c3c',
    'Polar': '#3498db',
    'Positive': '#2ecc71',
    'Negative': '#e67e22',
    'Special': '#9b59b6',
}

# ── Load splits ──
with open(os.path.join(RAW_DIR, "chain_set_splits.json")) as f:
    splits = json.load(f)

# ── Load all cached proteins (sample if too many) ──
print("Loading cached proteins...")
all_files = [f for f in os.listdir(PROC_DIR) if f.endswith(".pt")]
all_names = [f.replace(".pt", "") for f in all_files]

# Load all for statistics
lengths = []
aa_counts = np.zeros(20, dtype=np.int64)
valid_fracs = []
ca_ca_dists_all = []
proteins = {}

for i, fname in enumerate(all_files):
    pt = torch.load(os.path.join(PROC_DIR, fname), weights_only=False)
    L = pt['length']
    lengths.append(L)
    seq = pt['seq'].numpy()
    mask = pt['mask'].numpy()
    for aa_idx in seq:
        aa_counts[aa_idx] += 1
    valid_fracs.append(mask.sum() / L)
    proteins[fname.replace(".pt", "")] = pt

    if i % 5000 == 0:
        print(f"  Loaded {i}/{len(all_files)}...")

print(f"  Loaded {len(all_files)} proteins")

lengths = np.array(lengths)
valid_fracs = np.array(valid_fracs)

# ── Assign split labels ──
split_labels = {}
for name in all_names:
    if name in splits.get("train", []):
        split_labels[name] = "train"
    elif name in splits.get("validation", []):
        split_labels[name] = "val"
    elif name in splits.get("test", []):
        split_labels[name] = "test"
    else:
        split_labels[name] = "unknown"

# =====================================================================
# Figure 1: Dataset Overview (2x2)
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Protein Inverse Folding — Dataset Overview\n"
             f"CATH 4.2 | {len(all_files):,} Proteins | "
             f"Train: {len(splits['train']):,} | Val: {len(splits['validation']):,} | "
             f"Test: {len(splits['test']):,}",
             fontsize=15, fontweight="bold")

# ── Panel 1: Protein length distribution ──
ax = axes[0, 0]
ax.hist(lengths, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
ax.axvline(x=np.median(lengths), color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Median: {np.median(lengths):.0f} residues")
ax.axvline(x=np.mean(lengths), color="#f39c12", linestyle="--", linewidth=2,
           label=f"Mean: {np.mean(lengths):.0f} residues")
ax.set_xlabel("Protein Length (residues)")
ax.set_ylabel("Count")
ax.set_title("Protein Length Distribution")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)

# Add stats text
stats_text = (f"Min: {lengths.min()}\nMax: {lengths.max()}\n"
              f"Std: {lengths.std():.1f}\nTotal residues: {lengths.sum():,}")
ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
        fontsize=9, va="top", ha="right", fontfamily="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9, pad=5))

# ── Panel 2: Amino acid frequency ──
ax = axes[0, 1]
total_aa = aa_counts.sum()
aa_freqs = aa_counts / total_aa * 100

# Color by category
bar_colors = []
for aa in AA_VOCAB:
    for cat, aas in AA_CATEGORIES.items():
        if aa in aas:
            bar_colors.append(CAT_COLORS[cat])
            break

bars = ax.bar(range(20), aa_freqs, color=bar_colors, edgecolor="white", linewidth=0.5)

# Add frequency labels on bars
for i, (freq, count) in enumerate(zip(aa_freqs, aa_counts)):
    ax.text(i, freq + 0.2, f"{freq:.1f}%", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(range(20))
aa_labels = [f"{aa}\n{AA_NAMES[aa]}" for aa in AA_VOCAB]
ax.set_xticklabels(aa_labels, fontsize=8)
ax.set_ylabel("Frequency (%)")
ax.set_title("Amino Acid Distribution")
ax.grid(True, alpha=0.15, axis="y")

# Legend for categories
legend_patches = [mpatches.Patch(color=CAT_COLORS[cat], label=f"{cat} ({len(aas)})")
                  for cat, aas in AA_CATEGORIES.items()]
ax.legend(handles=legend_patches, fontsize=8, loc="upper right", ncol=2)

# ── Panel 3: Split sizes + length distributions per split ──
ax = axes[1, 0]
train_lengths = [proteins[n]['length'] for n in splits['train'] if n in proteins]
val_lengths = [proteins[n]['length'] for n in splits['validation'] if n in proteins]
test_lengths = [proteins[n]['length'] for n in splits['test'] if n in proteins]

bins = np.linspace(0, 500, 40)
ax.hist(train_lengths, bins=bins, alpha=0.6, color="#2ecc71", label=f"Train ({len(train_lengths):,})", edgecolor="white")
ax.hist(val_lengths, bins=bins, alpha=0.6, color="#3498db", label=f"Val ({len(val_lengths):,})", edgecolor="white")
ax.hist(test_lengths, bins=bins, alpha=0.6, color="#e74c3c", label=f"Test ({len(test_lengths):,})", edgecolor="white")
ax.set_xlabel("Protein Length (residues)")
ax.set_ylabel("Count")
ax.set_title("Length Distribution by Split")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)

# ── Panel 4: Valid residue fraction distribution ──
ax = axes[1, 1]
ax.hist(valid_fracs * 100, bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
ax.set_xlabel("Valid Residues (%)")
ax.set_ylabel("Count")
ax.set_title("Backbone Completeness per Protein")
ax.axvline(x=np.median(valid_fracs) * 100, color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Median: {np.median(valid_fracs)*100:.1f}%")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)

completeness_text = (f"100% complete: {(valid_fracs == 1.0).sum():,} ({(valid_fracs == 1.0).mean()*100:.1f}%)\n"
                     f">95% complete: {(valid_fracs > 0.95).sum():,} ({(valid_fracs > 0.95).mean()*100:.1f}%)\n"
                     f"Mean: {valid_fracs.mean()*100:.1f}%")
ax.text(0.03, 0.97, completeness_text, transform=ax.transAxes,
        fontsize=9, va="top", ha="left", fontfamily="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9, pad=5))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("dataset_overview.png", dpi=150, bbox_inches="tight")
print("Saved dataset_overview.png")
plt.close()

# =====================================================================
# Figure 2: Example protein backbone structures (2x3 grid)
# =====================================================================

# Pick 6 proteins of varying sizes from the test set
test_names = [n for n in splits['test'] if n in proteins]
test_by_len = sorted(test_names, key=lambda n: proteins[n]['length'])

# Pick small, medium, large examples
pick_indices = [
    len(test_by_len) // 10,       # small
    len(test_by_len) // 4,        # small-medium
    len(test_by_len) // 2,        # medium
    int(len(test_by_len) * 0.6),  # medium-large
    int(len(test_by_len) * 0.75), # large
    int(len(test_by_len) * 0.9),  # very large
]
example_names = [test_by_len[i] for i in pick_indices]

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
fig.suptitle("Example Protein Backbone Structures (CA trace, PCA-projected to 2D)\n"
             "Color: residue position (N-terminus = blue, C-terminus = red)",
             fontsize=14, fontweight="bold")

for ax_idx, name in enumerate(example_names):
    ax = axes[ax_idx // 3, ax_idx % 3]
    pt = proteins[name]
    coords = pt['coords'].numpy()  # (L, 5, 3)
    ca = coords[:, 1, :]  # CA atoms: (L, 3)
    seq = pt['seq'].numpy()
    mask = pt['mask'].numpy()
    L = pt['length']

    # PCA projection to 2D
    ca_valid = ca[mask]
    if len(ca_valid) < 3:
        ax.set_title(f"{name} (too few valid residues)")
        continue

    ca_centered = ca_valid - ca_valid.mean(axis=0)
    _, _, Vt = np.linalg.svd(ca_centered, full_matrices=False)
    proj = ca_centered @ Vt[:2].T  # (L_valid, 2)

    # Color by position (N-term to C-term)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(proj)))

    # Draw backbone as connected line segments
    points = proj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors[:-1], linewidths=1.5, alpha=0.7)
    ax.add_collection(lc)

    # Draw CA atoms as dots
    ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=15, zorder=3, edgecolors="none")

    # Mark N and C terminus
    ax.scatter(proj[0, 0], proj[0, 1], c="blue", s=100, zorder=5,
               edgecolors="black", linewidths=1, marker="^", label="N-term")
    ax.scatter(proj[-1, 0], proj[-1, 1], c="red", s=100, zorder=5,
               edgecolors="black", linewidths=1, marker="v", label="C-term")

    # Sequence composition
    aa_comp = np.bincount(seq, minlength=20)
    top3 = np.argsort(aa_comp)[-3:][::-1]
    top3_str = ", ".join([f"{AA_VOCAB[i]}({aa_comp[i]})" for i in top3])

    ax.set_title(f"{name}  |  {L} residues  |  {mask.sum()} valid", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Top AAs: {top3_str}", fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.1)

    # Remove axis numbers (PCA units are arbitrary)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("dataset_proteins.png", dpi=150, bbox_inches="tight")
print("Saved dataset_proteins.png")
plt.close()

# =====================================================================
# Figure 3: k-NN graph and featurization details
# =====================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Graph Construction and Featurization Details",
             fontsize=14, fontweight="bold")

# Pick one medium-sized protein for detailed visualization
example_name = test_by_len[len(test_by_len) // 2]
pt = proteins[example_name]
coords = pt['coords'].numpy()
ca = coords[:, 1, :]
knn = pt['knn_indices'].numpy()
L = pt['length']

# ── Panel 1: k-NN graph visualization (2D projection) ──
ax = axes[0]
ca_centered = ca - ca.mean(axis=0)
_, _, Vt = np.linalg.svd(ca_centered, full_matrices=False)
proj = ca_centered @ Vt[:2].T

# Draw a subset of edges (all edges would be too dense)
# Show edges for every 5th residue
for i in range(0, L, 5):
    for j_idx in range(min(8, knn.shape[1])):  # show 8 of 48 neighbors
        j = knn[i, j_idx]
        if j < L:
            ax.plot([proj[i, 0], proj[j, 0]], [proj[i, 1], proj[j, 1]],
                    color="#cccccc", linewidth=0.3, alpha=0.5, zorder=1)

# Draw all CA atoms
colors = plt.cm.viridis(np.linspace(0, 1, L))
ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=12, zorder=3, edgecolors="none")
ax.set_title(f"k-NN Graph (k=48)\n{example_name} | {L} residues\nShowing 8/48 edges for every 5th residue",
             fontsize=11)
ax.set_aspect("equal")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, alpha=0.1)

# ── Panel 2: CA-CA distance distribution for this protein ──
ax = axes[1]
ca_dists_flat = []
for i in range(L):
    for j_idx in range(knn.shape[1]):
        j = knn[i, j_idx]
        if j < L:
            d = np.linalg.norm(ca[i] - ca[j])
            ca_dists_flat.append(d)

ca_dists_flat = np.array(ca_dists_flat)
ax.hist(ca_dists_flat, bins=60, color="#e67e22", edgecolor="white", alpha=0.85)
ax.axvline(x=np.median(ca_dists_flat), color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Median: {np.median(ca_dists_flat):.1f} A")
ax.axvline(x=3.8, color="#2ecc71", linestyle=":", linewidth=2,
           label="Sequential CA-CA: 3.8 A")
ax.set_xlabel("CA-CA Distance (Angstroms)")
ax.set_ylabel("Count")
ax.set_title(f"CA-CA Distance Distribution\nk=48 neighbors | {example_name}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

dist_stats = (f"Min: {ca_dists_flat.min():.1f} A\n"
              f"Max: {ca_dists_flat.max():.1f} A\n"
              f"Mean: {ca_dists_flat.mean():.1f} A\n"
              f"Total edges: {len(ca_dists_flat):,}")
ax.text(0.97, 0.97, dist_stats, transform=ax.transAxes,
        fontsize=9, va="top", ha="right", fontfamily="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.9, pad=5))

# ── Panel 3: Gaussian RBF visualization ──
ax = axes[2]
D_min, D_max, num_rbf = 0.0, 20.0, 16
D_mu = np.linspace(D_min, D_max, num_rbf)
D_sigma = (D_max - D_min) / num_rbf

d_range = np.linspace(0, 22, 500)
rbf_colors = plt.cm.tab20(np.linspace(0, 1, num_rbf))

for k in range(num_rbf):
    rbf_vals = np.exp(-((d_range - D_mu[k]) ** 2) / (2 * D_sigma ** 2))
    ax.plot(d_range, rbf_vals, color=rbf_colors[k], linewidth=1.2, alpha=0.8)
    # Mark the center
    ax.scatter([D_mu[k]], [1.0], color=rbf_colors[k], s=25, zorder=5, edgecolors="black", linewidths=0.5)

ax.axvline(x=3.8, color="#2ecc71", linestyle=":", linewidth=2, alpha=0.5,
           label="Sequential CA-CA (3.8 A)")
ax.axvline(x=D_max, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5,
           label=f"RBF cutoff ({D_max} A)")

ax.set_xlabel("Distance (Angstroms)")
ax.set_ylabel("RBF Response")
ax.set_title(f"Gaussian RBF Encoding\n{num_rbf} bins | D_min={D_min} | D_max={D_max} A\n"
             f"sigma={D_sigma:.2f} A | 15 atom pairs x 16 bins = 240-dim edges")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)
ax.set_xlim(-0.5, 22)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("dataset_features.png", dpi=150, bbox_inches="tight")
print("Saved dataset_features.png")
plt.close()

print("\nDone! Generated 3 visualization files:")
print("  dataset_overview.png  — length distribution, AA frequencies, splits, completeness")
print("  dataset_proteins.png  — 6 example protein backbones (PCA-projected CA traces)")
print("  dataset_features.png  — k-NN graph, CA-CA distances, Gaussian RBF encoding")
