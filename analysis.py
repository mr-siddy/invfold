"""Generate progress charts from autoresearch results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# Load results
df = pd.read_csv("results.tsv", sep="\t")
df["val_metric"] = pd.to_numeric(df["val_metric"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

n_total = len(df)
n_keep = len(df[df["status"] == "KEEP"])
n_discard = len(df[df["status"] == "DISCARD"])
n_crash = len(df[df["status"] == "CRASH"]) if "CRASH" in df["status"].values else 0

baseline_val = df.iloc[0]["val_metric"]
best_val = df[df["status"] == "KEEP"]["val_metric"].max()
best_row = df.loc[df[df["status"] == "KEEP"]["val_metric"].idxmax()]
improvement = (best_val - baseline_val) / baseline_val * 100

print(f"Total experiments: {n_total}")
print(f"  Keep: {n_keep}, Discard: {n_discard}, Crash: {n_crash}")
print(f"  Baseline: {baseline_val*100:.2f}% -> Best: {best_val*100:.2f}% ({improvement:+.1f}%)")
print()

valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
disc = valid[valid["status"] == "DISCARD"]
kept = valid[valid["status"] == "KEEP"]

# =====================================================================
# Figure 1: Main progress chart — full detail, every label, exact values
# =====================================================================

fig, ax = plt.subplots(figsize=(22, 14))

# Discarded experiments (gray circles)
ax.scatter(disc.index, disc["val_metric"] * 100,
           c="#cccccc", s=100, alpha=0.7, zorder=2, label="Discarded",
           edgecolors="#999999", linewidths=0.8)

# Kept experiments (green circles, larger)
ax.scatter(kept.index, kept["val_metric"] * 100,
           c="#2ecc71", s=180, zorder=4, label="Kept",
           edgecolors="black", linewidths=1.2)

# Running best step line
kept_vals = valid.loc[valid["status"] == "KEEP", "val_metric"] * 100
running_best = kept_vals.cummax()
extended_idx = list(kept.index) + [len(valid) - 1]
extended_vals = list(running_best) + [running_best.iloc[-1]]
ax.step(extended_idx, extended_vals, where="post", color="#27ae60",
        linewidth=3, alpha=0.8, zorder=3, label="Running best")

# Baseline reference line
ax.axhline(y=baseline_val * 100, color="#3498db", linestyle="--",
           alpha=0.5, linewidth=1.5, label=f"Baseline ({baseline_val*100:.2f}%)")

# ── Annotate EVERY experiment with full description + exact value + delta ──

# Pre-compute label positions to avoid overlap
# Strategy: place labels on alternating sides (left/right) with vertical offsets
# that spread out based on value clustering

all_indices = list(valid.index)
all_vals = [valid.loc[i, "val_metric"] * 100 for i in all_indices]
all_descs = [str(valid.loc[i, "description"]).strip() for i in all_indices]
all_status = [valid.loc[i, "status"] for i in all_indices]
all_deltas = [(valid.loc[i, "val_metric"] - baseline_val) * 100 for i in all_indices]

for i, (idx, val, desc, status, delta) in enumerate(
        zip(all_indices, all_vals, all_descs, all_status, all_deltas)):

    is_kept = status == "KEEP"

    # Full label text: description + exact recovery + delta
    if delta == 0:
        label = f"{desc}\n{val:.2f}% (baseline)"
    else:
        label = f"{desc}\n{val:.2f}% ({delta:+.2f} pp)"

    if is_kept:
        # Kept: green box, arrow, placed to the right-above
        # Stagger vertically for kept points
        if idx == 0:
            ox, oy = 20, 30
        elif idx == 1:
            ox, oy = 20, -45
        else:
            ox, oy = 20, 35

        ax.annotate(label, (idx, val),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=10, color="#1a7a3a", fontweight="bold",
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1",
                              edgecolor="#27ae60", linewidth=1.5, alpha=0.95),
                    arrowprops=dict(arrowstyle="-|>", color="#27ae60",
                                   lw=1.5, connectionstyle="arc3,rad=0.15"))
    else:
        # Discarded: gray text with thin arrow, alternate placement
        # Use a deterministic spread pattern based on index
        side_patterns = [
            (25, -35),   # exp 2: right-below
            (-25, -30),  # exp 3: left-below
            (25, 25),    # exp 4: right-above
            (-25, -35),  # exp 6: left-below
            (25, 30),    # exp 7: right-above
            (-25, 25),   # exp 8: left-above
            (25, -30),   # exp 9: right-below
            (-25, 30),   # exp 10: left-above
        ]
        disc_count = list(disc.index).index(idx)
        ox, oy = side_patterns[disc_count % len(side_patterns)]
        ha = "left" if ox > 0 else "right"

        ax.annotate(label, (idx, val),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=9, color="#666666",
                    ha=ha, va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", linewidth=0.8, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="#cccccc",
                                   lw=0.8, connectionstyle="arc3,rad=0.1"))

ax.set_xlabel("Experiment #", fontsize=13)
ax.set_ylabel("Sequence Recovery (%)", fontsize=13)
ax.set_title(f"Protein Inverse Folding — Autoresearch Progress (300s budget, Apple Silicon MPS)\n"
             f"{n_total} Experiments | {n_keep} Kept | {n_discard} Discarded | "
             f"Baseline: {baseline_val*100:.2f}% | Best: {best_val*100:.2f}% "
             f"(+{(best_val-baseline_val)*100:.2f} pp, +{improvement:.1f}% relative)",
             fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=12, framealpha=0.95,
          edgecolor="#cccccc", fancybox=True)
ax.grid(True, alpha=0.2)
ax.set_xlim(-1.5, n_total + 1)
y_min = valid["val_metric"].min() * 100 - 6
y_max = valid["val_metric"].max() * 100 + 6
ax.set_ylim(y_min, y_max)
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")
plt.close()

# =====================================================================
# Figure 2: Detailed 4-panel analysis
# =====================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle(f"Protein Inverse Folding — Detailed Analysis\n"
             f"{n_total} Experiments | Baseline {baseline_val*100:.1f}% -> Best {best_val*100:.1f}%",
             fontsize=14, fontweight="bold", y=0.98)

# ── Panel 1: Bar chart with experiment names ──
ax = axes[0, 0]
colors = ["#2ecc71" if s == "KEEP" else "#bdc3c7" for s in valid["status"]]
edge_colors = ["#27ae60" if s == "KEEP" else "#95a5a6" for s in valid["status"]]
bars = ax.bar(range(len(valid)), valid["val_metric"] * 100, color=colors,
              edgecolor=edge_colors, linewidth=1.2)

ax.axhline(y=baseline_val * 100, color="#3498db", linestyle="--",
           alpha=0.6, linewidth=1.5, label=f"Baseline ({baseline_val*100:.1f}%)")

# Shortened labels below bars
short_names = []
for desc in valid["description"]:
    d = str(desc).strip()
    # Shorten common prefixes
    d = d.replace("precompute features + disable noise", "precompute feat.")
    d = d.replace("smaller batches (5K) + higher LR (2e-3)", "small batch+LR")
    d = d.replace("increase encoder layers 3->5", "5 enc. layers")
    d = d.replace("wider model (192-dim) + fewer layers (2)", "192-dim/2-layer")
    d = d.replace("remove dropout (underfitting regime)", "no dropout")
    d = d.replace("LR 2e-3 + no warmup", "LR 2e-3")
    d = d.replace("add input skip connection", "skip connect")
    d = d.replace("replace ReLU with GELU", "GELU")
    d = d.replace("remove edge updates for speed", "no edge upd.")
    d = d.replace("mean aggregation instead of sum", "mean agg.")
    short_names.append(d)

ax.set_xticks(range(len(valid)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Recovery (%)")
ax.set_title("All Experiments")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15, axis="y")

# ── Panel 2: Delta vs baseline ──
ax = axes[0, 1]
deltas = (valid["val_metric"] - baseline_val) * 100
colors_delta = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]
ax.bar(range(len(valid)), deltas, color=colors_delta,
       edgecolor="white", linewidth=0.5)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xticks(range(len(valid)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Delta vs Baseline (pp)")
ax.set_title("Improvement Over Baseline")
ax.grid(True, alpha=0.15, axis="y")

# ── Panel 3: Memory usage ──
ax = axes[1, 0]
mem_colors = ["#2ecc71" if s == "KEEP" else "#bdc3c7" for s in valid["status"]]
ax.bar(range(len(valid)), valid["memory_gb"], color=mem_colors,
       edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(valid)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Peak Memory (GB)")
ax.set_title("Memory Usage Per Experiment")
ax.grid(True, alpha=0.15, axis="y")

# ── Panel 4: Summary table ──
ax = axes[1, 1]
ax.axis("off")

# Build a proper table
kept_df = df[df["status"] == "KEEP"].copy()
kept_df["delta"] = (kept_df["val_metric"] - baseline_val) * 100

summary_text = (
    f"{'SUMMARY':^44}\n"
    f"{'=' * 44}\n\n"
    f"  Baseline recovery:    {baseline_val*100:.2f}%\n"
    f"  Best recovery:        {best_val*100:.2f}%\n"
    f"  Improvement:          +{(best_val-baseline_val)*100:.2f} pp ({improvement:+.1f}% rel.)\n\n"
    f"  Best experiment:\n"
    f"    {best_row['description']}\n\n"
    f"{'STATISTICS':^44}\n"
    f"{'=' * 44}\n\n"
    f"  Total experiments:    {n_total}\n"
    f"  Kept:                 {n_keep}\n"
    f"  Discarded:            {n_discard}\n"
    f"  Crashed:              {n_crash}\n"
    f"  Keep rate:            {n_keep}/{n_total} = {n_keep/n_total*100:.0f}%\n\n"
    f"{'KEPT IMPROVEMENTS':^44}\n"
    f"{'=' * 44}\n\n"
)
for _, row in kept_df.iterrows():
    marker = " *" if row["val_metric"] == best_val else "  "
    summary_text += f" {marker} {row['val_metric']*100:.2f}%  {row['description']}\n"

summary_text += (
    f"\n{'BENCHMARKS':^44}\n"
    f"{'=' * 44}\n\n"
    f"  ProteinMPNN:          52.4%\n"
    f"  PiFold:               51.7%\n"
    f"  SurfFold:             64.2%"
)

ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa",
                  edgecolor="#dee2e6", linewidth=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("analysis.png", dpi=150, bbox_inches="tight")
print("Saved analysis.png")
plt.close()

# =====================================================================
# Print results table
# =====================================================================

print("\n" + "=" * 80)
print("EXPERIMENT LOG")
print("=" * 80)
print(f"{'#':>3}  {'Status':>8}  {'Recovery':>10}  {'Delta':>8}  {'Mem(GB)':>8}  Description")
print("-" * 80)
for i, row in df.iterrows():
    status = row["status"].upper()
    marker = ">>>" if status == "KEEP" else "   "
    delta = (row["val_metric"] - baseline_val) * 100
    print(f"{i:3d}  {status:>8}  {row['val_metric']*100:9.2f}%  {delta:+7.2f}  {row['memory_gb']:7.1f}  {marker} {row['description']}")

print("-" * 80)
print(f"Baseline: {baseline_val*100:.2f}% -> Best: {best_val*100:.2f}% "
      f"(+{(best_val-baseline_val)*100:.2f} pp, {improvement:+.1f}% relative)")
