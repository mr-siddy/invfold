"""Generate progress charts from autoresearch results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

# Load results
df = pd.read_csv("results.tsv", sep="\t")
df["val_metric"] = pd.to_numeric(df["val_metric"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

n_total = len(df)
n_keep = len(df[df["status"] == "KEEP"])
n_discard = len(df[df["status"] == "DISCARD"])
n_crash = len(df[df["status"] == "CRASH"]) if "CRASH" in df["status"].values else 0

print(f"Total experiments: {n_total}")
print(f"  Keep: {n_keep}, Discard: {n_discard}, Crash: {n_crash}")
print()

# ─────────────────────────────────────────────────────────────
# Figure 1: Progress chart (val_metric over experiments)
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Protein Inverse Folding — Autoresearch (300s budget)\n"
             f"{n_total} Experiments | {n_keep} Kept | {n_discard} Discarded",
             fontsize=14, fontweight="bold")

# ── Panel 1: Recovery over experiments ──
ax = axes[0, 0]
valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)

# Discarded experiments (gray dots)
disc = valid[valid["status"] == "DISCARD"]
ax.scatter(disc.index, disc["val_metric"] * 100,
           c="#cccccc", s=60, alpha=0.7, zorder=2, label="Discarded",
           edgecolors="#999999", linewidths=0.5)

# Kept experiments (green dots)
kept = valid[valid["status"] == "KEEP"]
ax.scatter(kept.index, kept["val_metric"] * 100,
           c="#2ecc71", s=100, zorder=4, label="Kept",
           edgecolors="black", linewidths=0.8)

# Running best line
kept_vals = valid.loc[valid["status"] == "KEEP", "val_metric"] * 100
running_best = kept_vals.cummax()
ax.step(kept.index, running_best, where="post", color="#27ae60",
        linewidth=2.5, alpha=0.8, zorder=3, label="Running best")

# Annotate kept experiments
for idx in kept.index:
    desc = str(valid.loc[idx, "description"]).strip()
    val = valid.loc[idx, "val_metric"] * 100
    if len(desc) > 35:
        desc = desc[:32] + "..."
    ax.annotate(desc, (idx, val),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="#1a7a3a", alpha=0.9,
                rotation=15, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color="#1a7a3a", alpha=0.3))

# Annotate worst discards too
for idx in disc.index:
    val = valid.loc[idx, "val_metric"] * 100
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 30:
        desc = desc[:27] + "..."
    ax.annotate(desc, (idx, val),
                textcoords="offset points", xytext=(8, -12),
                fontsize=7, color="#999999", alpha=0.7,
                rotation=-10, ha="left", va="top")

ax.set_xlabel("Experiment #", fontsize=11)
ax.set_ylabel("Sequence Recovery (%)", fontsize=11)
ax.set_title("Recovery Over Experiments (higher is better)", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.2)
ax.set_xlim(-0.5, n_total - 0.5)

# ── Panel 2: Bar chart of all experiments ──
ax = axes[0, 1]
colors = ["#2ecc71" if s == "KEEP" else "#e74c3c" if s == "CRASH" else "#95a5a6"
          for s in valid["status"]]
bars = ax.bar(range(len(valid)), valid["val_metric"] * 100, color=colors,
              edgecolor="white", linewidth=0.5)

# Add experiment labels
for i, row in valid.iterrows():
    desc = str(row["description"]).strip()
    if len(desc) > 20:
        desc = desc[:17] + "..."
    ax.text(i, row["val_metric"] * 100 + 0.5, desc,
            ha="center", va="bottom", fontsize=6.5, rotation=45)

ax.axhline(y=valid.iloc[0]["val_metric"] * 100, color="blue",
           linestyle="--", alpha=0.4, label="Baseline")
ax.set_xlabel("Experiment #", fontsize=11)
ax.set_ylabel("Recovery (%)", fontsize=11)
ax.set_title("All Experiments", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="y")

# ── Panel 3: Memory usage ──
ax = axes[1, 0]
mem_colors = ["#2ecc71" if s == "KEEP" else "#95a5a6" for s in valid["status"]]
ax.bar(range(len(valid)), valid["memory_gb"], color=mem_colors,
       edgecolor="white", linewidth=0.5)
ax.set_xlabel("Experiment #", fontsize=11)
ax.set_ylabel("Peak Memory (GB)", fontsize=11)
ax.set_title("Memory Usage Per Experiment", fontsize=11)
ax.grid(True, alpha=0.2, axis="y")

# ── Panel 4: Summary stats ──
ax = axes[1, 1]
ax.axis("off")

baseline_val = df.iloc[0]["val_metric"]
best_val = df[df["status"].str.upper() == "KEEP"]["val_metric"].max()
best_row = df.loc[df[df["status"].str.upper() == "KEEP"]["val_metric"].idxmax()]
improvement = (best_val - baseline_val) / baseline_val * 100

summary_text = (
    f"Summary\n"
    f"{'─' * 40}\n\n"
    f"Baseline recovery:    {baseline_val*100:.2f}%\n"
    f"Best recovery:        {best_val*100:.2f}%\n"
    f"Improvement:          {improvement:+.2f}%\n\n"
    f"Best experiment:\n"
    f"  {best_row['description']}\n\n"
    f"Total experiments:    {n_total}\n"
    f"  Kept:               {n_keep}\n"
    f"  Discarded:          {n_discard}\n"
    f"  Crashed:            {n_crash}\n\n"
    f"Keep rate:            {n_keep}/{n_keep+n_discard} = "
    f"{n_keep/(n_keep+n_discard)*100:.0f}%\n\n"
    f"Known benchmarks:\n"
    f"  ProteinMPNN:        52.4%\n"
    f"  PiFold:             51.7%\n"
    f"  SurfFold:           64.2%"
)

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))

plt.tight_layout()
plt.savefig("progress.png", dpi=150, bbox_inches="tight")
print("Saved progress.png")

# ─────────────────────────────────────────────────────────────
# Print results table
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("EXPERIMENT LOG")
print("=" * 80)
print(f"{'#':>3}  {'Status':>8}  {'Recovery':>10}  {'Mem(GB)':>8}  Description")
print("-" * 80)
for i, row in df.iterrows():
    status = row["status"].upper()
    marker = ">>>" if status == "KEEP" else "   "
    print(f"{i:3d}  {status:>8}  {row['val_metric']*100:9.2f}%  {row['memory_gb']:7.1f}  {marker} {row['description']}")

print("-" * 80)
print(f"Baseline: {baseline_val*100:.2f}% -> Best: {best_val*100:.2f}% ({improvement:+.2f}% relative)")
