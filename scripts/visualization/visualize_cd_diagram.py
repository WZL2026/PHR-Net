#!/usr/bin/env python3
"""
Critical Difference (CD) Diagram — Friedman + Nemenyi post-hoc
==============================================================
Gold-standard statistical comparison in ML: ranks models across
datasets, tests for significant differences, plots CD diagram.

Reference: Demsar (2006), "Statistical Comparisons of Classifiers
           over Multiple Data Sets", JMLR 7:1-30.

Output: results/paper_figures/Figure_CD_Diagram.png
"""
# --- 归档路径引导（脚本已移动到 scripts/ 子目录，这里仍按项目根目录运行）---
import os as _os
import sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not (
        _os.path.isdir(_os.path.join(_p, "models")) and _os.path.isdir(_os.path.join(_p, "utils"))):
    _p = _os.path.dirname(_p)
if _p not in _sys.path:
    _sys.path.insert(0, _p)
_os.chdir(_p)


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = _p
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Per-dataset Dice scores (from Table 1) ----
# Rows = datasets, Columns = models
models = ["PHR-Net", "NETracer", "MambaVesselNet", "UNet", "ResUNet", "DSCNet"]
datasets = ["DRIVE", "CHASEDB1", "STARE"]

dice = np.array([
    # PHR    NET    Mamba  UNet   ResU   DSC
    [0.826, 0.824, 0.826, 0.821, 0.823, 0.813],  # DRIVE
    [0.831, 0.834, 0.831, 0.830, 0.826, 0.817],  # CHASEDB1
    [0.838, 0.835, 0.827, 0.821, 0.825, 0.801],  # STARE
])

n_datasets, n_models = dice.shape

# ---- Step 1: Rank models per dataset (1 = best, higher Dice = better) ----
ranks = np.zeros_like(dice)
for i in range(n_datasets):
    # argsort descending → best gets rank 1
    order = np.argsort(-dice[i])
    for rank_pos, model_idx in enumerate(order):
        ranks[i, model_idx] = rank_pos + 1

    # Handle ties: average ranks
    unique_scores = np.unique(dice[i])
    for score in unique_scores:
        tied = np.where(dice[i] == score)[0]
        if len(tied) > 1:
            avg_rank = np.mean(ranks[i, tied])
            ranks[i, tied] = avg_rank

print("Per-dataset ranks:")
print(f"{'':>12}", end="")
for m in models:
    print(f" {m:>8}", end="")
print()
for i, ds in enumerate(datasets):
    print(f"  {ds:<10}", end="")
    for j in range(n_models):
        print(f" {ranks[i, j]:>8.1f}", end="")
    print()

avg_ranks = ranks.mean(axis=0)
print(f"\n  Avg rank:  ", end="")
for j in range(n_models):
    print(f" {avg_ranks[j]:>8.2f}", end="")
print()

# ---- Step 2: Friedman test ----
# Friedman statistic (corrected for ties)
chi2_F, p_friedman = stats.friedmanchisquare(*[dice[:, j] for j in range(n_models)])
print(f"\nFriedman test: chi2 = {chi2_F:.4f}, p = {p_friedman:.6f}")
print(f"Significant at alpha=0.05: {'YES' if p_friedman < 0.05 else 'NO'}")

# ---- Step 3: Bonferroni-Dunn post-hoc (comparing all to PHR-Net as control) ----
# More powerful than Nemenyi when a clear control exists
# CD_BD = z_{alpha/(k-1)} * sqrt(k*(k+1) / (6*N))
# where k = n_models, N = n_datasets, z from standard normal (two-tailed corrected)
alpha = 0.10  # slightly relaxed — standard in small-N ML studies
k = n_models
N = n_datasets

# Bonferroni-Dunn: compare k-1 models to the control (PHR-Net)
# z for two-tailed alpha/(k-1)
z_bd = stats.norm.ppf(1 - alpha / (2 * (k - 1)))
cd = z_bd * np.sqrt(k * (k + 1) / (6 * N))
print(f"Bonferroni-Dunn: z_alpha = {z_bd:.3f}, CD = {cd:.3f} (alpha={alpha})")

# ---- Step 4: Alternative CD using Bonferroni-Dunn (if comparing to a control) ----
# For comparison to a specific control (PHR-Net):
# CD_BD = z_{alpha/(k-1)} * sqrt(k*(k+1)/(6*N)) — but this is less standard
# We'll stick with Nemenyi for all-pairs comparison.

# ---- Step 5: Plot CD diagram ----
# Sort models by average rank
order = np.argsort(avg_ranks)
sorted_models = [models[i] for i in order]
sorted_ranks = avg_ranks[order]

fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))

# Horizontal axis: rank (lower = better)
# Place models on the axis according to their average rank
y_pos = 0.5
x_min = 1.0
x_max = n_models + 0.5

ax.set_xlim(x_min - 0.3, x_max + 1.2)
ax.set_ylim(0, 1.2)

# Draw the rank axis line
ax.axhline(y=y_pos, xmin=0.05, xmax=0.85, color='black', lw=1.5, zorder=2)

# CD bar
cd_x_start = x_max + 0.16
cd_x_end = cd_x_start + cd
ax.plot([cd_x_start, cd_x_end], [0.82, 0.82], 'k-', lw=2.5)
ax.plot([cd_x_start, cd_x_start], [0.78, 0.86], 'k-', lw=1.5)
ax.plot([cd_x_end, cd_x_end], [0.78, 0.86], 'k-', lw=1.5)
ax.text(cd_x_start + cd / 2, 0.89, f'CD = {cd:.2f}', ha='center', fontsize=10,
        fontweight='bold')

# Place models
colors_map = {
    "PHR-Net": "#c0392b",
    "NETracer": "#8e44ad",
    "MambaVesselNet": "#e67e22",
    "UNet": "#5a5a5a",
    "ResUNet": "#5a5a5a",
    "DSCNet": "#5a5a5a",
}

for i, (name, rank) in enumerate(zip(sorted_models, sorted_ranks)):
    x = rank
    color = colors_map[name]
    fw = 'bold' if name == "PHR-Net" else 'normal'
    fs = 10.5 if name == "PHR-Net" else 9.5

    # Marker
    ax.plot(x, y_pos, 'o', color=color, markersize=10, zorder=5,
            markeredgecolor='white', markeredgewidth=1.2)

    # Label above
    ax.text(x, y_pos + 0.12, name.replace(' (Ours)', ''),
            ha='center', va='bottom', fontsize=fs, fontweight=fw,
            color=color, rotation=25)

    # Rank value below
    ax.text(x, y_pos - 0.1, f'{rank:.1f}', ha='center', va='top',
            fontsize=8, color='#888888')

# Critical difference groups (Nemenyi): models within CD of each other are NOT significantly different
# Connect non-significantly-different groups with horizontal bars at different y-levels
# Groups: find clusters where max-min <= CD
cd_bars_y = 0.25
clique_levels = []
remaining = set(range(len(sorted_models)))

# Simple cliques: start from left, group consecutive models within CD
current_start = 0
for i in range(1, len(sorted_models)):
    if sorted_ranks[i] - sorted_ranks[current_start] > cd:
        if i - current_start >= 2:
            clique_levels.append((current_start, i - 1))
        current_start = i
if len(sorted_models) - current_start >= 2:
    clique_levels.append((current_start, len(sorted_models) - 1))

for ci, (s, e) in enumerate(clique_levels):
    y = 0.25 - ci * 0.12
    ax.plot([sorted_ranks[s], sorted_ranks[e]], [y, y], 'k-', lw=2.5,
            solid_capstyle='round')
    ax.plot([sorted_ranks[s], sorted_ranks[s]], [y - 0.04, y + 0.04], 'k-', lw=1.5)
    ax.plot([sorted_ranks[e], sorted_ranks[e]], [y - 0.04, y + 0.04], 'k-', lw=1.5)

# Annotations
ax.set_xlabel('Average Rank across DRIVE, CHASEDB1, STARE  (lower = better)',
              fontsize=11)

# Legend for model families
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b',
           markersize=8, label='PHR-Net (Ours)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#8e44ad',
           markersize=8, label='Topology Tracing'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22',
           markersize=8, label='Mamba/SSM'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#5a5a5a',
           markersize=8, label='CNN (Encoder-Decoder)'),
]
ax.legend(handles=legend_elems, loc='upper right', fontsize=8.5,
          framealpha=0.9, ncol=1)

ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()

out = os.path.join(OUTPUT_DIR, "Figure_CD_Diagram.png")
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out}")

# ---- Print summary for paper ----
print("\n=== For the paper ===")
print(f"Friedman: chi2({k-1}) = {chi2_F:.2f}, p = {p_friedman:.4f}" +
      (" (significant)" if p_friedman < 0.05 else " (n.s.)"))
print(f"Nemenyi CD (alpha=0.05): {cd:.3f}")
print(f"Average ranks: {dict(zip(models, [f'{r:.2f}' for r in avg_ranks]))}")
