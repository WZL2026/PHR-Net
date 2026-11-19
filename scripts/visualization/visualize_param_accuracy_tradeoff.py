#!/usr/bin/env python3
"""
Figure 1(b): Parameter-Accuracy Trade-off in Retinal Vessel Segmentation
Clean, academic-style scatter plot showing PHR-Net's unique position.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = _p
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Data: (name, params_M, dice_stare, group) ----
# STARE 5-fold CV results from Table 1(c)
models = [
    ("UNet",             31.04,  0.821,  "CNN"),
    ("ResUNet",          13.04,  0.825,  "CNN"),
    ("DSCNet",            2.10,  0.801,  "CNN"),
    ("MambaVesselNet",  117.22,  0.827,  "Mamba/SSM"),
    ("NETracer",         37.67,  0.835,  "Tracing"),
    ("PHR-Net (Ours)",    2.62,  0.838,  "Ours"),
]

colors = {"CNN": "#5a5a5a", "Mamba/SSM": "#e67e22", "Tracing": "#8e44ad", "Ours": "#c0392b"}
markers = {"CNN": "o", "Mamba/SSM": "o", "Tracing": "o", "Ours": "o"}
sizes =   {"CNN": 8, "Mamba/SSM": 8, "Tracing": 8, "Ours": 12}

# ---- Plot ----
fig, ax = plt.subplots(1, 1, figsize=(9, 6))

# Light background zones
ax.axvspan(0.5, 5, alpha=0.06, color='#27ae60', zorder=0)
ax.axvspan(10, 200, alpha=0.04, color='#c0392b', zorder=0)

# Model points
for name, params, dice, group in models:
    ax.plot(params, dice,
            marker=markers[group], color=colors[group],
            markersize=sizes[group], markeredgewidth=0,
            zorder=5,
            label=group if group not in [lbl for _, _, _, lbl in models[:models.index((name,params,dice,group))]] else "")

# Annotations
offsets = {
    "UNet":            (0, -14),
    "ResUNet":         (0, -14),
    "DSCNet":          (0, -14),
    "MambaVesselNet":  (0, -14),
    "NETracer":        (0, -14),
    "PHR-Net (Ours)":  (0, -16),
}
for name, params, dice, group in models:
    dx, dy = offsets[name]
    ax.annotate(name, (params, dice), textcoords="offset points", xytext=(dx, dy),
                fontsize=9, fontweight='bold' if group == 'Ours' else 'normal',
                color=colors[group], ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))

# Axes
ax.set_xscale('log')
ax.set_xlim(0.8, 200)
ax.set_ylim(0.798, 0.842)
ax.set_xlabel('Number of Parameters (Million)', fontsize=12)
ax.set_ylabel('Dice Score (STARE)', fontsize=12)
ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100', '200'])
ax.set_yticks([0.800, 0.810, 0.820, 0.830, 0.840])
ax.tick_params(labelsize=10)
ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.6)

# Legend — deduplicate
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# Reorder
order = ["CNN", "Mamba/SSM", "Tracing", "Ours"]
leg_handles = [by_label[k] for k in order if k in by_label]
ax.legend(leg_handles, [k for k in order if k in by_label],
          loc='lower right', fontsize=9, framealpha=0.85,
          handletextpad=0.6, borderpad=0.5)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "Figure1b_ParameterAccuracy_Tradeoff.png")
fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
