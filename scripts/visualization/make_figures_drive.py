#!/usr/bin/env python3
"""Rebuild Figures 7-10 on DRIVE with the Table-6 model set (U-Net, ResUNet,
DSCNet, ERFNet, TransUNet, MambaVesselNet, PHR-Net).

Inputs (produced on the GPU server by export_drive_preds.py, copied back to
results/preds_drive_t7/):
    <Model>__<NN>.npy         float16 probability maps, 608x576 padded
    <Model>_per_image.csv     per-image Dice / IoU / HD95 / clDice
    width_bins.json           skeleton-width-bin tp/fp/fn per model

Outputs (results/paper_figures/):
    fig7_qualitative.png      full-image comparison, 2 rows x 9 columns
    fig8_roi.png              ROI crops of the same two images
    fig9_bootstrap.png        paired-bootstrap 95% CI of Dice differences
    fig10_width.png           per-vessel-width F1
"""
import os
import csv
import json

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.patches import Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_cands = [os.path.join(ROOT, "Dataset", "DRIVE", "test"),
          os.path.join(os.path.dirname(ROOT), "Dataset", "DRIVE", "test")]
DATA = next((p for p in _cands if os.path.isdir(os.path.join(p, "images"))), _cands[0])
PRED = os.path.join(ROOT, "results", "preds_drive_t7")
OUT = os.path.join(ROOT, "results", "paper_figures")
os.makedirs(OUT, exist_ok=True)

MODELS = [
    ("U-Net", "UNet", 31.04),
    ("ResUNet", "ResUNet", 13.04),
    ("DSCNet", "DSCNet", 2.10),
    ("ERFNet", "ERFNet", 2.06),
    ("TransUNet", "TransUNet", 105.32),
    ("MambaVesselNet", "MambaVesselNet", 117.22),
    ("PHR-Net (Ours)", "PHR_Net", 2.62),
]
COLORS = {
    "UNet": "#c8c8c8",
    "ResUNet": "#a9a9a9",
    "DSCNet": "#8a8a8a",
    "ERFNet": "#6d6d6d",
    "TransUNet": "#f0932b",
    "MambaVesselNet": "#7b3fa0",
    "PHR_Net": "#d62728",
}
ROI_CONFIG = [
    {"img": "02", "y": 60, "x": 300, "h": 80, "w": 80, "label": "A"},
    {"img": "17", "y": 420, "x": 120, "h": 80, "w": 80, "label": "B"},
]
SHORT = {"UNet": "U-Net", "ResUNet": "ResUNet", "DSCNet": "DSCNet", "ERFNet": "ERFNet",
         "TransUNet": "TransUNet", "MambaVesselNet": "Mamba", "PHR_Net": "PHR-Net"}
BINS = ["1-2", "3-5", "6-10", ">10"]
BIN_TITLES = ["1–2 px\n(Capillaries)", "3–5 px\n(Thin vessels)", "6–10 px\n(Medium)", ">10 px\n(Thick)"]
GREEN = np.array([0, 255, 0], dtype=np.float32)


def load_original(name):
    img = cv2.imread(os.path.join(DATA, "images", f"{name}_test.tif"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img


def load_gt(name):
    p = os.path.join(DATA, "1st_manual", f"{name}_manual1.gif")
    gt = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (gt > 127).astype(np.float32)


def load_fov(name):
    p = os.path.join(DATA, "mask", f"{name}_test_mask.gif")
    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (f > 127).astype(np.float32)


def load_prob(key, name):
    p = os.path.join(PRED, f"{key}__{name}.npy")
    prob = np.load(p).astype(np.float32)
    return prob


def overlay(img, mask):
    out = img.copy()
    h, w = out.shape[:2]
    m = (mask > 0.5)[:h, :w]
    out[m] = 0.35 * out[m] + 0.65 * GREEN
    return np.clip(out, 0, 255).astype(np.uint8)


def panel_image(img, mask, roi=None, zoom=1):
    vis = overlay(img, mask)
    if roi is not None:
        y, x, h, w = roi["y"], roi["x"], roi["h"], roi["w"]
        vis = vis[y:y + h, x:x + w]
        if zoom > 1:
            vis = cv2.resize(vis, (w * zoom, h * zoom), interpolation=cv2.INTER_NEAREST)
    return vis


def breakage_markers(gt, phr, dsc, roi, max_markers=5):
    """Thin-vessel points that PHR-Net recovers and DSCNet misses (ROI coords)."""
    h, w = gt.shape[:2]
    g = (gt[:h, :w] > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(g, cv2.DIST_L2, 5)
    thin = ((g > 0) & (dist < 2.5))
    cand = thin & (phr[:h, :w] > 0.5) & ~(dsc[:h, :w] > 0.5)
    n, lab = cv2.connectedComponents(cand.astype(np.uint8))
    pts = []
    y0, x0 = roi["y"], roi["x"]
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        if len(ys) < 3:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        if not (y0 + 6 <= cy < y0 + roi["h"] - 6 and x0 + 6 <= cx < x0 + roi["w"] - 6):
            continue
        pts.append((cx - x0, cy - y0))
    return pts[:max_markers]


def covered(pred, mx, my, rad=2):
    """Does this prediction cover the marker (ROI coords)?"""
    h, w = pred.shape[:2]
    x, y = int(round(mx)), int(round(my))
    y0, y1 = max(y - rad, 0), min(y + rad + 1, h)
    x0, x1 = max(x - rad, 0), min(x + rad + 1, w)
    return bool((pred[y0:y1, x0:x1] > 0.5).any())


def figure_78(roi_mode):
    rows = ROI_CONFIG
    n_mod = len(MODELS)
    ncol = 2 + n_mod
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.62 * ncol, 1.55 * len(rows)))
    zoom = 4 if roi_mode else 1
    for r, cfg in enumerate(rows):
        img = load_original(cfg["img"])
        gt = load_gt(cfg["img"])
        fov = load_fov(cfg["img"])
        roi = {"y": cfg["y"], "x": cfg["x"], "h": cfg["h"], "w": cfg["w"]} if roi_mode else None

        panels = [("Original", img, None), ("Ground Truth", img, gt)]
        h, w = img.shape[:2]
        for disp, key, _ in MODELS:
            prob = load_prob(key, cfg["img"])[:h, :w] * fov[:h, :w]
            panels.append((SHORT[key], img, prob))

        markers = []
        if roi_mode:
            markers = breakage_markers(gt, load_prob("PHR_Net", cfg["img"])[:h, :w] * fov[:h, :w],
                                       load_prob("DSCNet", cfg["img"])[:h, :w] * fov[:h, :w], roi)

        for c, (title, base, mask) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel_image(base, mask if mask is not None else np.zeros_like(gt), roi, zoom))
            if roi_mode and c >= 2:
                pred = mask[:gt.shape[0], :gt.shape[1]] if mask is not None else None
                for mx, my in markers:
                    ok = covered(pred, cfg["x"] + mx, cfg["y"] + my) if pred is not None else False
                    ax.plot(mx * zoom, my * zoom, "o", color="#22cc22" if ok else "#ff2222",
                            markersize=4.4, markeredgecolor="white", markeredgewidth=0.7, zorder=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=15, fontweight="bold", pad=5,
                             color="#b00020" if title.startswith("PHR-Net") else "black")
            for sp in ax.spines.values():
                sp.set_color("#b00020" if title.startswith("PHR-Net") else "#dddddd")
                sp.set_linewidth(3.0 if "Ours" in title else 1.0)
            if not roi_mode:
                ax.add_patch(Rectangle((cfg["x"], cfg["y"]), cfg["w"], cfg["h"],
                                       fill=False, edgecolor="#00b050", lw=1.6))
                ax.text(cfg["x"] + 3, cfg["y"] - 6, cfg["label"], color="#00b050",
                        fontsize=12, fontweight="bold")
        axes[r, 0].set_ylabel(f"Image {cfg['img']}", fontsize=13, rotation=90, labelpad=8, va="center")
    fig.subplots_adjust(left=0.045, right=0.995, top=0.92, bottom=0.01, hspace=0.06, wspace=0.02)
    out = os.path.join(OUT, "fig8_roi.png" if roi_mode else "fig7_qualitative.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def load_dice():
    d = {}
    for disp, key, _ in MODELS:
        rows = list(csv.DictReader(open(os.path.join(PRED, f"{key}_per_image.csv"))))
        d[key] = np.array([float(r["Dice"]) for r in rows])
    return d


def figure_9(n_boot=10000, seed=0):
    dice = load_dice()
    ours = dice["PHR_Net"]
    names = [disp for disp, key, _ in MODELS if key != "PHR_Net"]
    keys = [key for disp, key, _ in MODELS if key != "PHR_Net"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ours), size=(n_boot, len(ours)))
    res = []
    for disp, key in zip(names, keys):
        d = ours - dice[key]
        b = d[idx].mean(axis=1)
        res.append((disp, d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), d))

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ys = np.arange(len(res))[::-1]
    for y, (disp, mean, lo, hi, d) in zip(ys, res):
        sig = lo > 0 or hi < 0
        c = "#2b7bba" if mean > 0 else "#c0392b"
        ax.plot([lo, hi], [y, y], color=c, lw=2.6, alpha=0.95, solid_capstyle="round")
        ax.plot([lo, lo], [y - 0.11, y + 0.11], color=c, lw=2.0)
        ax.plot([hi, hi], [y - 0.11, y + 0.11], color=c, lw=2.0)
        ax.scatter([mean], [y], s=90, color=c, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(hi + 0.0022, y, f"{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                va="center", fontsize=8.2, color="#333333")
    ax.axvline(0, color="#555555", lw=1.3, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in res], fontsize=11)
    ax.set_xlabel("ΔDice (PHR-Net − baseline)", fontsize=12)
    ax.set_title("Paired bootstrap 95% CI on DRIVE (10,000 resamples)", fontsize=12.5,
                 fontweight="bold")
    ax.grid(axis="x", ls=":", color="#cccccc", lw=0.8)
    ax.set_axisbelow(True)
    xmax = max(r[3] for r in res)
    xmin = min(r[2] for r in res)
    ax.set_xlim(min(xmin - 0.012, -0.02), xmax + 0.006)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT, "fig9_bootstrap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    for disp, mean, lo, hi, _ in res:
        print(f"  {disp:16s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def figure_10():
    bins = json.load(open(os.path.join(PRED, "width_bins.json")))
    n_mod = len(MODELS)
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    width = 0.8 / n_mod
    for i, (disp, key, _) in enumerate(MODELS):
        vals = []
        for b in BINS:
            tp, fp, fn = bins[key][b]
            vals.append(2 * tp / (2 * tp + fp + fn + 1e-9))
        xs = np.arange(len(BINS)) + (i - (n_mod - 1) / 2) * width
        ax.bar(xs, vals, width * 0.92, color=COLORS[key], label=disp,
               edgecolor="white", linewidth=0.6,
               zorder=3 if key == "PHR_Net" else 2)
        if key == "PHR_Net":
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.4, color="#d62728", fontweight="bold")
    ax.set_xticks(np.arange(len(BINS)))
    ax.set_xticklabels(BIN_TITLES, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Per-Vessel-Width F1 on DRIVE (Skeleton-Based)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.47, 1.045)
    ax.grid(axis="y", ls=":", color="#cccccc", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, fontsize=9.2, loc="upper center", frameon=False,
              bbox_to_anchor=(0.5, -0.20), handlelength=1.4, columnspacing=1.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT, "fig10_width.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure_78(roi_mode=False)
    figure_78(roi_mode=True)
    figure_9()
    figure_10()
