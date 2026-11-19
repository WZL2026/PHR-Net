#!/usr/bin/env python3
"""
Per-Vessel-Width Performance Analysis (Revised)
================================================
Fixed methodology: measure width at skeleton (centerline) pixels only,
then evaluate in a small neighborhood around skeleton points of each width bin.
This avoids conflating thick-vessel edges with true capillaries.

Compares PHR-Net against all 5 models (incl. MambaVesselNet) on STARE.

Requires: mamba_ssm (install via `pip install mamba-ssm` on GPU server)
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
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
import torchvision.transforms.functional as TF

SCRIPT_DIR = _p
sys.path.insert(0, SCRIPT_DIR)
from models.unet import UNet
from models.resunet import ResUnet
from models.dscnet import DSCNet_pro
from models.mamba_vessel_net import MambaVesselNet
from models.netracer_net import NETracer_Net
from models.phr_net import PHR_Net

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Dataset", "STARE")
W_DIR = os.path.join(SCRIPT_DIR, "weights", "Comparison", "STARE")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All 5 models
MODEL_SPECS = [
    ("UNet",            UNet,            "best_unet_stare_fold1.pth",            "#b0b0b0"),
    ("ResUNet",         ResUnet,         "best_resunet_stare_fold1.pth",         "#b0b0b0"),
    ("DSCNet",          DSCNet_pro,      "best_dscnet_stare_fold1.pth",          "#7f7f7f"),
    ("MambaVesselNet",  MambaVesselNet,  "best_mambavesselnet_stare_fold1.pth",  "#e67e22"),
    ("PHR-Net (Ours)",  PHR_Net,         "best_phr_net_stare_fold1.pth",         "#d62728"),
]

WIDTH_BINS = [
    ("1-2 px\n(Capillaries)", 1, 2),
    ("3-5 px\n(Thin vessels)", 3, 5),
    ("6-10 px\n(Medium)",      6, 10),
    (">10 px\n(Thick)",        11, 999),
]


def skeleton_width_map(gt_mask):
    """
    Compute vessel width at skeleton (centerline) pixels only.

    Method:
      1. Skeletonize the binary vessel mask → centerline (1px wide)
      2. Distance transform: for each foreground pixel, distance to nearest background
      3. At each skeleton pixel, half-width = distance_transform value
      4. Width = 2 × half-width

    This avoids the edge contamination problem: thick vessel edges are NOT
    on the skeleton, so they won't be misclassified as "thin" vessels.
    """
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    if gt_bin.sum() == 0:
        return np.zeros_like(gt_bin, dtype=np.float32), np.zeros_like(gt_bin, dtype=bool)

    skeleton = skeletonize(gt_bin).astype(np.uint8)
    dist_bg = distance_transform_edt(gt_bin)

    width_map = np.zeros_like(gt_bin, dtype=np.float32)
    width_map[skeleton > 0] = 2.0 * dist_bg[skeleton > 0]
    width_map = np.clip(width_map, 0, 50)

    return width_map, skeleton > 0


def load_and_pad(img_path, gt_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fov = (gray > 10).astype(np.uint8) * 255

    H, W = image.shape[:2]
    pad_w = (32 - W % 32) % 32
    pad_h = (32 - H % 32) % 32
    if pad_w > 0 or pad_h > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        fov = np.pad(fov, ((0, pad_h), (0, pad_w)), mode='constant')

    img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = TF.normalize(img_t, mean=[0.215, 0.215, 0.215],
                              std=[0.151, 0.151, 0.151])
    mask_tensor = (torch.from_numpy(mask).float() / 255.0 > 0.5).float()
    return img_tensor, mask_tensor, fov


def compute_binned_metrics(pred, gt, width_map, skeleton, fov=None):
    """
    Evaluate per-bin metrics. Only consider pixels within a 2-pixel
    neighborhood of skeleton points in each width bin.
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)

    if fov is not None:
        fov_bin = (fov > 0).astype(bool)  # already 0/255, just binarize
    else:
        fov_bin = np.ones_like(gt_bin, dtype=bool)

    results = {}
    for label, w_min, w_max in WIDTH_BINS:
        # Select skeleton pixels in this width range
        skel_in_bin = skeleton & (width_map >= w_min) & (width_map <= w_max)
        if skel_in_bin.sum() == 0:
            results[label] = {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                              "tp": 0, "fp": 0, "fn": 0, "n_skel": 0}
            continue

        # Dilate skeleton by 2px to get evaluation neighborhood
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eval_mask = cv2.dilate(skel_in_bin.astype(np.uint8), kernel).astype(bool)
        eval_mask = eval_mask & fov_bin  # only within FOV

        tp = (pred_bin & gt_bin & eval_mask).sum()
        fp = (pred_bin & (1 - gt_bin) & eval_mask).sum()
        fn = ((1 - pred_bin) & gt_bin & eval_mask).sum()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[label] = {"recall": recall, "precision": precision, "f1": f1,
                           "tp": int(tp), "fp": int(fp), "fn": int(fn),
                           "n_skel": int(skel_in_bin.sum())}
    return results


def main():
    print(f"Device: {DEVICE}")

    # ---- Load all models ----
    models = {}
    for name, model_cls, weight_file, color in MODEL_SPECS:
        print(f"Loading {name}...")
        if name == "PHR-Net (Ours)":
            m = model_cls(in_channels=3, out_classes=1, use_a=True, use_b=True,
                          backbone="hrnet", dsconv_mode="none").to(DEVICE)
        elif name == "DSCNet":
            m = model_cls(3, 1).to(DEVICE)
        elif name == "MambaVesselNet":
            m = model_cls(in_ch=3, out_ch=1).to(DEVICE)
        elif name == "NETracer":
            m = model_cls(in_channels=3, out_channels=1).to(DEVICE)
        else:
            m = model_cls(in_channels=3, out_channels=1).to(DEVICE)

        ckpt = torch.load(os.path.join(W_DIR, weight_file), map_location=DEVICE, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models[name] = m

    # ---- Per-bin accumulators for all models ----
    all_stats = {name: {label: {"tp": 0, "fp": 0, "fn": 0}
                 for label, _, _ in WIDTH_BINS}
                 for name, _, _, _ in MODEL_SPECS}

    # ---- Process all STARE images ----
    img_dir = os.path.join(DATA_DIR, "images")
    mask_dir = os.path.join(DATA_DIR, "masks")
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

    for img_name in img_files:
        base = img_name.replace('.png', '')
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(mask_dir, f"{base}_ah.png")

        print(f"  {base}...", end=" ", flush=True)
        img_t, gt_t, fov_np = load_and_pad(img_path, gt_path)
        img_batch = img_t.unsqueeze(0).to(DEVICE)
        gt_np = gt_t.numpy()
        width_map, skeleton = skeleton_width_map(gt_np)

        with torch.no_grad():
            for name, _, _, _ in MODEL_SPECS:
                pred = torch.sigmoid(models[name](img_batch))
                # Handle tuple outputs (some models return tuples in eval mode)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred_np = pred.squeeze().cpu().numpy()

                # Auto FOV from image
                gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                H, W = gray.shape
                pad_w = (32 - W % 32) % 32; pad_h = (32 - H % 32) % 32
                fov = (gray > 10).astype(np.uint8)
                if pad_w > 0 or pad_h > 0:
                    fov = np.pad(fov, ((0, pad_h), (0, pad_w)), mode='constant')

                rm = compute_binned_metrics(pred_np, gt_np, width_map, skeleton, fov)
                for label, _, _ in WIDTH_BINS:
                    all_stats[name][label]["tp"] += rm[label]["tp"]
                    all_stats[name][label]["fp"] += rm[label]["fp"]
                    all_stats[name][label]["fn"] += rm[label]["fn"]

        print("OK", flush=True)

    # ---- Compute global metrics for all models ----
    bin_labels = [b[0] for b in WIDTH_BINS]

    def global_metrics(d):
        out = {}
        for label, _, _ in WIDTH_BINS:
            tp, fp, fn = d[label]["tp"], d[label]["fp"], d[label]["fn"]
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            out[label] = {"recall": rec, "precision": prec, "f1": f1}
        return out

    all_metrics = {name: global_metrics(all_stats[name])
                   for name, _, _, _ in MODEL_SPECS}

    # ---- Print table ----
    print(f"\n=== Per-Vessel-Width F1 (STARE, skeleton-based) ===")
    header = f"{'Model':<18}"
    for label, _, _ in WIDTH_BINS:
        header += f" {label:<25}"
    print(header)
    print("-" * len(header))
    for name, _, _, _ in MODEL_SPECS:
        line = f"{name:<18}"
        for label, _, _ in WIDTH_BINS:
            f1 = all_metrics[name][label]["f1"]
            rec = all_metrics[name][label]["recall"]
            line += f" F1={f1:.4f} R={rec:.4f}  "
        print(line)

    # ---- Plot: grouped bar chart of F1 across all models per bin ----
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    x = np.arange(len(bin_labels))
    n_models = len(MODEL_SPECS)
    w = 0.8 / n_models

    for mi, (name, _, _, color) in enumerate(MODEL_SPECS):
        vals = [all_metrics[name][label]["f1"] for label, _, _ in WIDTH_BINS]
        offset = (mi - (n_models - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=name, color=color,
                      edgecolor='white', lw=0.5)
        # Value labels (only for PHR-Net to avoid clutter)
        if name == "PHR-Net (Ours)":
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8.5,
                        fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Vessel-Width F1: All 6 Models on STARE (Skeleton-Based)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8.5, loc='lower right', ncol=2)
    ax.set_ylim(0.63, 1.05)
    ax.grid(axis='y', alpha=0.2, linestyle=':')

    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, "Figure_FineVessel_Analysis.png")
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
