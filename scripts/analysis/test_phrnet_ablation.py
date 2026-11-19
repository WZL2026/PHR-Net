#!/usr/bin/env python3
"""
PHR-Net 全排列消融实验测试脚本
  模块:
    C = HRNet (高分辨率双流) vs UNet (深度下采样编解码)
    A = DPE+DPSS+DSSB (拉普拉斯先验 + 先验放大器 + 双域选择性块)
    B = LKA+RSFH (大核注意力 + 条带池化融合)
    P = PGCA (prior-gated channel attention, HRNet only)
  输出: 主效应 + 交互效应 + 分组消融表 + 模块贡献排名

  运行方式:
    python test_phrnet_ablation.py                          # 全部 8 configs
    python test_phrnet_ablation.py --backbone unet          # 只测试 UNet (4 新增)
    python test_phrnet_ablation.py --backbone hrnet         # 只测试 HRNet (4 已有)
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
import torch
import random
import numpy as np
import csv
from datetime import datetime


def worker_init_fn(worker_id):
    """Seed numpy/random per worker for reproducibility with epoch diversity.

    PyTorch already seeds each worker's torch RNG with base_seed + worker_id
    BEFORE calling worker_init_fn, and base_seed varies each epoch.  We read
    that seed and propagate it to numpy/random WITHOUT overriding torch's RNG.
    This gives:
      - Different seeds per epoch (natural diversity from DataLoader internals)
      - Different seeds per worker (worker_id baked into initial_seed)
      - Full run reproducibility (set_seed(42) determines the seed progression)
    """
    seed = int(torch.initial_seed() % (2**31 - 1))
    np.random.seed(seed)
    random.seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.phr_net import PHR_Net
from utils.dataset import DRIVEDataset, CHASEDB1Dataset, STAREDataset
from utils.metrics import compute_hd95, compute_cldice, sliding_window_inference
from utils.kfold import create_kfold_dataloaders
from utils.logger import Logger, count_params, section_header

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


# ==============================================================================
# Argument parsing (simple, before Logger hijacks stdout)
# ==============================================================================
BACKBONE_FILTER = "all"
PGCA_FILTER = None
for i, a in enumerate(sys.argv):
    if a == "--backbone" and i + 1 < len(sys.argv):
        BACKBONE_FILTER = sys.argv[i + 1]
        if BACKBONE_FILTER not in ("all", "hrnet", "unet"):
            print(f"ERROR: --backbone must be 'all', 'hrnet', or 'unet' (got '{BACKBONE_FILTER}')")
            sys.exit(1)
    if a == "--pgca" and i + 1 < len(sys.argv):
        PGCA_FILTER = sys.argv[i + 1]
        if PGCA_FILTER not in ("on", "off"):
            print(f"ERROR: --pgca must be 'on' or 'off' (got '{PGCA_FILTER}')")
            sys.exit(1)

# ==============================================================================
# 消融配置 — 3 模块全排列 (C × A × B × P)
#   C = HRNet vs UNet (backbone)
#   A = DPE+DPSS+DSSB (Laplacian prior + prior amplifier + dual-scope selective)
#   B = LKA+RSFH (large-kernel attention + strip fusion)
#   P = PGCA     (prior-gated channel attention, HRNet only)
# ==============================================================================
ABLATION_CONFIGS = {
    # ═══════════════════════════════════════════════════════════════════════
    # HRNet backbone: 2³ full factorial (A × B × P)
    #   A = DPE+DPSS+DSSB (prior enhancement + dual-scope selective)
    #   B = LKA+RSFH      (attention + strip fusion)
    #   P = PGCA          (prior-gated channel exchange)
    # ═══════════════════════════════════════════════════════════════════════

    # ── P=0 (no PGCA: plain residual exchange) ──
    "000_Base": {
        "desc": "HRNet Base (no prior, no attn, no PGCA)",
        "backbone": "hrnet", "use_a": False, "use_b": False, "use_pgca": False,
    },
    "001_B": {
        "desc": "HRNet +B (attn, no prior, no PGCA)",
        "backbone": "hrnet", "use_a": False, "use_b": True, "use_pgca": False,
    },
    "010_A": {
        "desc": "HRNet +A (prior, no attn, no PGCA)",
        "backbone": "hrnet", "use_a": True, "use_b": False, "use_pgca": False,
    },
    "011_Full": {
        "desc": "HRNet +A+B (prior+attn, no PGCA)",
        "backbone": "hrnet", "use_a": True, "use_b": True, "use_pgca": False,
    },

    # ── P=1 (with PGCA: prior-gated channel exchange) ──
    "100_Base_PGCA": {
        "desc": "HRNet+PGCA Base (prior-gated exchange, no prior, no attn)",
        "backbone": "hrnet", "use_a": False, "use_b": False, "use_pgca": True,
    },
    "101_B_PGCA": {
        "desc": "HRNet+PGCA +B (prior-gated exchange + attn)",
        "backbone": "hrnet", "use_a": False, "use_b": True, "use_pgca": True,
    },
    "110_A_PGCA": {
        "desc": "HRNet+PGCA +A (prior-gated exchange + prior)",
        "backbone": "hrnet", "use_a": True, "use_b": False, "use_pgca": True,
    },
    "111_Full_PGCA": {
        "desc": "HRNet+PGCA Full = A+B+P (complete PHR-Net)",
        "backbone": "hrnet", "use_a": True, "use_b": True, "use_pgca": True,
    },

    # ═══════════════════════════════════════════════════════════════════════
    # UNet backbone: 2² full factorial (A × B) — PGCA not applicable
    # ═══════════════════════════════════════════════════════════════════════
    "00_UNet_Base": {
        "desc": "UNet Base (no prior, no attn)",
        "backbone": "unet", "use_a": False, "use_b": False,
    },
    "01_UNet_B": {
        "desc": "UNet +B (attn, no prior)",
        "backbone": "unet", "use_a": False, "use_b": True,
    },
    "10_UNet_A": {
        "desc": "UNet +A (prior, no attn)",
        "backbone": "unet", "use_a": True, "use_b": False,
    },
    "11_UNet_Full": {
        "desc": "UNet +A+B (prior+attn)",
        "backbone": "unet", "use_a": True, "use_b": True,
    },
}

# Apply --backbone filter
if BACKBONE_FILTER in ("hrnet", "unet"):
    ABLATION_CONFIGS = {
        k: v for k, v in ABLATION_CONFIGS.items()
        if v["backbone"] == BACKBONE_FILTER
    }

# Apply --pgca filter (HRNet-only)
if PGCA_FILTER is not None:
    ABLATION_CONFIGS = {
        k: v for k, v in ABLATION_CONFIGS.items()
        if v.get("backbone") == "hrnet" and v.get("use_pgca") == (PGCA_FILTER == "on")
    }

DATASETS = ["DRIVE", "CHASEDB1", "STARE"]
WEIGHTS_BASE = "weights/Ablation"
RESULTS_BASE = "results/phrnet"
os.makedirs(RESULTS_BASE, exist_ok=True)

METRIC_NAMES = ["Dice", "IoU", "Precision", "Recall", "Spec", "FPR", "HD95", "clDice"]


def measure_complexity(model, input_size=(1, 3, 256, 256)):
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6
    flops_g = 0.0
    if HAS_THOP:
        try:
            dummy = torch.randn(input_size).to(next(model.parameters()).device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops_g = macs / 1e9
        except Exception:
            flops_g = 0.0
    return params_m, flops_g


def evaluate_model(model, test_loader):
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_hd95, hd95_count, total_cldice = 0.0, 0, 0.0

    with torch.no_grad():
        for images, masks, fovs in test_loader:
            images, masks, fovs = images.to(DEVICE), masks.to(DEVICE), fovs.to(DEVICE)
            pred_np = sliding_window_inference(model, images, window_size=(512, 512), stride=(256, 256), device=DEVICE)
            pred_mask = (pred_np > 0.5).astype(np.float32)
            fov_np = fovs.cpu().numpy()[0, 0]
            pred_mask = pred_mask * fov_np
            masks_np = masks.cpu().numpy()[0, 0] * fov_np

            total_tp += (pred_mask * masks_np).sum()
            total_fp += (pred_mask * (1 - masks_np)).sum()
            total_fn += ((1 - pred_mask) * masks_np).sum()
            total_tn += ((1 - pred_mask) * (1 - masks_np)).sum()

            hd95 = compute_hd95(pred_mask, masks_np)
            if not np.isnan(hd95):
                total_hd95 += hd95
                hd95_count += 1
            total_cldice += compute_cldice(pred_mask, masks_np)

    eps = 1e-6
    dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)
    spec = (total_tn + eps) / (total_tn + total_fp + eps)
    fpr = (total_fp + eps) / (total_fp + total_tn + eps)
    avg_hd95 = total_hd95 / hd95_count if hd95_count > 0 else np.nan
    avg_cldice = total_cldice / len(test_loader.dataset)

    return [dice, iou, precision, recall, spec, fpr, avg_hd95, avg_cldice]


def fmt_val(v, decimals=4):
    if np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def main():
    logger = Logger(log_dir="logs", prefix="test_ablation")
    sys.stdout = logger

    print(f"\n{'#'*100}")
    print(f"  PHR-Net 3-Factor Ablation Testing  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Configs: {len(ABLATION_CONFIGS)}  |  Datasets: {DATASETS}")
    print(f"  Modules: C(backbone) × A(prior) × B(attention)")
    if BACKBONE_FILTER != "all":
        print(f"  Backbone filter: {BACKBONE_FILTER}")
    print(f"  Device: {DEVICE}")
    print(f"{'#'*100}")

    # Pre-compute complexity for all configs
    print(section_header("Model Complexity"))
    complexity = {}
    for ab_name, cfg in ABLATION_CONFIGS.items():
        model_kwargs = {k: v for k, v in cfg.items() if k not in ("desc",)}
        m = PHR_Net(in_channels=3, out_classes=1, **model_kwargs).to(DEVICE)
        complexity[ab_name] = measure_complexity(m)
        del m
        torch.cuda.empty_cache()

    # Collect results
    all_results = []
    for dataset_name in DATASETS:
        base_dir = f"../Dataset/{dataset_name}"
        print(section_header(f"Testing on {dataset_name}"))

        for ab_name, cfg in ABLATION_CONFIGS.items():
            desc = cfg["desc"]
            model_kwargs = {k: v for k, v in cfg.items() if k not in ("desc",)}
            params_m, flops_g = complexity[ab_name]
            weight_dir = os.path.join(WEIGHTS_BASE, dataset_name, ab_name)

            if not os.path.isdir(weight_dir):
                print(f"  [SKIP] {ab_name}: no weights at {weight_dir}")
                continue

            print(f"\n  [{ab_name}] {desc}  ({params_m:.2f}M / {flops_g:.2f}G)")

            try:
                if dataset_name == "DRIVE":
                    test_dataset = DRIVEDataset(base_dir, mode="test")
                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
                    weight_path = os.path.join(weight_dir, "best.pth")
                    if not os.path.exists(weight_path):
                        print(f"    [WARN] missing: {weight_path}")
                        continue
                    model = PHR_Net(in_channels=3, out_classes=1, **model_kwargs).to(DEVICE)
                    try:
                        model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
                    except RuntimeError:
                        print(f"    [SKIP] Weight shape mismatch (need retrain): {weight_path}")
                        continue
                    metrics = evaluate_model(model, test_loader)
                    for fi in range(len(METRIC_NAMES)):
                        all_results.append({
                            "Dataset": dataset_name, "Config": ab_name,
                            "Desc": desc, "Params_M": params_m, "FLOPs_G": flops_g,
                            "Fold": "—", "Metric": METRIC_NAMES[fi], "Value": metrics[fi], "Std": 0.0,
                        })
                    print(f"    Dice={metrics[0]:.4f}  IoU={metrics[1]:.4f}  HD95={fmt_val(metrics[6])}")
                    del model, test_loader, test_dataset

                else:
                    # CHASEDB1 / STARE: 5-fold cross-validation
                    DatasetClass = CHASEDB1Dataset if dataset_name == "CHASEDB1" else STAREDataset
                    train_dataset = DatasetClass(base_dir, mode="training")
                    val_dataset = DatasetClass(base_dir, mode="test")

                    fold_metrics = []
                    missing_folds = []

                    print(f"    {'Fold':<8s} {'Dice':>8s} {'IoU':>8s} {'HD95':>8s}")
                    print(f"    {'-'*36}")

                    for _, test_loader_fold, fold_idx in create_kfold_dataloaders(
                            train_dataset, val_dataset, batch_size=1,
                            worker_init_fn=worker_init_fn):
                        weight_path = os.path.join(weight_dir, f"best_fold{fold_idx+1}.pth")
                        if not os.path.exists(weight_path):
                            print(f"    F{fold_idx+1:<7d} [WARN] missing: {weight_path}")
                            missing_folds.append(fold_idx + 1)
                            continue

                        model = PHR_Net(in_channels=3, out_classes=1, **model_kwargs).to(DEVICE)
                        try:
                            model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
                        except RuntimeError:
                            print(f"    F{fold_idx+1:<7d} [SKIP] shape mismatch: {weight_path}")
                            missing_folds.append(fold_idx + 1)
                            del model
                            continue

                        metrics = evaluate_model(model, test_loader_fold)
                        fold_metrics.append(metrics)
                        print(f"    F{fold_idx+1:<7d} {metrics[0]:8.4f} {metrics[1]:8.4f} {metrics[6]:8.4f}")
                        del model

                    del train_dataset, val_dataset
                    torch.cuda.empty_cache()

                    if not fold_metrics:
                        print(f"    [WARN] no valid fold weights for {ab_name}")
                        continue

                    fm = np.array(fold_metrics)
                    means = fm.mean(axis=0)
                    stds = fm.std(axis=0)
                    if stds[0] == 0 and len(fold_metrics) == 1:
                        stds = np.array([0.0] * len(METRIC_NAMES))

                    fold_tag = f"5-fold ({len(fold_metrics)}/5)" if missing_folds else "5-fold"
                    print(f"    {'Mean':<8s} {means[0]:8.4f} {means[1]:8.4f} {means[6]:8.4f}")
                    print(f"    {'+-Std':<8s} {stds[0]:8.4f} {stds[1]:8.4f} {stds[6]:8.4f}")

                    for fi in range(len(METRIC_NAMES)):
                        all_results.append({
                            "Dataset": dataset_name, "Config": ab_name,
                            "Desc": desc, "Params_M": params_m, "FLOPs_G": flops_g,
                            "Fold": fold_tag,
                            "Metric": METRIC_NAMES[fi], "Value": means[fi], "Std": stds[fi],
                        })

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    [FAIL] {e}")
                import traceback
                traceback.print_exc()

    # ==========================================================================
    # Pivot results: config-level rows for display
    # ==========================================================================
    config_results = {}  # (Dataset, Config) → {metric: value, metric_std: std, ...}
    for r in all_results:
        key = (r["Dataset"], r["Config"])
        if key not in config_results:
            config_results[key] = {
                "Dataset": r["Dataset"], "Config": r["Config"],
                "Desc": r["Desc"], "Params_M": r["Params_M"], "FLOPs_G": r["FLOPs_G"],
                "Fold": r["Fold"],
            }
        config_results[key][r["Metric"]] = r["Value"]
        config_results[key][r["Metric"] + "_std"] = r["Std"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================================
    # Helper: get a specific config row from ds_results
    # ==========================================================================
    def get_row(ds_results, config_name):
        return next((r for r in ds_results if r["Config"] == config_name), None)

    # ==========================================================================
    # Helper: compute avg dice over a set of configs
    # ==========================================================================
    def avg_dice(ds_results, names):
        vals = [r["Dice"] for r in ds_results if r["Config"] in names]
        return np.mean(vals) if vals else np.nan

    for dataset_name in DATASETS:
        ds_results = [v for v in config_results.values() if v["Dataset"] == dataset_name]
        if not ds_results:
            continue

        # Attach backbone + n_modules from ABLATION_CONFIGS
        for r in ds_results:
            cfg = ABLATION_CONFIGS.get(r["Config"], {})
            r["backbone"] = cfg.get("backbone", "?")
            r["n_modules"] = cfg.get("n_modules", -1)

        ds_results.sort(key=lambda r: (r["backbone"], r["Config"]))

        is_kfold = (ds_results[0]["Fold"] not in ("—", "-", None))
        col_w = 14 if is_kfold else 10

        # ======================================================================
        # Per-Backbone Ablation Tables
        # ======================================================================
        backbones_present = sorted(set(r["backbone"] for r in ds_results))
        bb_labels = {"hrnet": "HRNet (Dual-Stream, High-Res Features)",
                     "unet": "UNet (Encoder-Decoder, Deep Downsampling)"}

        for bb in backbones_present:
            bb_results = [r for r in ds_results if r["backbone"] == bb]
            bb_results.sort(key=lambda r: r["n_modules"])

            base_row = get_row(bb_results, f"000_Base" if bb == "hrnet" else "00_UNet_Base")
            full_row = get_row(bb_results, f"011_Full" if bb == "hrnet" else "11_UNet_Full")
            base_dice = base_row["Dice"] if base_row else None

            hdr = (f"  {'#M':>2} {'Config':<18} {'Params':>7} {'FLOPs':>6} "
                   + "".join(f"{m:>{col_w}}" for m in METRIC_NAMES)
                   + f"  {'Delta_Dice':>10}")

            print(section_header(f"Ablation — {dataset_name} / {bb_labels.get(bb, bb)}", 150))
            print(hdr)
            print(f"  {'-'*len(hdr)}")

            for r in bb_results:
                n_mod = r["n_modules"]
                delta = f"{r['Dice'] - base_dice:+.4f}" if base_dice is not None else "—"

                if is_kfold:
                    vals = "".join(f"{fmt_val(r[m],4)}+-{fmt_val(r.get(m+'_std',0),3)}".rjust(col_w) for m in METRIC_NAMES)
                else:
                    vals = "".join(fmt_val(r[m], 4).rjust(col_w) for m in METRIC_NAMES)
                print(f"  {n_mod:2d} {r['Config']:<18} {r['Params_M']:6.2f}M {r['FLOPs_G']:5.2f}G {vals}  {delta:>10}")

            # Module contribution within this backbone
            if full_row:
                row_a = get_row(bb_results, "010_A" if bb == "hrnet" else "10_UNet_A")
                row_b = get_row(bb_results, "001_B" if bb == "hrnet" else "01_UNet_B")
                row_base_local = get_row(bb_results, "000_Base" if bb == "hrnet" else "00_UNet_Base")

                contributions = {}
                if row_b:
                    contributions["A (DPE+DPSS+DSSB)"] = full_row["Dice"] - row_b["Dice"]
                if row_a:
                    contributions["B (LKA+RSFH)"] = full_row["Dice"] - row_a["Dice"]

                print(f"\n  Module Contribution (Full minus module) — {bb}:")
                print(f"  {'Module':<18} {'Delta_Dice':>11}  {'Verdict':>20}")
                print(f"  {'-'*55}")
                ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
                for mod, dd in ranked:
                    if dd > +0.003:
                        verdict = "*** Beneficial"
                    elif dd > +0.001:
                        verdict = "**  Beneficial"
                    elif dd > -0.001:
                        verdict = "*   Beneficial/Neutral"
                    elif dd > -0.003:
                        verdict = "--  Marginal"
                    else:
                        verdict = "    Harmful"
                    print(f"  {mod:<18} {dd:+11.4f}  {verdict:>20}")

                if row_base_local:
                    print(f"\n  Standalone gain vs Base ({bb}):")
                    if row_a:
                        print(f"  {'A alone':<18} {row_a['Dice'] - row_base_local['Dice']:+11.4f}")
                    if row_b:
                        print(f"  {'B alone':<18} {row_b['Dice'] - row_base_local['Dice']:+11.4f}")
                    print(f"  {'A+B (Full)':<18} {full_row['Dice'] - row_base_local['Dice']:+11.4f}")
            print()

        # ======================================================================
        # 3-Factor Factorial Analysis (only if both backbones present)
        # ======================================================================
        if len(backbones_present) >= 2:
            print(section_header(f"Factorial Analysis — {dataset_name}  (2³: Backbone × Prior × Attention)", 110))

            hrnet_cfgs = {r["Config"]: r for r in ds_results if r["backbone"] == "hrnet"}
            unet_cfgs = {r["Config"]: r for r in ds_results if r["backbone"] == "unet"}

            # Helper: unified access by (backbone, a, b)
            def dice_val(bb, a, b):
                if bb == "hrnet":
                    key = "011_Full" if a and b else ("010_A" if a else ("001_B" if b else "000_Base"))
                else:
                    key = "11_UNet_Full" if a and b else ("10_UNet_A" if a else ("01_UNet_B" if b else "00_UNet_Base"))
                row = (hrnet_cfgs if bb == "hrnet" else unet_cfgs).get(key)
                return row["Dice"] if row else np.nan

            # Main Effects
            # Effect(C) = avg(HRNet) - avg(UNet)
            hrnet_vals = [r["Dice"] for r in ds_results if r["backbone"] == "hrnet"]
            unet_vals = [r["Dice"] for r in ds_results if r["backbone"] == "unet"]
            eff_c = np.mean(hrnet_vals) - np.mean(unet_vals) if hrnet_vals and unet_vals else np.nan

            # Effect(A) = avg(+A) - avg(-A), averaging over B and C
            a_on  = [r["Dice"] for r in ds_results if "A" in r["Config"] or "Full" in r["Config"]]
            a_on  = [r["Dice"] for r in ds_results if ABLATION_CONFIGS.get(r["Config"], {}).get("use_a", False)]
            a_off = [r["Dice"] for r in ds_results if not ABLATION_CONFIGS.get(r["Config"], {}).get("use_a", False)]
            eff_a = np.mean(a_on) - np.mean(a_off) if a_on and a_off else np.nan

            # Effect(B) = avg(+B) - avg(-B), averaging over A and C
            b_on  = [r["Dice"] for r in ds_results if ABLATION_CONFIGS.get(r["Config"], {}).get("use_b", False)]
            b_off = [r["Dice"] for r in ds_results if not ABLATION_CONFIGS.get(r["Config"], {}).get("use_b", False)]
            eff_b = np.mean(b_on) - np.mean(b_off) if b_on and b_off else np.nan

            print(f"\n  Main Effects:")
            print(f"  {'Effect':<28} {'Delta_Dice':>10}  {'Direction'}")
            print(f"  {'-'*55}")
            eff_c_str = f"{eff_c:+.4f}" if not np.isnan(eff_c) else "N/A"
            print(f"  {'C: Backbone (HRNet vs UNet)':<28} {eff_c_str:>10}  {'HRNet > UNet' if eff_c > 0 else 'UNet > HRNet' if eff_c < 0 else '≈'}")
            eff_a_str = f"{eff_a:+.4f}" if not np.isnan(eff_a) else "N/A"
            print(f"  {'A: Module A (DPE+DPSS+DSSB)':<28} {eff_a_str:>10}  {'+Module A better' if eff_a > 0 else '−Module A hurts' if eff_a < 0 else '≈'}")
            eff_b_str = f"{eff_b:+.4f}" if not np.isnan(eff_b) else "N/A"
            print(f"  {'B: Attn (LKA+RSFH on/off)':<28} {eff_b_str:>10}  {'+Attn better' if eff_b > 0 else '−Attn hurts' if eff_b < 0 else '≈'}")

            # Interaction Effects
            # C×A: does A help more on HRNet than UNet?
            a_eff_hrnet = avg_dice(ds_results, ["010_A", "011_Full"]) - avg_dice(ds_results, ["000_Base", "001_B"])
            a_eff_unet = avg_dice(ds_results, ["10_UNet_A", "11_UNet_Full"]) - avg_dice(ds_results, ["00_UNet_Base", "01_UNet_B"])
            cxa = a_eff_hrnet - a_eff_unet

            # C×B: does B help more on HRNet than UNet?
            b_eff_hrnet = avg_dice(ds_results, ["001_B", "011_Full"]) - avg_dice(ds_results, ["000_Base", "010_A"])
            b_eff_unet = avg_dice(ds_results, ["01_UNet_B", "11_UNet_Full"]) - avg_dice(ds_results, ["00_UNet_Base", "10_UNet_A"])
            cxb = b_eff_hrnet - b_eff_unet

            # A×B: synergy between A and B
            # (Full - A_only - B_only + Base) — all averaged over both backbones
            ab_synergy_hrnet = (dice_val("hrnet", True, True) - dice_val("hrnet", True, False)
                                - dice_val("hrnet", False, True) + dice_val("hrnet", False, False))
            ab_synergy_unet = (dice_val("unet", True, True) - dice_val("unet", True, False)
                               - dice_val("unet", False, True) + dice_val("unet", False, False))
            ab_synergy = (ab_synergy_hrnet + ab_synergy_unet) / 2.0

            print(f"\n  Interaction Effects (key ablation evidence):")
            print(f"  {'Interaction':<28} {'Delta_Dice':>10}  {'Interpretation'}")
            print(f"  {'-'*75}")
            cxa_str = f"{cxa:+.4f}" if not np.isnan(cxa) else "N/A"
            cxa_int = "Prior benefits MORE from high-res features ✓" if cxa > 0.001 else ("Prior benefits from UNet more ✗" if cxa < -0.001 else "≈ no difference")
            print(f"  {'C×A: Backbone × Prior':<28} {cxa_str:>10}  {cxa_int}")
            cxb_str = f"{cxb:+.4f}" if not np.isnan(cxb) else "N/A"
            cxb_int = "Attn benefits MORE from high-res features ✓" if cxb > 0.001 else ("Attn benefits from UNet more ✗" if cxb < -0.001 else "≈ no difference")
            print(f"  {'C×B: Backbone × Attn':<28} {cxb_str:>10}  {cxb_int}")
            ab_str = f"{ab_synergy:+.4f}" if not np.isnan(ab_synergy) else "N/A"
            ab_int = "Modules synergize ✓" if ab_synergy > 0.001 else ("Modules conflict ✗" if ab_synergy < -0.001 else "≈ additive")
            print(f"  {'A×B: Prior × Attn':<28} {ab_str:>10}  {ab_int}")

        # ======================================================================
        # CSV
        # ======================================================================
        csv_path = os.path.join(RESULTS_BASE, f"ablation_{dataset_name.lower()}_{timestamp}.csv")
        fields = ["Config", "Desc", "backbone", "n_modules", "Params_M", "FLOPs_G", "Fold"] + [f"{m}_mean" for m in METRIC_NAMES] + [f"{m}_std" for m in METRIC_NAMES]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in ds_results:
                row = {k: r.get(k, "") for k in ["Config", "Desc", "backbone", "n_modules", "Params_M", "FLOPs_G", "Fold"]}
                for m in METRIC_NAMES:
                    row[f"{m}_mean"] = fmt_val(r.get(m, np.nan))
                    row[f"{m}_std"] = fmt_val(r.get(m + "_std", np.nan))
                writer.writerow(row)
        print(f"\n  CSV: {csv_path}")

    # ==========================================================================
    # Global Summary (grouped by backbone)
    # ==========================================================================
    print(section_header("FINAL SUMMARY: Best Dice per Dataset", 110))

    # HRNet block
    print(f"\n  {'─'*100}")
    print(f"  HRNet Backbone (Dual-Stream, High-Res)")
    print(f"  {'─'*100}")
    print(f"  {'Config':<20} " + "".join(f"{d:>14}" for d in DATASETS))
    print(f"  {'-'*62}")
    for cfg_name in sorted(ABLATION_CONFIGS):
        if ABLATION_CONFIGS[cfg_name]["backbone"] != "hrnet":
            continue
        vals = []
        for ds in DATASETS:
            key = (ds, cfg_name)
            if key in config_results:
                val = f"{config_results[key]['Dice']:.4f}"
                std = config_results[key].get("Dice_std", 0)
                if isinstance(std, (int, float)) and std > 0:
                    val += f"+-{std:.3f}"
                vals.append(val)
            else:
                vals.append("N/A")
        print(f"  {cfg_name:<20} " + "".join(f"{v:>14}" for v in vals))

    # UNet block
    print(f"\n  {'─'*100}")
    print(f"  UNet Backbone (Encoder-Decoder, Deep Downsample)")
    print(f"  {'─'*100}")
    print(f"  {'Config':<20} " + "".join(f"{d:>14}" for d in DATASETS))
    print(f"  {'-'*62}")
    for cfg_name in sorted(ABLATION_CONFIGS):
        if ABLATION_CONFIGS[cfg_name]["backbone"] != "unet":
            continue
        vals = []
        for ds in DATASETS:
            key = (ds, cfg_name)
            if key in config_results:
                val = f"{config_results[key]['Dice']:.4f}"
                std = config_results[key].get("Dice_std", 0)
                if isinstance(std, (int, float)) and std > 0:
                    val += f"+-{std:.3f}"
                vals.append(val)
            else:
                vals.append("N/A")
        print(f"  {cfg_name:<20} " + "".join(f"{v:>14}" for v in vals))

    print(f"\n  Log saved to: {logger.log_path}")
    logger.close()


if __name__ == "__main__":
    main()
