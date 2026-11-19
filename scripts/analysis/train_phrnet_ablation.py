#!/usr/bin/env python3
"""
PHR-Net 全排列消融实验训练脚本
  3 模块全排列: C(backbone) × A(prior) × B(attention) = 8 configs
  模块定义:
    C = HRNet (高分辨率双流) vs UNet (深度下采样编解码)
    A = DPE+DPSS+DSSB (拉普拉斯先验 + 先验放大器 + 双域选择性块)
    B = LKA+RSFH (大核注意力 + 条带池化融合)
    P = PGCA (prior-gated channel attention, HRNet only)

  运行方式:
    python train_phrnet_ablation.py                         # 全部 8 configs
    python train_phrnet_ablation.py --backbone unet          # 只跑 UNet (4 新增)
    python train_phrnet_ablation.py --backbone hrnet         # 只跑 HRNet (4 已有)
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
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import csv
import gc
import sys
from datetime import datetime

# ==============================================================================
# Argument parsing (simple, before Logger hijacks stdout)
# ==============================================================================
BACKBONE_FILTER = "all"
PGCA_FILTER = None   # "on" = only PGCA configs, "off" = only non-PGCA, None = all
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
# 消融实验配置 — 3 模块全排列 (C × A × B) = 8 configs
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

# Apply --pgca filter (HRNet-only: "on" = PGCA enabled, "off" = plain residual)
if PGCA_FILTER is not None:
    ABLATION_CONFIGS = {
        k: v for k, v in ABLATION_CONFIGS.items()
        if v.get("backbone") == "hrnet" and v.get("use_pgca") == (PGCA_FILTER == "on")
    }

# ==============================================================================
# 训练超参数
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
BATCH_SIZE = 8
ACCUMULATION_STEPS = 1
CROP_SIZE = 256
DATASETS = ["DRIVE", "CHASEDB1", "STARE"]

WEIGHTS_BASE = "weights/Ablation"
RESULTS_BASE = "results/phrnet"
os.makedirs(WEIGHTS_BASE, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)


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


from models.phr_net import PHR_Net
from utils.dataset import DRIVEDataset, CHASEDB1Dataset, STAREDataset
from utils.kfold import create_kfold_dataloaders
from utils.logger import Logger, count_params, section_header

try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


# ==============================================================================
# 模型复杂度分析
# ==============================================================================
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


# ==============================================================================
# 损失函数
# ==============================================================================
def calculate_bce_with_fov(logits, targets, fovs):
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    return (bce_loss * fovs).sum() / (fovs.sum() + 1e-6)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, fovs):
        probs = torch.sigmoid(logits) * fovs
        targets = targets * fovs
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=4 / 3, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets, fovs):
        probs = torch.sigmoid(logits) * fovs
        targets = targets * fovs
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky) ** self.gamma


class FastSoftBoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def extract_boundary(self, mask):
        dilated = F.max_pool2d(mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        eroded = -F.max_pool2d(-mask, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        return dilated - eroded

    def forward(self, logits, targets, fovs):
        probs = torch.sigmoid(logits) * fovs
        targets = targets * fovs
        gt_boundary = self.extract_boundary(targets)
        pred_boundary = self.extract_boundary(probs)
        return F.mse_loss(pred_boundary, gt_boundary)


# ==============================================================================
# 训练循环
# ==============================================================================
def train_model(model, train_loader, val_loader, dataset_name, ablation_name, fold_idx=None):
    desc = ABLATION_CONFIGS[ablation_name]["desc"]

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1.5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    criterion_ft = FocalTverskyLoss()
    criterion_dice = DiceLoss()
    criterion_bd = FastSoftBoundaryLoss()

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        # -- train --
        model.train()
        epoch_loss = 0.0
        total_tp, total_fp, total_fn = 0, 0, 0
        n_batches = len(train_loader)

        optimizer.zero_grad()
        for i, (images, masks, fovs) in enumerate(train_loader):
            images, masks, fovs = images.to(DEVICE), masks.to(DEVICE), fovs.to(DEVICE)
            output = model(images)

            if isinstance(output, tuple) and len(output) == 2:
                out_fused, aux_outputs = output
                if not isinstance(aux_outputs, list):
                    aux_outputs = [aux_outputs]
                loss_fused = (calculate_bce_with_fov(out_fused, masks, fovs)
                              + criterion_ft(out_fused, masks, fovs)
                              + 0.5 * criterion_bd(out_fused, masks, fovs))
                loss_aux = 0.0
                for j, aux_out in enumerate(aux_outputs):
                    aux_out_up = F.interpolate(aux_out, size=masks.shape[2:], mode="bilinear", align_corners=False)
                    if j == 0:
                        loss_aux += (calculate_bce_with_fov(aux_out_up, masks, fovs)
                                     + criterion_ft(aux_out_up, masks, fovs)
                                     + 0.5 * criterion_bd(aux_out_up, masks, fovs))
                    else:
                        loss_aux += (calculate_bce_with_fov(aux_out_up, masks, fovs)
                                     + criterion_dice(aux_out_up, masks, fovs))
                loss = loss_fused + loss_aux
                pred_out = out_fused
            else:
                out = output[0] if isinstance(output, (tuple, list)) else output
                loss = (calculate_bce_with_fov(out, masks, fovs)
                        + criterion_ft(out, masks, fovs)
                        + 0.5 * criterion_bd(out, masks, fovs))
                pred_out = out

            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            epoch_loss += loss.item() * ACCUMULATION_STEPS

            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                pred_prob = torch.sigmoid(pred_out) * fovs
                masks_fov = masks * fovs
                pred_mask = (pred_prob > 0.5).float()
                total_tp += (pred_mask * masks_fov).sum().item()
                total_fp += (pred_mask * (1 - masks_fov)).sum().item()
                total_fn += ((1 - pred_mask) * masks_fov).sum().item()

        if n_batches % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        eps = 1e-6
        train_loss_avg = epoch_loss / n_batches
        train_dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)

        # -- val --
        model.eval()
        val_loss = 0.0
        val_tp, val_fp, val_fn = 0, 0, 0
        with torch.no_grad():
            for images, masks, fovs in val_loader:
                images, masks, fovs = images.to(DEVICE), masks.to(DEVICE), fovs.to(DEVICE)
                output = model(images)
                if isinstance(output, tuple) and len(output) == 2:
                    out_fused, aux_outputs = output
                    if not isinstance(aux_outputs, list):
                        aux_outputs = [aux_outputs]
                    vloss_fused = (calculate_bce_with_fov(out_fused, masks, fovs)
                                   + criterion_ft(out_fused, masks, fovs)
                                   + 0.5 * criterion_bd(out_fused, masks, fovs))
                    vloss_aux = 0.0
                    for j, aux_out in enumerate(aux_outputs):
                        aux_out_up = F.interpolate(aux_out, size=masks.shape[2:], mode="bilinear", align_corners=False)
                        if j == 0:
                            vloss_aux += (calculate_bce_with_fov(aux_out_up, masks, fovs)
                                          + criterion_ft(aux_out_up, masks, fovs)
                                          + 0.5 * criterion_bd(aux_out_up, masks, fovs))
                        else:
                            vloss_aux += (calculate_bce_with_fov(aux_out_up, masks, fovs)
                                          + criterion_dice(aux_out_up, masks, fovs))
                    vloss = vloss_fused + vloss_aux
                    vpred = out_fused
                else:
                    out = output[0] if isinstance(output, (tuple, list)) else output
                    vloss = (calculate_bce_with_fov(out, masks, fovs)
                             + criterion_ft(out, masks, fovs)
                             + 0.5 * criterion_bd(out, masks, fovs))
                    vpred = out
                val_loss += vloss.item()
                pred_prob = torch.sigmoid(vpred) * fovs
                masks_fov = masks * fovs
                pred_mask = (pred_prob > 0.5).float()
                val_tp += (pred_mask * masks_fov).sum().item()
                val_fp += (pred_mask * (1 - masks_fov)).sum().item()
                val_fn += ((1 - pred_mask) * masks_fov).sum().item()

        val_loss_avg = val_loss / max(len(val_loader), 1)
        val_dice = (2 * val_tp + eps) / (2 * val_tp + val_fp + val_fn + eps)

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            best_epoch = epoch + 1
            save_dir = os.path.join(WEIGHTS_BASE, dataset_name, ablation_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"best_fold{fold_idx+1}.pth" if fold_idx is not None else "best.pth")
            torch.save(model.state_dict(), save_path)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        note = " *BEST" if improved else ""
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss_avg:.4f}, train_dice={train_dice:.4f}, "
              f"val_loss={val_loss_avg:.4f}, val_dice={val_dice:.4f}, lr={lr:.2e}{note}")

    print(f"  Best: Dice={best_dice:.4f} @ epoch {best_epoch}")
    return best_dice


# ==============================================================================
# 主函数
# ==============================================================================
def main():
    ablation_names = list(ABLATION_CONFIGS.keys())

    logger = Logger(log_dir="logs", prefix="train_ablation")
    sys.stdout = logger

    print(f"\n{'#'*100}")
    print(f"  PHR-Net 3-Factor Ablation — {len(ablation_names)} configs x {len(DATASETS)} datasets")
    if BACKBONE_FILTER != "all":
        print(f"  Backbone filter: {BACKBONE_FILTER}")
    print(f"  Modules: C(backbone) × A(prior) × B(attention)")
    print(f"  Datasets: {DATASETS}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Crop: {CROP_SIZE}")
    print(f"  Weights -> {WEIGHTS_BASE}/<dataset>/<config>/")
    print(f"  Log: {logger.log_path}")
    print(f"{'#'*100}")

    # Pre-compute complexity
    print(section_header("Model Complexity Summary"))
    complexity_cache = {}
    for ablation_name in ablation_names:
        cfg = ABLATION_CONFIGS[ablation_name]
        model_kwargs = {k: v for k, v in cfg.items() if k not in ("desc",)}
        sample_model = PHR_Net(in_channels=3, out_classes=1, **model_kwargs).to(DEVICE)
        params_m, flops_g = measure_complexity(sample_model)
        complexity_cache[ablation_name] = (params_m, flops_g)
        print(f"  {ablation_name:<10}  Params: {params_m:.2f}M  FLOPs: {flops_g:.2f}G  |  {cfg['desc']}")
        del sample_model
        torch.cuda.empty_cache()

    # Train
    all_results = []
    for dataset_name in DATASETS:
        base_dir = f"../Dataset/{dataset_name}"
        print(section_header(f"Dataset: {dataset_name}  |  {len(ablation_names)} configs"))

        for ablation_name in ablation_names:

            # 每个 Config 启动前强制重置随机种子, 确保公平起跑线
            set_seed(42)

            cfg = ABLATION_CONFIGS[ablation_name]
            desc = cfg["desc"]
            model_kwargs = {k: v for k, v in cfg.items() if k not in ("desc",)}
            params_m, flops_g = complexity_cache[ablation_name]

            print(f"\n  [{ablation_name}] {desc}  |  {params_m:.2f}M / {flops_g:.2f}G FLOPs")

            try:
                if dataset_name == "DRIVE":
                    train_dataset = DRIVEDataset(base_dir, mode="training", crop_size=CROP_SIZE)
                    val_dataset = DRIVEDataset(base_dir, mode="test")
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

                    model = PHR_Net(in_channels=3, out_classes=1, **model_kwargs).to(DEVICE)
                    best_dice = train_model(model, train_loader, val_loader, dataset_name, ablation_name)
                    all_results.append({
                        "Dataset": dataset_name, "Config": ablation_name,
                        "Desc": desc, "Params_M": f"{params_m:.2f}", "FLOPs_G": f"{flops_g:.2f}",
                        "Fold": "—", "BestDice": f"{best_dice:.4f}",
                    })
                    del model, train_loader, val_loader, train_dataset, val_dataset

                else:
                    DatasetClass = CHASEDB1Dataset if dataset_name == "CHASEDB1" else STAREDataset
                    train_dataset = DatasetClass(base_dir, mode="training", crop_size=CROP_SIZE)
                    val_dataset = DatasetClass(base_dir, mode="test")

                    fold_results = []
                    for train_loader, val_loader, fold_idx in create_kfold_dataloaders(train_dataset, val_dataset, batch_size=BATCH_SIZE, worker_init_fn=worker_init_fn):
                        set_seed(42)
                        model = PHR_Net(in_channels=3, out_classes=1, **model_kwargs).to(DEVICE)
                        best_dice = train_model(model, train_loader, val_loader, dataset_name, ablation_name, fold_idx)
                        fold_results.append(best_dice)
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()

                    avg_dice = np.mean(fold_results)
                    std_dice = np.std(fold_results)
                    folds_str = ", ".join(f"{v:.4f}" for v in fold_results)
                    print(f"  5-Fold: [{folds_str}]  =>  {avg_dice:.4f} +- {std_dice:.4f}")
                    all_results.append({
                        "Dataset": dataset_name, "Config": ablation_name,
                        "Desc": desc, "Params_M": f"{params_m:.2f}", "FLOPs_G": f"{flops_g:.2f}",
                        "Fold": "5-fold", "BestDice": f"{avg_dice:.4f} +- {std_dice:.4f}",
                        "PerFold": str([f"{v:.4f}" for v in fold_results]),
                    })
                    del train_dataset, val_dataset

                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"[SKIP] {dataset_name}/{ablation_name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "Dataset": dataset_name, "Config": ablation_name,
                    "Desc": desc, "Params_M": f"{params_m:.2f}", "FLOPs_G": f"{flops_g:.2f}",
                    "Fold": "—", "BestDice": f"ERROR: {e}",
                })

    # Summary — grouped by backbone then n_modules
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(section_header("ABLATION TRAINING SUMMARY", 100))

    # Group by backbone
    by_backbone = {}
    for r in all_results:
        cfg = ABLATION_CONFIGS.get(r["Config"], {})
        bb = cfg.get("backbone", "unknown")
        by_backbone.setdefault(bb, []).append(r)

    for bb in ("hrnet", "unet"):
        if bb not in by_backbone:
            continue
        bb_label = {"hrnet": "HRNet (Dual-Stream, High-Res)", "unet": "UNet (Encoder-Decoder, Deep Downsample)"}
        print(f"\n  {'='*60}")
        print(f"  {bb_label.get(bb, bb)}")
        print(f"  {'='*60}")
        by_n = {}
        for r in by_backbone[bb]:
            n = ABLATION_CONFIGS.get(r["Config"], {}).get("n_modules", -1)
            by_n.setdefault(n, []).append(r)
        for n_mod in sorted(by_n):
            group_label = {0: "Baseline", 1: "+A or +B (single module)", 2: "Full (A+B)"}
            print(f"\n  {group_label.get(n_mod, str(n_mod))}:")
            for r in by_n[n_mod]:
                print(f"    {r['Config']:<18}  {r['Params_M']:>5}M  {r['FLOPs_G']:>5}G  "
                      f"{r['Fold']:<10}  Dice={r['BestDice']}")

    csv_path = os.path.join(RESULTS_BASE, f"ablation_summary_{timestamp}.csv")
    fields = ["Dataset", "Config", "Desc", "Params_M", "FLOPs_G", "Fold", "BestDice", "PerFold"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"\n  CSV: {csv_path}")
    print(f"  Log: {logger.log_path}")
    logger.close()


if __name__ == "__main__":
    main()
