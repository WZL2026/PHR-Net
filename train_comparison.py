#!/usr/bin/env python3
"""
统一对比实验训练脚本
支持 6 个模型: UNet, ResUNet, DSCNet, MambaVesselNet, PHR_Net, NETracer
3 个数据集: DRIVE, CHASEDB1, STARE

运行方式:
  python train_comparison.py                              # 自动训练所有模型×所有数据集
  python train_comparison.py --model UNet --dataset DRIVE  # 训练单个模型
"""
import argparse
import os
import sys
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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


from models.unet import UNet
from models.resunet import ResUnet
from models.phr_net import PHR_Net
from models.dscnet import DSCNet_pro
# _install_mamba_fallback()
# from models.mamba_vessel_net import MambaVesselNet  # MODEL_IMPORT_LAZY：改为 get_model() 内惰性导入
from models.netracer import NETracer_Net, compute_netracer_loss

from utils.dataset import DRIVEDataset, CHASEDB1Dataset, STAREDataset
from utils.netracer_dataset import NETracerDatasetWrapper
from utils.kfold import create_kfold_dataloaders
from utils.logger import Logger, count_params, section_header

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
CROP_SIZE = 256
BATCH_SIZE = 8
# 学习率：默认沿用统一协议 2e-4；个别模型（如 0.55 M 的 M2U-Net，原文用 1e-3）
# 可用环境变量 PHRNET_LR 覆盖，例如 PHRNET_LR=1e-3
LR = float(os.environ.get("PHRNET_LR", "2e-4"))
ACCUMULATION_STEPS = 1

# NETracer 中心线头的监督目标：
#   "skeleton"（默认）= 官方定义（对骨架做监督）→ 输出是细骨架，不能与血管掩膜算 Dice；
#   "vessel"         = 稠密血管分割统一协议下的公平适配（对血管掩膜做监督）。
# 用环境变量切换： NETRACER_CL_TARGET=vessel python train_comparison.py --model NETracer --dataset DRIVE
NETRACER_CL_TARGET = os.environ.get("NETRACER_CL_TARGET", "skeleton")

MODEL_NAMES = ['PHR_Net', 'NETracer', 'UNet', 'ResUNet', 'DSCNet', 'MambaVesselNet', 'CENet', 'FRUNet', 'M2UNet', 'ERFNet', 'DRIU', 'VMUNet', 'SwinUnet', 'TransUNet', 'UMamba', 'SwinUMamba', 'SerpMamba']
EXT_MODEL_NAMES = ['CENet', 'FRUNet', 'M2UNet', 'ERFNet', 'DRIU', 'VMUNet', 'SwinUnet', 'TransUNet', 'UMamba', 'SwinUMamba', 'SerpMamba']  # 第三方官方仓库模型，统一适配见 baselines_ext/wrappers.py

DATASETS = ['DRIVE', 'CHASEDB1', 'STARE']


# ============================================================================
# 损失函数
# ============================================================================

def calculate_bce_with_fov(logits, targets, fovs):
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
    return (bce_loss * fovs).sum() / (fovs.sum() + 1e-6)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, fovs):
        probs = torch.sigmoid(logits) * fovs
        targets = targets * fovs
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
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


def compute_loss(outputs, masks, fovs, model_name, criterion_ft, criterion_bd,
                 criterion_dice=None):
    if model_name == 'PHR_Net' and isinstance(outputs, tuple) and len(outputs) == 2:
        out_fused, aux_outputs = outputs
        if not isinstance(aux_outputs, list):
            aux_outputs = [aux_outputs]
        loss_fused = (calculate_bce_with_fov(out_fused, masks, fovs)
                      + criterion_ft(out_fused, masks, fovs)
                      + 0.5 * criterion_bd(out_fused, masks, fovs))
        loss_aux = 0.0
        for j, aux_out in enumerate(aux_outputs):
            aux_out_up = F.interpolate(aux_out, size=masks.shape[2:], mode='bilinear', align_corners=False)
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
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        loss = (calculate_bce_with_fov(out, masks, fovs)
                + criterion_ft(out, masks, fovs)
                + 0.5 * criterion_bd(out, masks, fovs))
        pred_out = out
    return loss, pred_out


def get_model(model_name):
    # === EXT BASELINES (baselines_ext) ===
    if model_name in EXT_MODEL_NAMES:
        from baselines_ext.wrappers import EXT_INDEX, build as _build_ext
        if model_name not in EXT_INDEX:
            raise KeyError(f"{model_name} 未在 baselines_ext/wrappers.py 的 BUILDERS 中注册")
        return _build_ext(EXT_INDEX[model_name], in_channels=3, num_classes=1)
    if model_name == 'UNet':            return UNet(in_channels=3, out_channels=1)
    if model_name == 'ResUNet':         return ResUnet(in_channels=3, out_channels=1)
    if model_name == 'DSCNet':          return DSCNet_pro(in_channels=3, out_channels=1)
    if model_name == 'MambaVesselNet':
        # 惰性导入：mamba_ssm 在 import 时会初始化 triton 驱动，登录节点(无 GPU)直接 import 会失败
        # mamba_ssm 真扩展不可用时启用纯 PyTorch 回退（否则下一行 import 会抛 GLIBC 错）
        from baselines_ext.mamba_fallback import install_mamba_fallback as _install_mamba_fallback
        _install_mamba_fallback()
        from models.mamba_vessel_net import MambaVesselNet
        return MambaVesselNet(in_ch=3, out_ch=1)
    if model_name == 'PHR_Net':         return PHR_Net(in_channels=3, out_classes=1)
    if model_name == 'NETracer':        return NETracer_Net(in_channels=3, out_channels=1)
    raise ValueError(f"Unknown model: {model_name}")


def train_epoch(model, train_loader, optimizer, model_name,
                criterion_ft, criterion_bd, criterion_dice):
    model.train()
    epoch_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    n_batches = len(train_loader)

    is_netracer = (model_name == 'NETracer')

    optimizer.zero_grad()
    for batch_idx, batch_data in enumerate(train_loader):
        if is_netracer:
            # NETracer batch: (img, walked_path, cl, node, edge, local_br, is_pos)
            images, walked_paths, cl_gts, node_gts, edge_gts, lb_gts, is_pos = [
                d.to(DEVICE) for d in batch_data]

            # Full 3-output forward
            cl_sig, pt_sig, edge_sig = model.forward_tracing(images, walked_paths)

            # NETracer loss
            loss, loss_dict = compute_netracer_loss(
                cl_sig, pt_sig, edge_sig,
                cl_gts, node_gts, edge_gts, lb_gts,
                positive_mask=is_pos,
            )

            # For Dice monitoring, use standard forward (centreline logits → sigmoid)
            with torch.no_grad():
                cl_logits = model(images)
                pred_prob = torch.sigmoid(cl_logits)
                pred_out = pred_prob
                masks_fov = cl_gts  # use centreline GT as mask for FOV
        else:
            images, masks, fovs = [d.to(DEVICE) for d in batch_data]

            outputs = model(images)
            loss, pred_out = compute_loss(outputs, masks, fovs, model_name,
                                          criterion_ft, criterion_bd, criterion_dice)

        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        epoch_loss += loss.item() * ACCUMULATION_STEPS

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            if is_netracer:
                pred_mask = (pred_out > 0.5).float()
                # Dice on centreline
                total_tp += (pred_mask * cl_gts).sum().item()
                total_fp += (pred_mask * (1 - cl_gts)).sum().item()
                total_fn += ((1 - pred_mask) * cl_gts).sum().item()
            else:
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
    avg_loss = epoch_loss / n_batches
    avg_dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)
    return avg_loss, avg_dice


def val_epoch(model, val_loader, model_name,
              criterion_ft, criterion_bd, criterion_dice):
    """Validation — always uses standard segmentation forward pass for fair comparison."""
    model.eval()
    total_loss = 0.0
    total_tp, total_fp, total_fn = 0, 0, 0
    with torch.no_grad():
        for images, masks, fovs in val_loader:
            images, masks, fovs = images.to(DEVICE), masks.to(DEVICE), fovs.to(DEVICE)
            outputs = model(images)
            loss, pred_out = compute_loss(outputs, masks, fovs, model_name,
                                          criterion_ft, criterion_bd, criterion_dice)
            total_loss += loss.item()
            pred_prob = torch.sigmoid(pred_out) * fovs
            masks_fov = masks * fovs
            pred_mask = (pred_prob > 0.5).float()
            total_tp += (pred_mask * masks_fov).sum().item()
            total_fp += (pred_mask * (1 - masks_fov)).sum().item()
            total_fn += ((1 - pred_mask) * masks_fov).sum().item()
    eps = 1e-6
    avg_loss = total_loss / max(len(val_loader), 1)
    avg_dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)
    return avg_loss, avg_dice


def train_single(model_name, dataset_name, batch_size, epochs):
    """训练单个模型在单个数据集上。返回最佳验证 Dice。"""
    crop_info = f"  Crop size:   {CROP_SIZE}x{CROP_SIZE}"
    base_dir = f"../Dataset/{dataset_name}"

    criterion_ft = FocalTverskyLoss().to(DEVICE)
    criterion_bd = FastSoftBoundaryLoss().to(DEVICE)
    criterion_dice = DiceLoss().to(DEVICE)

    # -- DRIVE --
    if dataset_name == 'DRIVE':
        train_dataset = DRIVEDataset(base_dir, mode='training', crop_size=CROP_SIZE)
        val_dataset = DRIVEDataset(base_dir, mode='test')

        if model_name == 'NETracer':
            train_dataset = NETracerDatasetWrapper(train_dataset, patch_size=CROP_SIZE, neg_ratio=0.0, centreline_target=NETRACER_CL_TARGET)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
            worker_init_fn=worker_init_fn)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
            worker_init_fn=worker_init_fn)

        model = get_model(model_name).to(DEVICE)
        total_m, train_m = count_params(model)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        loss_desc = "L_cl+L_node+L_edge+L_conn" if model_name == 'NETracer' else "BCE+FocalTversky+0.5*Boundary"
        print(section_header(f"{model_name} on {dataset_name}  |  {total_m:.2f}M params  |  {DEVICE}"))
        print(f"  Batch: {batch_size}  |  Epochs: {epochs}  |  Samples: {len(train_dataset)}  |  {crop_info}")
        print(f"  Loss: {loss_desc}  |  Optimizer: AdamW(lr={LR:.1e})+CosAnnealing")
        print("=" * 80)

        best_dice = 0.0
        best_epoch = 0
        os.makedirs(f"weights/Comparison/{dataset_name}", exist_ok=True)

        for epoch in range(epochs):
            train_loss, train_dice = train_epoch(
                model, train_loader, optimizer, model_name,
                criterion_ft, criterion_bd, criterion_dice)
            val_loss, val_dice = val_epoch(
                model, val_loader, model_name,
                criterion_ft, criterion_bd, criterion_dice)
            scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                best_epoch = epoch + 1
                torch.save(model.state_dict(),
                           f"weights/Comparison/{dataset_name}/best_{model_name.lower()}_{dataset_name.lower()}.pth")

            note = " *BEST" if improved else ""
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
                  f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, lr={lr:.2e}{note}")

        print(f"\n  => Best Val Dice: {best_dice:.4f} @ epoch {best_epoch}")
        del model, train_loader, val_loader, train_dataset, val_dataset

    # -- CHASEDB1 / STARE --
    else:
        DatasetClass = CHASEDB1Dataset if dataset_name == 'CHASEDB1' else STAREDataset
        train_dataset = DatasetClass(base_dir, mode='training', crop_size=CROP_SIZE)
        val_dataset = DatasetClass(base_dir, mode='test')

        if model_name == 'NETracer':
            train_dataset = NETracerDatasetWrapper(train_dataset, patch_size=CROP_SIZE, neg_ratio=0.0)

        os.makedirs(f"weights/Comparison/{dataset_name}", exist_ok=True)

        sample_model = get_model(model_name)
        total_m, train_m = count_params(sample_model)
        del sample_model

        loss_desc = "L_cl+L_node+L_edge+L_conn" if model_name == 'NETracer' else "BCE+FocalTversky+0.5*Boundary"
        print(section_header(f"{model_name} on {dataset_name} (5-Fold CV)  |  {total_m:.2f}M params  |  {DEVICE}"))
        print(f"  Batch: {batch_size}  |  Epochs: {epochs}  |  Samples: {len(train_dataset)}  |  {crop_info}")
        print(f"  Loss: {loss_desc}  |  Optimizer: AdamW(lr={LR:.1e})+CosAnnealing")
        print("=" * 80)

        fold_results = []
        for train_loader, val_loader, fold_idx in create_kfold_dataloaders(
                train_dataset, val_dataset, batch_size=batch_size,
                worker_init_fn=worker_init_fn):
            set_seed(42)
            print(f"\n  Fold {fold_idx+1}/5:")

            model = get_model(model_name).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1.5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            best_fold_dice = 0.0
            best_fold_epoch = 0
            for epoch in range(epochs):
                train_loss, train_dice = train_epoch(
                    model, train_loader, optimizer, model_name,
                    criterion_ft, criterion_bd, criterion_dice)
                val_loss, val_dice = val_epoch(
                    model, val_loader, model_name,
                    criterion_ft, criterion_bd, criterion_dice)
                scheduler.step()

                if val_dice > best_fold_dice:
                    best_fold_dice = val_dice
                    best_fold_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               f"weights/Comparison/{dataset_name}/best_{model_name.lower()}_{dataset_name.lower()}_fold{fold_idx+1}.pth")

                lr = optimizer.param_groups[0]['lr']
                improved = val_dice > best_fold_dice
                note = " *BEST" if improved else ""
                print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_dice={train_dice:.4f}, "
                      f"val_loss={val_loss:.4f}, val_dice={val_dice:.4f}, lr={lr:.2e}{note}")

            fold_results.append(best_fold_dice)
            print(f"    => Best: {best_fold_dice:.4f} @ epoch {best_fold_epoch}")
            del model
            torch.cuda.empty_cache()

        best_dice = np.mean(fold_results)
        folds_str = ", ".join(f"{v:.4f}" for v in fold_results)
        print(f"  => 5-Fold: {best_dice:.4f} +- {np.std(fold_results):.4f}  [{folds_str}]")
        del train_dataset, val_dataset

    torch.cuda.empty_cache()
    return best_dice


def main():
    parser = argparse.ArgumentParser(description="Comparison Training")
    parser.add_argument("--model", type=str, default=None, choices=MODEL_NAMES,
                        help="Train a single model (default: all models)")
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS,
                        help="Train on a single dataset (default: all datasets)")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODEL_NAMES
    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    model_tag = args.model or "AllModels"
    data_tag = args.dataset or "AllDatasets"
    logger = Logger(log_dir="logs", prefix=f"train_{model_tag}_{data_tag}")
    sys.stdout = logger

    print(f"\n{'#'*100}")
    print(f"  Comparison Training  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {models_to_run}")
    print(f"  Datasets: {datasets_to_run}")
    print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")
    print(f"  Log: {logger.log_path}")
    print(f"{'#'*100}")

    results = []
    t_start = datetime.now()
    for model_name in models_to_run:
        for dataset_name in datasets_to_run:
            try:
                best = train_single(model_name, dataset_name, BATCH_SIZE, EPOCHS)
                results.append((model_name, dataset_name, best))
            except Exception as e:
                print(f"\n  [FAIL] {model_name}/{dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((model_name, dataset_name, None))

    elapsed = (datetime.now() - t_start).total_seconds() / 3600.0

    if len(results) > 0:
        print(section_header("TRAINING SUMMARY", 80))
        print(f"  {'Model':<18} {'Dataset':<12} {'Best Dice':<12}")
        for model_name, dataset_name, best in results:
            best_str = f"{best:.4f}" if best is not None else "FAILED"
            print(f"  {model_name:<18} {dataset_name:<12} {best_str:<12}")
        print(f"\n  Total time: {elapsed:.2f} hours")

    logger.close()


if __name__ == '__main__':
    main()
