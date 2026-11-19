#!/usr/bin/env python3
"""
统一对比实验测试脚本
支持 6 个模型: UNet, ResUNet, DSCNet, MambaVesselNet, PHR_Net, NETracer
3 个数据集: DRIVE, CHASEDB1, STARE
输出 8 项指标: Dice, IoU, Precision, Recall, Specificity, FPR, HD95, clDice

运行方式:
  python test_comparison.py                              # 自动测试所有已训练的模型x所有数据集
  python test_comparison.py --model UNet --dataset DRIVE  # 测试单个模型
  python test_comparison.py --visualize                   # 同时输出每张图的预测可视化
"""
import argparse
import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

from models.unet import UNet
from models.resunet import ResUnet
from models.phr_net import PHR_Net
from models.dscnet import DSCNet_pro
# # mamba_ssm 真扩展不可用时启用垫片/回退（训练脚本有，测试脚本原来漏了 ->
# # 训练能过但测试导入 mamba_ssm 失败，报 triton.set_allocator 之类错误）
# from baselines_ext.mamba_fallback import install_mamba_fallback as _install_mamba_fallback
# _install_mamba_fallback()
# from models.mamba_vessel_net import MambaVesselNet  # MODEL_IMPORT_LAZY：改为 get_model() 内惰性导入
from models.netracer import NETracer_Net
from utils.dataset import DRIVEDataset, CHASEDB1Dataset, STAREDataset
from utils.metrics import compute_hd95, compute_cldice, sliding_window_inference
from utils.kfold import create_kfold_dataloaders
from utils.logger import Logger, count_params, section_header

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = ['PHR_Net', 'NETracer', 'UNet', 'ResUNet', 'DSCNet', 'MambaVesselNet', 'CENet', 'FRUNet', 'M2UNet', 'ERFNet', 'DRIU', 'VMUNet', 'SwinUnet', 'TransUNet', 'UMamba', 'SwinUMamba', 'SerpMamba']
EXT_MODEL_NAMES = ['CENet', 'FRUNet', 'M2UNet', 'ERFNet', 'DRIU', 'VMUNet', 'SwinUnet', 'TransUNet', 'UMamba', 'SwinUMamba', 'SerpMamba']  # 第三方官方仓库模型，统一适配见 baselines_ext/wrappers.py

DATASETS = ['DRIVE', 'CHASEDB1', 'STARE']
METRIC_NAMES = ["Dice", "IoU", "Precision", "Recall", "Spec", "FPR", "HD95", "clDice"]

# --- Visualization ---
VIS_DIR = "results/visualizations"
# Dataset denormalization params (from utils/dataset.py)
_DS_MEAN = np.array([0.215, 0.215, 0.215], dtype=np.float32)
_DS_STD  = np.array([0.151, 0.151, 0.151], dtype=np.float32)


def denormalize_image(img_tensor):
    """Denormalize a CHW tensor to HWC numpy [0,1] RGB image."""
    img = img_tensor.cpu().numpy()
    img = img * _DS_STD[:, None, None] + _DS_MEAN[:, None, None]
    img = np.clip(img.transpose(1, 2, 0), 0, 1)
    return img


def save_visualization_figures(model, test_loader, dataset_name, model_name, save_dir):
    """Generate and save per-image prediction visualization figures.

    Each figure contains 4 panels in a 1×4 layout:
      Original Image  |  Ground Truth  |  Prediction (Heatmap)  |  Binary Prediction

    Args:
        model: loaded segmentation model in eval mode
        test_loader: DataLoader yielding (images, masks, fovs)
        dataset_name: str, e.g. "DRIVE"
        model_name: str, e.g. "PHR_Net"
        save_dir: str, output directory path
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for img_idx, (images, masks, fovs) in enumerate(test_loader):
            images_gpu = images.to(DEVICE)
            masks_np = masks.squeeze().cpu().numpy()
            fov_np = fovs.squeeze().cpu().numpy()

            # --- Inference ---
            pred_np = sliding_window_inference(
                model, images_gpu, window_size=(512, 512), stride=(256, 256),
                device=DEVICE)
            pred_prob = pred_np * fov_np
            pred_bin = (pred_prob > 0.5).astype(np.float32)

            # --- Per-image metrics ---
            gt_fov = masks_np * fov_np
            eps = 1e-6
            tp = (pred_bin * gt_fov).sum()
            fp = (pred_bin * (1 - gt_fov)).sum()
            fn = ((1 - pred_bin) * gt_fov).sum()
            dice_img = (2 * tp + eps) / (2 * tp + fp + fn + eps)
            iou_img = (tp + eps) / (tp + fp + fn + eps)

            # --- Denormalize original image ---
            orig_rgb = denormalize_image(images.squeeze(0))

            # --- Build figure ---
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
            fig.suptitle(
                f"{model_name}  |  {dataset_name}  |  Image #{img_idx + 1}  |  "
                f"Dice={dice_img:.4f}  IoU={iou_img:.4f}",
                fontsize=13, fontweight="bold", y=0.98)

            # Panel 1: Original Image
            axes[0].imshow(orig_rgb)
            axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            # Panel 2: Ground Truth
            axes[1].imshow(orig_rgb)
            gt_overlay = np.zeros((*gt_fov.shape, 4), dtype=np.float32)
            gt_overlay[gt_fov > 0.5] = [0.0, 1.0, 0.0, 0.55]  # translucent green
            axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold")
            axes[1].axis("off")

            # Panel 3: Prediction Probability Heatmap
            im3 = axes[2].imshow(pred_prob, cmap="hot", vmin=0, vmax=1)
            axes[2].set_title("Prediction (Probability)", fontsize=11, fontweight="bold")
            axes[2].axis("off")
            cbar = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            cbar.set_label("Probability", fontsize=8)

            # Panel 4: Binary Prediction (green overlay)
            axes[3].imshow(orig_rgb)
            pred_overlay = np.zeros((*pred_bin.shape, 4), dtype=np.float32)
            pred_overlay[pred_bin > 0.5] = [0.0, 1.0, 0.0, 0.55]
            axes[3].imshow(pred_overlay)
            axes[3].set_title("Binary Prediction", fontsize=11, fontweight="bold")
            axes[3].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.94])
            save_path = os.path.join(
                save_dir,
                f"{dataset_name}_{model_name}_img{img_idx + 1:02d}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight",
                       facecolor="white", edgecolor="none")
            plt.close(fig)

    print(f"  -> Visualizations saved to: {save_dir}")


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
        from models.mamba_vessel_net import MambaVesselNet
        return MambaVesselNet(in_ch=3, out_ch=1)
    if model_name == 'PHR_Net':         return PHR_Net(in_channels=3, out_classes=1)
    if model_name == 'NETracer':        return NETracer_Net(in_channels=3, out_channels=1)
    raise ValueError(f"Unknown model: {model_name}")


def get_param_count(model_name):
    if model_name not in _PARAM_CACHE:
        m = get_model(model_name)
        total_m, _ = count_params(m)
        _PARAM_CACHE[model_name] = f"{total_m:.2f}M"
        del m
    return _PARAM_CACHE[model_name]


_PARAM_CACHE = {}


def evaluate_model(model, test_loader, return_per_image=False):
    """Evaluate model on test_loader.

    Args:
        model: segmentation model
        test_loader: DataLoader
        return_per_image: if True, also return per-image metric arrays
    Returns:
        if return_per_image: (aggregated_metrics, per_image_dict)
        else: aggregated_metrics list
    """
    model.eval()
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_hd95, hd95_count, total_cldice = 0.0, 0, 0.0
    per_image = {"Dice": [], "IoU": [], "Precision": [], "Recall": [],
                 "Spec": [], "FPR": [], "HD95": [], "clDice": []}

    eps = 1e-6
    with torch.no_grad():
        for images, masks, fovs in test_loader:
            images, masks, fovs = images.to(DEVICE), masks.to(DEVICE), fovs.to(DEVICE)
            pred_np = sliding_window_inference(
                model, images, window_size=(512, 512), stride=(256, 256),
                device=DEVICE)
            pred_mask = (pred_np > 0.5).astype(np.float32)

            fov_np = fovs.cpu().numpy()[0, 0]
            pred_mask = pred_mask * fov_np
            masks_np = masks.cpu().numpy()[0, 0] * fov_np

            # Per-image metrics
            if return_per_image:
                tp = (pred_mask * masks_np).sum()
                fp = (pred_mask * (1 - masks_np)).sum()
                fn = ((1 - pred_mask) * masks_np).sum()
                tn = ((1 - pred_mask) * (1 - masks_np)).sum()
                pid = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                pio = (tp + eps) / (tp + fp + fn + eps)
                ppr = (tp + eps) / (tp + fp + eps)
                pre = (tp + eps) / (tp + fn + eps)
                psp = (tn + eps) / (tn + fp + eps)
                pfp = (fp + eps) / (fp + tn + eps)
                phd = compute_hd95(pred_mask, masks_np)
                pcl = compute_cldice(pred_mask, masks_np)
                per_image["Dice"].append(float(pid))
                per_image["IoU"].append(float(pio))
                per_image["Precision"].append(float(ppr))
                per_image["Recall"].append(float(pre))
                per_image["Spec"].append(float(psp))
                per_image["FPR"].append(float(pfp))
                per_image["HD95"].append(float(phd) if not np.isnan(phd) else np.nan)
                per_image["clDice"].append(float(pcl))

            total_tp += (pred_mask * masks_np).sum()
            total_fp += (pred_mask * (1 - masks_np)).sum()
            total_fn += ((1 - pred_mask) * masks_np).sum()
            total_tn += ((1 - pred_mask) * (1 - masks_np)).sum()

            hd95 = compute_hd95(pred_mask, masks_np)
            if not np.isnan(hd95):
                total_hd95 += hd95
                hd95_count += 1
            total_cldice += compute_cldice(pred_mask, masks_np)

    dice = (2 * total_tp + eps) / (2 * total_tp + total_fp + total_fn + eps)
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)
    spec = (total_tn + eps) / (total_tn + total_fp + eps)
    fpr = (total_fp + eps) / (total_fp + total_tn + eps)
    avg_hd95 = total_hd95 / hd95_count if hd95_count > 0 else np.nan
    avg_cldice = total_cldice / len(test_loader.dataset)

    agg = [dice, iou, precision, recall, spec, fpr, avg_hd95, avg_cldice]
    return (agg, per_image) if return_per_image else agg



def load_weights_flexible(model, path, device="cpu"):
    """加载权重并**明确报告命中张量数**：自动尝试无前缀与 net./module. 等前缀。

    为什么需要：第三方官方权重（例如 M2U-Net 作者发布的 M2UNetDRIVE.pth）键名不带适配层
    前缀，而我们的模型被 ToLogits 包了一层（键为 net.*）。原写法
    `model.load_state_dict(torch.load(path), strict=False)` 在键名对不上时会**静默命中 0 个
    张量**，等于拿随机初始化的模型去评估（结果会莫名很低且无任何提示）。
    """
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if isinstance(sd.get(k), dict):
                sd = sd[k]
                break
    if not isinstance(sd, dict):
        print(f"  [WARN] {os.path.basename(path)}: 无法识别的权重格式（{type(sd).__name__}）")
        return 0
    msd = model.state_dict()
    clean = {k.replace("module.", ""): v for k, v in sd.items() if hasattr(v, "shape")}
    best_hit, best = 0, {}
    for pre in ("", "net.", "net.net.", "model.", "swin_unet."):
        cand = {pre + k: v for k, v in clean.items()}
        hit = {k: v for k, v in cand.items()
               if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
        if len(hit) > best_hit:
            best_hit, best = len(hit), hit
    if best_hit == 0:
        print(f"  [WARN] {os.path.basename(path)}: 没有任何张量匹配模型 -> 随机初始化！"
              f"（请确认权重与模型是否对应）")
        return 0
    model.load_state_dict(best, strict=False)
    print(f"  [weights] {os.path.basename(path)}: 命中 {best_hit}/{len(msd)} 个张量"
          f"（strict=False）")
    return best_hit


def test_single(model_name, dataset_name, save_per_image=False, per_image_dir="results/per_image",
                visualize=False):
    base_dir = f"../Dataset/{dataset_name}"

    print(section_header(f"{model_name} on {dataset_name}"))

    if dataset_name == 'DRIVE':
        test_dataset = DRIVEDataset(base_dir, mode='test')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
            worker_init_fn=worker_init_fn)

        model = get_model(model_name).to(DEVICE)
        weight_path = f"weights/Comparison/{dataset_name}/best_{model_name.lower()}_{dataset_name.lower()}.pth"
        per_image = None
        if os.path.exists(weight_path):
            load_weights_flexible(model, weight_path, DEVICE)
            result = evaluate_model(model, test_loader, return_per_image=save_per_image)
            if save_per_image:
                metrics, per_image = result
            else:
                metrics = result
            print(f"  Samples: {len(test_dataset)}")
            print(f"  {'Dice':>8s} {'IoU':>8s} {'Precision':>10s} {'Recall':>8s} "
                  f"{'Spec':>8s} {'FPR':>8s} {'HD95':>8s} {'clDice':>8s}")
            print(f"  {metrics[0]:8.4f} {metrics[1]:8.4f} {metrics[2]:10.4f} {metrics[3]:8.4f} "
                  f"{metrics[4]:8.4f} {metrics[5]:8.4f} {metrics[6]:8.4f} {metrics[7]:8.4f}")

            # --- Visualization ---
            if visualize:
                vis_dir = os.path.join(VIS_DIR, dataset_name, model_name)
                test_loader_v = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                    worker_init_fn=worker_init_fn)
                save_visualization_figures(model, test_loader_v, dataset_name, model_name, vis_dir)
                del test_loader_v
        else:
            print(f"  [WARN] Weight not found: {weight_path}")
            metrics = [np.nan] * 8
        del model, test_loader, test_dataset
        torch.cuda.empty_cache()

        # Save per-image CSV for t-test
        if save_per_image and per_image:
            os.makedirs(per_image_dir, exist_ok=True)
            csv_path = f"{per_image_dir}/{model_name}_{dataset_name}_per_image.csv"
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["ImageIdx"] + METRIC_NAMES)
                writer.writeheader()
                for i in range(len(per_image["Dice"])):
                    writer.writerow({
                        "ImageIdx": i,
                        **{name: per_image[name][i] for name in METRIC_NAMES}
                    })
            print(f"  -> Per-image saved: {csv_path}")

        return metrics, [0.0] * 8, "-"

    else:
        DatasetClass = CHASEDB1Dataset if dataset_name == 'CHASEDB1' else STAREDataset
        train_dataset = DatasetClass(base_dir, mode='training')
        val_dataset = DatasetClass(base_dir, mode='test')
        fold_metrics = []
        missing_folds = []
        all_fold_per_image = []  # store per-image data per fold

        print(f"  {'Fold':<8s} {'Dice':>8s} {'IoU':>8s} {'Precision':>10s} {'Recall':>8s} "
              f"{'Spec':>8s} {'FPR':>8s} {'HD95':>8s} {'clDice':>8s}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for _, test_loader, fold_idx in create_kfold_dataloaders(
                train_dataset, val_dataset, batch_size=1,
                worker_init_fn=worker_init_fn):
            model = get_model(model_name).to(DEVICE)
            weight_path = (f"weights/Comparison/{dataset_name}/"
                           f"best_{model_name.lower()}_{dataset_name.lower()}_fold{fold_idx+1}.pth")
            per_image = None
            if os.path.exists(weight_path):
                load_weights_flexible(model, weight_path, DEVICE)
                result = evaluate_model(model, test_loader, return_per_image=save_per_image)
                if save_per_image:
                    m, per_image = result
                else:
                    m = result
                fold_metrics.append(m)
                if save_per_image:
                    all_fold_per_image.append((fold_idx, per_image))
                print(f"  F{fold_idx+1:<7d} {m[0]:8.4f} {m[1]:8.4f} {m[2]:10.4f} {m[3]:8.4f} "
                      f"{m[4]:8.4f} {m[5]:8.4f} {m[6]:8.4f} {m[7]:8.4f}")

                # --- Visualization (first valid fold only) ---
                if visualize and len(fold_metrics) == 1:
                    vis_dir = os.path.join(VIS_DIR, dataset_name,
                                          f"{model_name}_fold{fold_idx + 1}")
                    save_visualization_figures(model, test_loader, dataset_name,
                                              f"{model_name}_F{fold_idx + 1}", vis_dir)
            else:
                print(f"  F{fold_idx+1:<7d} [WARN] missing: {weight_path}")
                missing_folds.append(fold_idx + 1)
            del model
            torch.cuda.empty_cache()

        del train_dataset, val_dataset

        # Save per-fold per-image CSV
        if save_per_image and all_fold_per_image:
            os.makedirs(per_image_dir, exist_ok=True)
            csv_path = f"{per_image_dir}/{model_name}_{dataset_name}_per_image.csv"
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Fold"] + METRIC_NAMES)
                writer.writeheader()
                for fold_idx, pi in all_fold_per_image:
                    for i in range(len(pi["Dice"])):
                        writer.writerow({
                            "Fold": f"F{fold_idx+1}",
                            **{name: pi[name][i] for name in METRIC_NAMES}
                        })
            print(f"  -> Per-image saved: {csv_path}")

        if fold_metrics:
            fm = np.array(fold_metrics)
            means = fm.mean(axis=0)
            stds = fm.std(axis=0)
            print(f"  {'-'*72}")
            print(f"  {'Mean':<8s} {means[0]:8.4f} {means[1]:8.4f} {means[2]:10.4f} {means[3]:8.4f} "
                  f"{means[4]:8.4f} {means[5]:8.4f} {means[6]:8.4f} {means[7]:8.4f}")
            print(f"  {'+-Std':<8s} {stds[0]:8.4f} {stds[1]:8.4f} {stds[2]:10.4f} {stds[3]:8.4f} "
                  f"{stds[4]:8.4f} {stds[5]:8.4f} {stds[6]:8.4f} {stds[7]:8.4f}")
            fold_tag = f"5-fold ({len(fold_metrics)}/5)" if missing_folds else "5-fold"
            return list(means), list(stds), fold_tag
        else:
            return [np.nan] * 8, [0.0] * 8, "no weights"


def main():
    parser = argparse.ArgumentParser(description="Comparison Testing")
    parser.add_argument("--model", type=str, default=None, choices=MODEL_NAMES,
                        help="Test a single model (default: all models)")
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS,
                        help="Test on a single dataset (default: all datasets)")
    parser.add_argument("--save_per_image", action="store_true", default=True,
                        help="Save per-image metrics to CSV for t-test analysis (default: True)")
    parser.add_argument("--visualize", action="store_true", default=False,
                        help="Generate per-image prediction visualization figures")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODEL_NAMES
    datasets_to_run = [args.dataset] if args.dataset else DATASETS

    model_tag = args.model or "AllModels"
    data_tag = args.dataset or "AllDatasets"
    logger = Logger(log_dir="logs", prefix=f"test_{model_tag}_{data_tag}")
    sys.stdout = logger

    print("=" * 80)
    print(f"  Comparison Testing  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {models_to_run}")
    print(f"  Datasets: {datasets_to_run}")
    print(f"  Device: {DEVICE}  |  Visualize: {args.visualize}  |  Log: {logger.log_path}")
    print("=" * 80)

    all_results = []
    for model_name in models_to_run:
        for dataset_name in datasets_to_run:
            try:
                means, stds, fold_str = test_single(
                    model_name, dataset_name,
                    save_per_image=args.save_per_image,
                    visualize=args.visualize)
                all_results.append({
                    "Model": model_name, "Dataset": dataset_name, "Fold": fold_str,
                    "Params": get_param_count(model_name),
                    **{METRIC_NAMES[i]: means[i] for i in range(len(METRIC_NAMES))},
                    **{f"{METRIC_NAMES[i]}_std": stds[i] for i in range(len(METRIC_NAMES))},
                })
            except Exception as e:
                print(f"\n  [SKIP] {model_name}/{dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "Model": model_name, "Dataset": dataset_name, "Fold": "ERROR",
                    "Params": get_param_count(model_name),
                    **{name: np.nan for name in METRIC_NAMES},
                    **{f"{name}_std": 0.0 for name in METRIC_NAMES},
                })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    for dataset_name in datasets_to_run:
        ds_results = [r for r in all_results if r["Dataset"] == dataset_name]
        if not ds_results:
            continue

        ds_results.sort(key=lambda r: r["Dice"] if not np.isnan(r["Dice"]) else 0.0, reverse=True)

        print(section_header(f"RESULTS: {dataset_name}", 150))
        hdr = (f"  {'Model':<18} {'Params':>8} {'Dice':>16} {'IoU':>16} {'Precision':>14} "
               f"{'Recall':>12} {'Spec':>12} {'FPR':>12} {'HD95':>10} {'clDice':>10}")
        print(hdr)
        print(f"  {'-'*140}")
        for r in ds_results:
            def fmt_val(v, s):
                if np.isnan(v):
                    return "N/A"
                if s > 0:
                    return f"{v:.4f}+-{s:.4f}"
                return f"{v:.4f}"
            print(f"  {r['Model']:<18} {r['Params']:>8} "
                  f"{fmt_val(r['Dice'], r['Dice_std']):>16} "
                  f"{fmt_val(r['IoU'], r['IoU_std']):>16} "
                  f"{fmt_val(r['Precision'], r['Precision_std']):>14} "
                  f"{fmt_val(r['Recall'], r['Recall_std']):>12} "
                  f"{fmt_val(r['Spec'], r['Spec_std']):>12} "
                  f"{fmt_val(r['FPR'], r['FPR_std']):>12} "
                  f"{fmt_val(r['HD95'], r['HD95_std']):>10} "
                  f"{fmt_val(r['clDice'], r['clDice_std']):>10}")

        csv_path = f"results/comparison_{dataset_name.lower()}_{ts}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=["Model", "Params", "Fold"] + METRIC_NAMES)
            writer.writeheader()
            for r in ds_results:
                row = {"Model": r["Model"], "Params": r["Params"], "Fold": r["Fold"]}
                for i, name in enumerate(METRIC_NAMES):
                    v = r[name]
                    s = r[f"{name}_std"]
                    if np.isnan(v):
                        row[name] = "N/A"
                    elif s > 0:
                        row[name] = f"{v:.4f}+-{s:.4f}"
                    else:
                        row[name] = f"{v:.4f}"
                writer.writerow(row)
        print(f"  -> {csv_path}")

    print(f"\n  Log saved to: {logger.log_path}")
    logger.close()


if __name__ == '__main__':
    main()
