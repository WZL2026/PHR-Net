#!/usr/bin/env python3
"""
可视化 UNet 下采样过程中的高频信息丢失
===============================================
用于支撑论文中的论述：「考虑一条宽度仅为2像素的毛细血管：经过三次
2×下采样操作后，其空间足迹被压缩了8倍，解码器几乎无信号可恢复。」

读取一张 DRIVE 测试图像，用训练好的 UNet 做前向传播，提取编码器
各阶段的中间特征图，展示细血管信号在逐级池化中被逐步淹没的过程。

输出：一张拼接图，包含
  (a) 输入图像 (带 ROI 标注框)
  (b) 编码器 Stage 1 特征图 (64 ch, H×W) → ROI 放大
  (c) 编码器 Stage 2 特征图 (128 ch, H/2×W/2) → ROI 放大
  (d) 编码器 Stage 3 特征图 (256 ch, H/4×W/4) → ROI 放大
  (e) 编码器 Stage 4 特征图 (512 ch, H/8×W/8) → ROI 放大
  (f) 瓶颈层特征图 (1024 ch, H/16×W/16) → ROI 放大

运行:
  python visualize_unet_downsampling.py
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
# 必须在 import torch 之前设置，解决 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ---- 路径配置 ----
try:
    SCRIPT_DIR = _p
except NameError:
    SCRIPT_DIR = _p
MODEL_PATH = os.path.join(SCRIPT_DIR, "weights", "Comparison", "DRIVE", "best_unet_drive.pth")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Dataset", "DRIVE")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

from models.unet import UNet

class UNetWithFeatures(nn.Module):
    """Wrapper around the REAL models.unet.UNet, using hooks to extract
       intermediate encoder features at each stage and bottleneck."""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels, features=features)
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        # Hook encoder outputs (after each DoubleConv, before pool)
        for i, down in enumerate(self.unet.downs):
            down.register_forward_hook(
                lambda m, inp, out, idx=i: self._features.update({f'enc_stage{idx+1}': out.detach()})
            )
        # Hook bottleneck output
        self.unet.bottleneck.register_forward_hook(
            lambda m, inp, out: self._features.update({'bottleneck': out.detach()})
        )

    def forward(self, x, return_features=True):
        self._features.clear()
        seg = self.unet(x)
        if isinstance(seg, tuple):
            seg = seg[0]  # (final_seg, edge_maps) in training mode
        return seg, dict(self._features)


def load_image_with_padding(img_path, gt_path):
    """用 OpenCV 加载图像并转换为 tensor（支持 .tif/.png/.jpg 等格式）"""
    import cv2

    print("  Loading with OpenCV...", flush=True)
    # 用 cv2 读取（BGR → RGB）
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8 [0,255]
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)      # (H, W) uint8 [0,255]
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {gt_path}")
    print(f"  Image shape: {image.shape}", flush=True)

    H, W = image.shape[:2]
    pad_w = (32 - W % 32) % 32
    pad_h = (32 - H % 32) % 32
    print(f"  Padding: {pad_w}x{pad_h}", flush=True)

    # 对 numpy 数组做 zero-padding
    if pad_w > 0 or pad_h > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
    print(f"  Padded shape: {image.shape}", flush=True)

    # 转为 torch tensor (H,W,C) → (C,H,W) 并归一化到 [0,1]
    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0         # (1, H, W)

    # 归一化（与 dataset.py 相同参数）
    image_tensor = TF.normalize(image_t, mean=[0.215, 0.215, 0.215],
                                std=[0.151, 0.151, 0.151])
    mask_tensor = (mask_t > 0.5).float()
    print("  Done loading.", flush=True)

    # image_np 用于 matplotlib 可视化 (H, W, 3) [0,1]
    image_np = image.astype(np.float32) / 255.0
    return image_tensor, mask_tensor, image_np


def roi_resize(heatmap_2d, target_h, target_w):
    """将任意尺寸的2D热力图用最近邻上采样到目标尺寸"""
    import cv2
    return cv2.resize(heatmap_2d.astype(np.float32), (target_w, target_h),
                      interpolation=cv2.INTER_NEAREST)


def feature_to_heatmap(feature_map):
    """将 (C,H,W) 特征图转为单通道热力图（通道均值 + min-max 归一化）"""
    fm = feature_map.squeeze(0)  # (C, H, W)
    heatmap = fm.mean(dim=0)  # (H, W)
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    return heatmap.numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 1. 加载模型 ----
    print(f"Loading UNet from: {MODEL_PATH}")
    model = UNetWithFeatures(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    # 处理可能的 state_dict key 包装
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")

    # ---- 2. 加载测试图像 ----
    test_dir = os.path.join(DATA_DIR, "test")
    img_files = sorted([
        f for f in os.listdir(os.path.join(test_dir, "images")) if f.endswith(".tif")
    ])

    # 选一张血管丰富的图（第 2 张 02_test.tif 通常血管信息比较丰富）
    img_name = img_files[1]  # 02_test.tif
    img_path = os.path.join(test_dir, "images", img_name)
    base_name = img_name.split('_')[0]
    gt_path = os.path.join(test_dir, "1st_manual", f"{base_name}_manual1.gif")

    print(f"Loading image: {img_path}", flush=True)
    img_tensor, mask_tensor, img_np = load_image_with_padding(img_path, gt_path)
    print(f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}", flush=True)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Image size after padding: {H}x{W}", flush=True)

    # ---- 3. 前向传播，提取中间特征 ----
    with torch.no_grad():
        seg, features = model(img_tensor, return_features=True)

    # ---- 4. 选择一个 ROI 区域（包含细血管） ----
    # 这里选择一个固定 ROI，可根据实际情况调整
    # DRIVE 02_test 图片中，右下角区域有较好的细血管
    roi_cy, roi_cx = H // 2 + 60, W // 3      # ROI 中心点
    roi_h, roi_w = 64, 64                      # ROI 尺寸

    # 确保 ROI 在图像范围内
    y1 = max(0, roi_cy - roi_h // 2)
    y2 = min(H, roi_cy + roi_h // 2)
    x1 = max(0, roi_cx - roi_w // 2)
    x2 = min(W, roi_cx + roi_w // 2)
    roi_actual_h = y2 - y1
    roi_actual_w = x2 - x1

    print(f"ROI: ({x1},{y1}) → ({x2},{y2}), size {roi_actual_w}×{roi_actual_h}")

    # ---- 5. 可视化 ----
    # 原始图像用于显示（直接使用未归一化的 numpy 数组）
    img_vis = img_np  # (H, W, 3) [0,1]
    mask_vis = mask_tensor.squeeze(0).numpy()

    # 6 个阶段: input + 4 encoder stages + bottleneck = 6 columns
    stage_names = [
        ('Input\n(Original)', None),
        ('Stage 1\n(64 ch, H×W)', features['enc_stage1']),
        ('Stage 2\n(128 ch, H/2×W/2)', features['enc_stage2']),
        ('Stage 3\n(256 ch, H/4×W/4)', features['enc_stage3']),
        ('Stage 4\n(512 ch, H/8×W/8)', features['enc_stage4']),
        ('Bottleneck\n(1024 ch, H/16×W/16)', features['bottleneck']),
    ]

    # ===== 大图版本：2行×6列 =====
    fig, axes = plt.subplots(2, 6, figsize=(26, 10))

    # ---- 第一行：全特征图 + ROI 标注框 ----
    # (a) 输入图像 + ROI 框
    ax = axes[0, 0]
    ax.imshow(img_vis)
    rect = patches.Rectangle((x1, y1), roi_actual_w, roi_actual_h,
                              linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.set_title('(a) Input + ROI', fontsize=11, fontweight='bold')
    ax.axis('off')

    # (b)-(f) 各阶段全特征图
    for i, (name, feat) in enumerate(stage_names[1:], start=1):
        ax = axes[0, i]
        hm = feature_to_heatmap(feat)
        ax.imshow(hm, cmap='inferno')
        scale = hm.shape[0] / H
        rx1 = int(x1 * scale); ry1 = int(y1 * scale)
        rw = int(roi_actual_w * scale); rh = int(roi_actual_h * scale)
        rect = patches.Rectangle((rx1, ry1), rw, rh,
                                  linewidth=1.5, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f'({chr(97+i)}) {name}', fontsize=11, fontweight='bold')
        ax.axis('off')

    # ---- 第二行：ROI 放大对比（全部上采样到统一尺寸） ----
    # (g) 输入图像 ROI
    ax = axes[1, 0]
    roi_img = img_vis[y1:y2, x1:x2]
    ax.imshow(roi_img)
    ax.set_title('(g) ROI: Input', fontsize=11, fontweight='bold')
    ax.axis('off')

    # (h)-(l) 各阶段 ROI 上采样到原始尺寸
    for i, (name, feat) in enumerate(stage_names[1:], start=1):
        ax = axes[1, i]
        hm = feature_to_heatmap(feat)
        scale = hm.shape[0] / H
        ry1 = int(y1 * scale); ry2 = int(y2 * scale)
        rx1 = int(x1 * scale); rx2 = int(x2 * scale)
        roi_hm = hm[ry1:ry2, rx1:rx2]
        roi_resized = roi_resize(roi_hm, roi_actual_h, roi_actual_w)
        ax.imshow(roi_resized, cmap='inferno', vmin=0, vmax=1)
        res = H // (2 ** i)
        ax.set_title(f'({chr(103+i)}) Stage {i} ROI\n'
                     f'(res {res}×{res}, {2**i}× down)',
                     fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    # 保存大图
    output_path = os.path.join(OUTPUT_DIR, "Figure1a_UNet_Downsampling_Evidence.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved full figure to: {output_path}")

    # ===== 紧凑版：1行×6列（仅 ROI，适合论文插图） =====
    fig2, axes2 = plt.subplots(1, 6, figsize=(20, 3.8))

    titles_compact = [
        'Input (ROI)',
        'Stage 1\n(H×W)',
        'Stage 2\n(H/2×W/2)',
        'Stage 3\n(H/4×W/4)',
        'Stage 4\n(H/8×W/8)',
        'Bottleneck\n(H/16×W/16)',
    ]

    # Input ROI
    axes2[0].imshow(roi_img)
    axes2[0].set_title(titles_compact[0], fontsize=10, fontweight='bold')
    axes2[0].axis('off')

    # Feature ROI at each stage
    all_features = [features[k] for k in
                    ['enc_stage1', 'enc_stage2', 'enc_stage3', 'enc_stage4', 'bottleneck']]
    for i, feat in enumerate(all_features):
        hm = feature_to_heatmap(feat)
        scale = hm.shape[0] / H
        ry1 = int(y1 * scale); ry2 = int(y2 * scale)
        rx1 = int(x1 * scale); rx2 = int(x2 * scale)
        roi_hm = hm[ry1:ry2, rx1:rx2]
        roi_resized = roi_resize(roi_hm, roi_actual_h, roi_actual_w)
        axes2[i + 1].imshow(roi_resized, cmap='inferno', vmin=0, vmax=1)
        axes2[i + 1].set_title(titles_compact[i + 1], fontsize=10, fontweight='bold')
        axes2[i + 1].axis('off')

    plt.tight_layout()

    compact_path = os.path.join(OUTPUT_DIR,
                                "Figure1a_UNet_Downsampling_ROI_Compact.png")
    fig2.savefig(compact_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved compact figure to: {compact_path}")


if __name__ == "__main__":
    main()
