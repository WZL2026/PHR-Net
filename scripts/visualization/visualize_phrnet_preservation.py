#!/usr/bin/env python3
"""
可视化 PHR-Net 的高分辨率特征保持能力
=============================================
与 UNet 形成对比：展示 PHR-Net 的 Stream 1 全分辨率细节流在整个网络
深度中始终维持 H×W 分辨率，细血管信号不会因深度下采样而丢失。

提取 PHR-Net（HRNet backbone, Full PGCA）各中间阶段的 Stream 1 特征图，
跟踪同一 ROI 区域，展示细血管细节在不同深度均清晰可辨。

输出对比图：
  - 上半部分：UNet 下采样过程中血管信号逐级消失
  - 下半部分：PHR-Net Stream 1 全分辨率保持，血管信号全程保留

运行:
  python visualize_phrnet_preservation.py
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
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ---- 路径配置 ----
try:
    SCRIPT_DIR = _p
except NameError:
    SCRIPT_DIR = _p
sys.path.insert(0, SCRIPT_DIR)

from models.phr_net import PHR_Net
from models.unet import UNet

UNET_PATH = os.path.join(SCRIPT_DIR, "weights", "Comparison", "DRIVE", "best_unet_drive.pth")
PHRNET_PATH = os.path.join(SCRIPT_DIR, "weights", "Comparison", "DRIVE", "best_phr_net_drive.pth")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Dataset", "DRIVE")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Feature-extracting PHR-Net wrapper
# ============================================================
class PHRNetWithFeatures(nn.Module):
    """PHR-Net wrapper that captures Stream 1 features at each stage"""

    def __init__(self, phr_net):
        super().__init__()
        self.model = phr_net

    def forward(self, x):
        features = {}
        model = self.model
        backbone = model.backbone

        # Module A: Prior Enhancement
        x_feat, prior_feat = model.module_a(x)
        features['after_pgfe'] = x_feat.detach()

        if backbone == "hrnet":
            s1 = x_feat
            if model.use_a:
                s1 = model.stream1_dssb(s1)
            features['stream1_dssb'] = s1.detach()

            s1 = model.stream1_conv1(s1)
            features['stream1_stage1'] = s1.detach()

            s2 = model.transition_s2(s1)
            s2 = model.stream2_conv1(s2)
            if model.use_pgca:
                gate_s2 = model.pgca_s2(prior_feat)
                gate_s2 = gate_s2.view(-1, 128, 1, 1)
                s2 = s2 * gate_s2

            s1, s2 = model.cross_1_2(s1, s2)
            s1 = model.stream1_conv2(s1)
            s2 = model.stream2_conv2(s2)
            features['stream1_stage2'] = s1.detach()

            s1, s2 = model.cross_1_2_2(s1, s2)
            s1 = model.stream1_conv3(s1)
            s2 = model.stream2_conv3(s2)
            features['stream1_stage3'] = s1.detach()

            # Module B (LKA + bottleneck)
            if model.use_b and model.lka_mode == "deep":
                s2_deep = model.s2_down_deep(s2)
                s2_deep = model.s2_bottleneck(s2_deep)
                features['s2_bottleneck'] = s2_deep.detach()
                s2_deep = model.s2_lka_deep(s2_deep)
                s2_deep_up = F.interpolate(s2_deep, scale_factor=2, mode='bilinear',
                                           align_corners=False)
                s2_attn = model.s2_deep_gate(s2_deep_up)
                s2_attn = F.interpolate(s2_attn, size=s2.shape[2:],
                                        mode='bilinear', align_corners=False)
                if model.lka_fusion == "residual":
                    s2 = s2 + model.lka_strength * s2_attn
                else:
                    s2 = s2 * (1.0 + model.lka_strength * s2_attn)
            elif model.lka_mode == "deep":
                s2_deep = model.s2_down_deep(s2)
                s2_deep = model.s2_bottleneck(s2_deep)
                features['s2_bottleneck'] = s2_deep.detach()
                s2_deep_up = F.interpolate(s2_deep, scale_factor=2, mode='bilinear',
                                           align_corners=False)
                s2_attn = model.s2_deep_gate(s2_deep_up)
                s2_attn = F.interpolate(s2_attn, size=s2.shape[2:],
                                        mode='bilinear', align_corners=False)
                s2 = s2 * (1.0 + model.lka_strength * s2_attn)

        elif backbone == "unet":
            raise RuntimeError("UNet backbone not supported for PHR-Net visualization")

        # Heads + Fusion
        out1 = model.head1(s1)
        out2 = model.head2(s2)
        out_fused = model.fusion_head(s1, s2, out1, out2)
        features['final_pred'] = torch.sigmoid(out_fused).detach()

        return out_fused, features


# ============================================================
# Utility
# ============================================================
def feature_to_heatmap(feature_map):
    """(1, C, H, W) → (H, W) [0, 1]"""
    fm = feature_map.squeeze(0)
    heatmap = fm.mean(dim=0)
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    return heatmap.numpy()


def roi_resize(heatmap_2d, target_h, target_w):
    return cv2.resize(heatmap_2d.astype(np.float32), (target_w, target_h),
                      interpolation=cv2.INTER_NEAREST)


def load_image(img_path, gt_path):
    """OpenCV 加载 + padding + normalize（与 dataset.py 一致）"""
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    H, W = image.shape[:2]
    pad_w = (32 - W % 32) % 32
    pad_h = (32 - H % 32) % 32
    if pad_w > 0 or pad_h > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')

    img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
    img_tensor = TF.normalize(img_t, mean=[0.215, 0.215, 0.215],
                              std=[0.151, 0.151, 0.151])
    mask_tensor = (mask_t > 0.5).float()
    img_np = image.astype(np.float32) / 255.0
    return img_tensor, mask_tensor, img_np


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Load models ----
    # UNet
    print("Loading UNet...")
    unet = UNet(in_channels=3, out_channels=1).to(device)
    ckpt_unet = torch.load(UNET_PATH, map_location=device, weights_only=False)
    sd_unet = ckpt_unet.get('model_state_dict', ckpt_unet)
    unet.load_state_dict(sd_unet, strict=False)
    unet.eval()

    # PHR-Net (HRNet backbone, use_a=True, use_b=True)
    print("Loading PHR-Net...")
    phr_net_base = PHR_Net(in_channels=3, out_classes=1,
                           use_a=True, use_b=True,
                           backbone="hrnet",
                           dsconv_mode="none")
    phr_net_base = phr_net_base.to(device)
    ckpt_phr = torch.load(PHRNET_PATH, map_location=device, weights_only=False)
    sd_phr = ckpt_phr.get('model_state_dict', ckpt_phr)
    phr_net_base.load_state_dict(sd_phr, strict=False)
    phr_net_base.eval()
    phr_net = PHRNetWithFeatures(phr_net_base).to(device)

    # ---- 2. Load image ----
    test_dir = os.path.join(DATA_DIR, "test")
    img_files = sorted([f for f in os.listdir(os.path.join(test_dir, "images"))
                        if f.endswith(".tif")])
    img_name = img_files[1]  # 02_test.tif
    img_path = os.path.join(test_dir, "images", img_name)
    base_name = img_name.split('_')[0]
    gt_path = os.path.join(test_dir, "1st_manual", f"{base_name}_manual1.gif")

    print(f"Image: {img_path}")
    img_tensor, mask_tensor, img_np = load_image(img_path, gt_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Padded size: {H}×{W}")

    # ---- 3. Forward both models ----
    with torch.no_grad():
        # UNet: capture intermediate features via a custom forward
        # We reuse the logic from visualize_unet_downsampling.py here
        from models.unet import DoubleConv
        unet_feats = {}

        x = img_tensor
        # Encoder
        for i, down in enumerate(unet.downs):
            x = down(x)
            unet_feats[f'enc_stage{i+1}'] = x.detach()
            x = unet.pool(x)
        x = unet.bottleneck(x)
        unet_feats['bottleneck'] = x.detach()

        # PHR-Net
        pred_phr, phr_feats = phr_net(img_tensor)

        # Store PHR-Net Stream 1 features explicitly
        phr_s1_keys = ['after_pgfe', 'stream1_dssb', 'stream1_stage1',
                       'stream1_stage2', 'stream1_stage3']
        phr_s1_feats = {k: phr_feats[k] for k in phr_s1_keys if k in phr_feats}

        print(f"UNet features: {list(unet_feats.keys())}")
        print(f"PHR-Net S1 features: {list(phr_s1_feats.keys())}")

    # ---- 4. Select ROI ----
    roi_cy, roi_cx = H // 2 + 60, W // 3
    roi_h, roi_w = 64, 64
    y1 = max(0, roi_cy - roi_h // 2); y2 = min(H, roi_cy + roi_h // 2)
    x1 = max(0, roi_cx - roi_w // 2); x2 = min(W, roi_cx + roi_w // 2)
    roi_ah = y2 - y1; roi_aw = x2 - x1
    print(f"ROI: ({x1},{y1}) → ({x2},{y2}), {roi_aw}×{roi_ah}")

    # ---- 5. Create comparison figure ----
    # Layout: 2 rows × 5 columns
    # Row 1: UNet encoder stages (progressive loss)
    # Row 2: PHR-Net Stream 1 stages (full resolution preservation)

    unet_stages = [
        ('UNet\nStage 1 (H×W)', unet_feats['enc_stage1']),
        ('UNet\nStage 2 (H/2×W/2)', unet_feats['enc_stage2']),
        ('UNet\nStage 3 (H/4×W/4)', unet_feats['enc_stage3']),
        ('UNet\nStage 4 (H/8×W/8)', unet_feats['enc_stage4']),
        ('UNet\nBottleneck (H/16×W/16)', unet_feats['bottleneck']),
    ]

    phr_stages = [
        ('PHR-Net S1\nAfter PGFE (H×W)', phr_feats['after_pgfe']),
        ('PHR-Net S1\nStage 1 (H×W)', phr_feats['stream1_stage1']),
        ('PHR-Net S1\nStage 2 (H×W)', phr_feats['stream1_stage2']),
        ('PHR-Net S1\nStage 3 (H×W)', phr_feats['stream1_stage3']),
        ('PHR-Net\nFinal Pred (H×W)', phr_feats['final_pred']),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(24, 11))

    # Colored label bands
    for col in range(5):
        axes[0, col].patch.set_edgecolor('#d62728')
        axes[0, col].patch.set_linewidth(3)
        axes[1, col].patch.set_edgecolor('#2ca02c')
        axes[1, col].patch.set_linewidth(3)

    # ---- Row 1: UNet ----
    for i, (name, feat) in enumerate(unet_stages):
        ax = axes[0, i]
        hm = feature_to_heatmap(feat)
        scale = hm.shape[0] / H
        ry1 = int(y1 * scale); ry2 = int(y2 * scale)
        rx1 = int(x1 * scale); rx2 = int(x2 * scale)
        roi_hm = hm[ry1:ry2, rx1:rx2]
        roi_rz = roi_resize(roi_hm, roi_ah, roi_aw)
        ax.imshow(roi_rz, cmap='inferno', vmin=0, vmax=1)
        res = H // (2 ** i)
        ax.set_title(name + f'\n(res={res}×{res})', fontsize=10, fontweight='bold')
        ax.axis('off')

    # ---- Row 2: PHR-Net Stream 1 ----
    for i, (name, feat) in enumerate(phr_stages):
        ax = axes[1, i]
        # PHR-Net S1 features are all at H×W resolution
        if 'Final Pred' in name:
            # Final prediction is already sigmoid, single channel
            hm = feat.squeeze(0).squeeze(0).cpu().numpy()
        else:
            hm = feature_to_heatmap(feat)
        # All at H×W, same ROI extraction
        roi_hm = hm[y1:y2, x1:x2]
        if roi_hm.shape[0] != roi_ah or roi_hm.shape[1] != roi_aw:
            roi_hm = roi_resize(roi_hm, roi_ah, roi_aw)
        ax.imshow(roi_hm, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.axis('off')

    # ---- Labels ----
    # Row labels
    fig.text(0.01, 0.75, 'UNet\n(Deep\nDown-\nsampling)',
             fontsize=13, fontweight='bold', color='#d62728',
             ha='center', va='center', rotation=0)
    fig.text(0.01, 0.25, 'PHR-Net\n(Full-Res\nStream 1)',
             fontsize=13, fontweight='bold', color='#2ca02c',
             ha='center', va='center', rotation=0)

    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    # Save
    compare_path = os.path.join(OUTPUT_DIR, "Figure1a_PHRNet_vs_UNet_FeaturePreservation.png")
    fig.savefig(compare_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved comparison: {compare_path}")

    # ---- 6. Also save PHR-Net-only ROI figure (clean, for paper) ----
    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))

    phr_clean_names = [
        'After PGFE\n(H×W)',
        'Stream 1 Stg 1\n(H×W)',
        'Stream 1 Stg 2\n(H×W)',
        'Stream 1 Stg 3\n(H×W)',
        'Final Prediction\n(H×W)',
    ]

    for i, (name, feat) in enumerate(phr_stages):
        ax = axes3[i]
        if 'Final Pred' in name:
            hm = feat.squeeze(0).squeeze(0).cpu().numpy()
        else:
            hm = feature_to_heatmap(feat)
        roi_hm = hm[y1:y2, x1:x2]
        if roi_hm.shape[0] != roi_ah or roi_hm.shape[1] != roi_aw:
            roi_hm = roi_resize(roi_hm, roi_ah, roi_aw)
        ax.imshow(roi_hm, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()

    phr_only_path = os.path.join(OUTPUT_DIR, "Figure1c_PHRNet_FeaturePreservation.png")
    fig3.savefig(phr_only_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved PHR-Net-only: {phr_only_path}")

    print("\nDone! Three images generated:")
    print(f"  1. UNet-only:        results/paper_figures/Figure1a_UNet_Downsampling_ROI_Compact.png")
    print(f"  2. PHR-Net-only:     {phr_only_path}")
    print(f"  3. Side-by-side:     {compare_path}")


if __name__ == "__main__":
    main()
