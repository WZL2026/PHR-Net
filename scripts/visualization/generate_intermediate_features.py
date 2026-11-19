#!/usr/bin/env python3
"""
PHR-Net 中间特征图生成脚本
============================
为总体框架图 (Figure 2) 生成各环节的中间特征图可视化,
用于插入到 draw.io 框架图中展示数据流动和特征演化。

用法:
  python generate_intermediate_features.py

输出:
  results/intermediate_features/
    ├── 00_input_image.png              # 输入图像
    ├── 01_laplacian_prior.png           # Laplacian 先验边缘图
    ├── 02_enhanced_prior.png            # 空间注意力增强后的先验
    ├── 03_pgfe_output.png              # PGFE 输出 (S1_in)
    ├── 04_stream1_dssb.png             # Stream 1 DSSB 后
    ├── 05_stream1_stage1.png           # Stream 1 Stage 1
    ├── 06_stream2_entry.png            # Stream 2 入口 (BlurPool后)
    ├── 07_cross_exchange1_s1.png       # 第一次跨流交换后 S1
    ├── 08_cross_exchange1_s2.png       # 第一次跨流交换后 S2
    ├── 09_stream1_stage2.png           # Stream 1 Stage 2
    ├── 10_stream2_stage2.png           # Stream 2 Stage 2
    ├── 11_cross_exchange2_s1.png       # 第二次跨流交换后 S1
    ├── 12_cross_exchange2_s2.png       # 第二次跨流交换后 S2
    ├── 13_stream1_stage3.png           # Stream 1 Stage 3 (最终细节特征)
    ├── 14_stream2_stage3.png           # Stream 2 Stage 3 (最终上下文特征)
    ├── 15_agf_attention_map.png        # AGF 多尺度大核注意力图
    ├── 16_strip_pool_attention.png     # 条带池化注意力图
    ├── 17_fusion_weight_s1.png         # 融合门控权重 W_S1
    ├── 18_fusion_weight_s2.png         # 融合门控权重 W_S2
    ├── 19_final_prediction.png         # 最终分割预测
    └── 20_overview_grid.png            # 综合概览网格图
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
import matplotlib.gridspec as gridspec

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

from models.phr_net import PHR_Net, BlurPool


# ======================================================================
# Feature-extracting PHR-Net — 捕获所有中间特征图
# ======================================================================
class PHRNetFullFeatures(nn.Module):
    """PHR-Net wrapper that captures ALL intermediate features for diagram annotation."""

    def __init__(self, phr_net):
        super().__init__()
        self.model = phr_net

    def forward(self, x):
        feats = {}  # name → tensor (1, C, H, W) or (1, 1, H, W)
        model = self.model
        backbone = model.backbone

        # ---- Module A: Prior Enhancement (PGFE) ----
        # Stage 1: Laplacian prior extraction
        lap_prior = model.module_a.dpe(x)  # (1, 3, H, W) — raw Laplacian
        feats['01_laplacian_prior'] = lap_prior.detach()

        # Project prior to 64ch
        prior_feat = model.module_a.lap_proj(lap_prior)  # (1, 64, H, W)
        feats['01b_prior_projected'] = prior_feat.detach()

        # Stage 2: Spatial attention enhancement
        if model.use_a:
            prior_feat_enh = model.module_a.dpss(prior_feat)  # residual amplify
        else:
            prior_feat_enh = prior_feat
        feats['02_enhanced_prior'] = prior_feat_enh.detach()

        # Main stem
        main_feat = model.module_a.enc_stem(x)  # (1, 64, H, W)

        # Stage 3: Multi-scale selector + fusion (inside stream1_dssb)
        x_feat = main_feat + prior_feat_enh  # S1_in — PGFE output
        feats['03_pgfe_output'] = x_feat.detach()

        if backbone == "hrnet":
            # ============ PGCA: Dual-Stream Backbone ============

            # DSSB (DualScopeSelectiveBlock) — multi-scale selection
            s1 = x_feat
            if model.use_a:
                # Capture DSSB internal: micro/macro weights
                dssb_feat = model.stream1_dssb.bn_in(
                    model.stream1_dssb.input_proj(s1))
                decision_map = model.stream1_dssb.selector(dssb_feat)
                w_micro, w_macro = torch.chunk(decision_map, 2, dim=1)
                feats['04a_dssb_weight_micro'] = w_micro.detach()
                feats['04b_dssb_weight_macro'] = w_macro.detach()

                s1 = model.stream1_dssb(s1)
            feats['04_stream1_dssb'] = s1.detach()

            # Stream 1 Stage 1
            s1 = model.stream1_conv1(s1)
            feats['05_stream1_stage1'] = s1.detach()

            # ---- Stream 2 transition ----
            s2 = model.transition_s2(s1)
            s2 = model.stream2_conv1(s2)

            # PGCA-S2: prior-gated channel attention
            if model.use_pgca:
                gate_s2_raw = model.pgca_s2(prior_feat_enh)  # (B, 128)
                gate_s2 = gate_s2_raw.view(-1, 128, 1, 1)
                feats['06a_pgca_gate_vector'] = gate_s2_raw.detach()
                s2 = s2 * gate_s2

            feats['06_stream2_entry'] = s2.detach()

            # ---- Cross-Stream Exchange 1 ----
            # S1→S2: downsample S1, add to S2 (no gating)
            s1_to_s2 = model.cross_1_2.down_conv(s1)
            # S2→S1: upsample S2, add to S1 (no gating in CrossStreamResidual)
            s2_up = F.interpolate(s2, size=s1.shape[2:],
                                  mode='bilinear', align_corners=False)
            s2_to_s1 = model.cross_1_2.up_conv(s2_up)

            s1_ex1 = s1 + s2_to_s1  # after S2→S1 injection
            s2_ex1 = s2 + s1_to_s2  # after S1→S2 injection
            feats['07_cross_exchange1_s1'] = s1_ex1.detach()
            feats['08_cross_exchange1_s2'] = s2_ex1.detach()

            s1 = model.stream1_conv2(s1_ex1)
            s2 = model.stream2_conv2(s2_ex1)
            feats['09_stream1_stage2'] = s1.detach()
            feats['10_stream2_stage2'] = s2.detach()

            # ---- Cross-Stream Exchange 2 ----
            s1_to_s2_2 = model.cross_1_2_2.down_conv(s1)
            s2_up_2 = F.interpolate(s2, size=s1.shape[2:],
                                     mode='bilinear', align_corners=False)
            s2_to_s1_2 = model.cross_1_2_2.up_conv(s2_up_2)

            s1_ex2 = s1 + s2_to_s1_2
            s2_ex2 = s2 + s1_to_s2_2
            feats['11_cross_exchange2_s1'] = s1_ex2.detach()
            feats['12_cross_exchange2_s2'] = s2_ex2.detach()

            s1 = model.stream1_conv3(s1_ex2)
            s2 = model.stream2_conv3(s2_ex2)
            feats['13_stream1_stage3'] = s1.detach()  # final detail features
            feats['14_stream2_stage3'] = s2.detach()  # final context features

            # ---- Module B: AGF Stage 1 — Multi-Scale LKA ----
            if model.use_b and model.lka_mode == "deep":
                s2_deep = model.s2_down_deep(s2)
                s2_deep = model.s2_bottleneck(s2_deep)
                feats['15a_bottleneck_before_lka'] = s2_deep.detach()

                # Capture LKA attention map
                lka_attn = model.s2_lka_deep(s2_deep)
                feats['15_agf_attention_map'] = lka_attn.detach()

                s2_deep = lka_attn  # already u * attn in MacroscopicLKA
                s2_deep_up = F.interpolate(s2_deep, scale_factor=2, mode='bilinear',
                                           align_corners=False)
                s2_attn = model.s2_deep_gate(s2_deep_up)
                s2_attn = F.interpolate(s2_attn, size=s2.shape[2:],
                                        mode='bilinear', align_corners=False)
                s2_lka_out = s2 * (1.0 + model.lka_strength * s2_attn)
                feats['15c_s2_after_lka'] = s2_lka_out.detach()

                s2_for_fusion = s2_lka_out
            elif model.lka_mode == "deep":
                s2_deep = model.s2_down_deep(s2)
                s2_deep = model.s2_bottleneck(s2_deep)
                feats['15a_bottleneck_before_lka'] = s2_deep.detach()
                s2_deep_up = F.interpolate(s2_deep, scale_factor=2, mode='bilinear',
                                           align_corners=False)
                s2_attn = model.s2_deep_gate(s2_deep_up)
                s2_attn = F.interpolate(s2_attn, size=s2.shape[2:],
                                        mode='bilinear', align_corners=False)
                s2_for_fusion = s2 * (1.0 + model.lka_strength * s2_attn)
            else:
                s2_for_fusion = s2

            # ---- Module B: AGF Stage 2 — Residual Strip Fusion ----
            # Decode the fusion head internals
            fusion = model.fusion_head
            s2_up_feat = F.interpolate(s2_for_fusion, size=s1.shape[2:],
                                       mode='bilinear', align_corners=False)
            s2_up_feat_sm = fusion.smooth_up(s2_up_feat)
            concat_feat = torch.cat([s1, s2_up_feat_sm], dim=1)

            # Gating weights
            weight_map = fusion.gate(concat_feat)
            w1 = weight_map[:, 0:1, :, :]
            w2 = weight_map[:, 1:2, :, :]
            feats['17_fusion_weight_s1'] = w1.detach()
            feats['18_fusion_weight_s2'] = w2.detach()

            # Strip pooling attention
            if fusion.use_rsfh:
                fused_feat = fusion.fusion_conv(concat_feat)
                strip_out = fusion.strip_pool(fused_feat)
                feats['16_strip_pool_attention'] = strip_out.detach()

            # Final prediction
            out1 = model.head1(s1)
            out2 = model.head2(s2_for_fusion)
            out_fused = fusion(s1, s2_for_fusion, out1, out2)
            feats['19_final_prediction'] = torch.sigmoid(out_fused).detach()

        elif backbone == "unet":
            raise RuntimeError("UNet backbone not supported for this visualization")

        return out_fused, feats


# ======================================================================
# Visualization utilities
# ======================================================================
def feature_to_heatmap(feature_map, method='mean'):
    """(1, C, H, W) -> (H, W) [0, 1] via channel-mean + min-max normalization."""
    fm = feature_map.squeeze(0)
    if fm.ndim == 2:
        arr = fm.cpu().numpy()
    elif fm.ndim == 3:
        if method == 'mean':
            arr = fm.mean(dim=0).cpu().numpy()
        elif method == 'max':
            arr = fm.max(dim=0)[0].cpu().numpy()
        else:
            arr = fm.mean(dim=0).cpu().numpy()
    else:
        arr = np.zeros((64, 64))

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    return arr


def save_feature_image(heatmap_2d, save_path, title=None, cmap='inferno',
                       dpi=150, add_colorbar=False, figsize=(6, 6)):
    """Save a single feature map as a clean image file."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(heatmap_2d, cmap=cmap, aspect='equal')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activation', fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                edgecolor='none')
    plt.close(fig)


def save_overlay_image(heatmap_2d, background_img, save_path, title=None,
                       alpha=0.6, dpi=150, figsize=(6, 6)):
    """Overlay heatmap on original image for spatial context."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Background: original image
    if background_img.max() > 1.0:
        bg = background_img.astype(np.float32) / 255.0
    else:
        bg = background_img.astype(np.float32)

    ax.imshow(bg, aspect='equal')
    # Overlay heatmap
    hm_resized = cv2.resize(heatmap_2d.astype(np.float32),
                            (bg.shape[1], bg.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    ax.imshow(hm_resized, cmap='jet', alpha=alpha, aspect='equal')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white',
                edgecolor='none')
    plt.close(fig)


def save_side_by_side(feats_dict, save_path, bg_img=None, cols=5,
                      figsize_per_cell=(3, 3), dpi=150):
    """Create a multi-panel grid of feature maps."""
    keys = list(feats_dict.keys())
    rows = (len(keys) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per_cell[0],
                                                   rows * figsize_per_cell[1]))
    if rows == 1:
        axes = axes[np.newaxis, :]
    if cols == 1:
        axes = axes[:, np.newaxis]

    for idx, key in enumerate(keys):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        hm = feature_to_heatmap(feats_dict[key])
        if bg_img is not None:
            if bg_img.max() > 1.0:
                bg = bg_img.astype(np.float32) / 255.0
            else:
                bg = bg_img.astype(np.float32)
            ax.imshow(bg, aspect='equal')
            hm_rz = cv2.resize(hm.astype(np.float32),
                               (bg.shape[1], bg.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            ax.imshow(hm_rz, cmap='inferno', alpha=0.5, aspect='equal')
        else:
            ax.imshow(hm, cmap='inferno', aspect='equal')
        ax.set_title(key, fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for idx in range(len(keys), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ======================================================================
# Image loading (matching dataset.py preprocessing)
# ======================================================================
def load_drive_image(img_path, gt_path=None):
    """Load and preprocess a DRIVE image matching the training pipeline."""
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = image.shape[:2]
    pad_w = (32 - W % 32) % 32
    pad_h = (32 - H % 32) % 32
    original_shape = (H, W)
    if pad_w > 0 or pad_h > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = TF.normalize(img_t, mean=[0.215, 0.215, 0.215],
                              std=[0.151, 0.151, 0.151])

    mask_tensor = None
    if gt_path is not None and os.path.exists(gt_path):
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pad_w > 0 or pad_h > 0:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

    image_np = image.astype(np.float32) / 255.0

    return img_tensor, mask_tensor, image_np, original_shape


# ======================================================================
# Main
# ======================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 1. Paths ----
    PHRNET_PATH = os.path.join(
        SCRIPT_DIR, "weights", "Comparison", "DRIVE", "best_phr_net_drive.pth")
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Dataset", "DRIVE")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "intermediate_features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 2. Load model ----
    print("Loading PHR-Net (Full PGCA, use_a=True, use_b=True)...")
    phr_net_base = PHR_Net(in_channels=3, out_classes=1,
                           use_a=True, use_b=True,
                           backbone="hrnet",
                           dsconv_mode="none")
    phr_net_base = phr_net_base.to(device)

    ckpt = torch.load(PHRNET_PATH, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    phr_net_base.load_state_dict(sd, strict=False)
    phr_net_base.eval()

    phr_net = PHRNetFullFeatures(phr_net_base).to(device)
    print(f"  Params: {sum(p.numel() for p in phr_net.parameters()) / 1e6:.2f} M")

    # ---- 3. Load image ----
    test_dir = os.path.join(DATA_DIR, "test")
    # Use a representative test image with diverse vessel structures
    img_name = "02_test.tif"
    img_path = os.path.join(test_dir, "images", img_name)
    base_name = img_name.split('_')[0]
    gt_path = os.path.join(test_dir, "1st_manual", f"{base_name}_manual1.gif")

    print(f"Loading: {img_path}")
    img_tensor, mask_tensor, img_np, orig_shape = load_drive_image(img_path, gt_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"  Image size: {orig_shape[1]}×{orig_shape[0]} → padded: {W}×{H}")

    # ---- 4. Forward pass and capture ALL features ----
    print("Running forward pass with full feature extraction...")
    with torch.no_grad():
        pred, all_feats = phr_net(img_tensor)

    print(f"\nCaptured {len(all_feats)} intermediate feature maps:")
    for k, v in all_feats.items():
        print(f"  {k}: {list(v.shape)}")

    # ---- 5. Generate individual feature map visualizations ----
    print("\n" + "=" * 60)
    print("Generating feature map visualizations...")
    print("=" * 60)

    # Crop to original image (remove padding)
    oh, ow = orig_shape
    img_np_cropped = img_np[:oh, :ow]

    # Key feature maps for the framework diagram
    # Each entry: (dict_key, title, cmap, save_filename, description)
    feature_specs = [
        # ---- PGFE Module ----
        ('01_laplacian_prior', 'Laplacian Prior (F_lap)',
         'RdBu_r', '01_laplacian_prior.png',
         'PGFE Stage 1: 零参数Laplacian边缘检测, 提取二阶梯度响应'),
        ('02_enhanced_prior', 'Enhanced Prior (F_prior^enh)',
         'inferno', '02_enhanced_prior.png',
         'PGFE Stage 2: 空间注意力残差增强后的先验特征'),
        ('03_pgfe_output', 'PGFE Output (S1_in)',
         'inferno', '03_pgfe_output.png',
         'PGFE Stage 3: 多尺度选择器融合后的PGFE输出, 进入PGCA主干'),

        # ---- PGCA: Stream 1 (Full Resolution) ----
        ('04_stream1_dssb', 'S1 after DSSB (H×W)',
         'inferno', '04_stream1_dssb.png',
         'Stream 1 经过DualScopeSelectiveBlock后的全分辨率特征'),
        ('05_stream1_stage1', 'S1 Stage 1 (H×W)',
         'inferno', '05_stream1_stage1.png',
         'Stream 1 第一个FactorizedConvBlock后, 维持H×W'),

        # ---- PGCA: Stream 2 (Half Resolution) ----
        ('06_stream2_entry', 'S2 Entry (H/2×W/2, 128ch)',
         'inferno', '06_stream2_entry.png',
         'Stream 2 入口: BlurPool↓ + PGCA-S2门控后, 半分辨率上下文流'),

        # ---- PGCA: Cross-Stream Exchange 1 ----
        ('07_cross_exchange1_s1', 'S1 after Exchange 1 (H×W)',
         'inferno', '07_cross_exchange1_s1.png',
         '第一次跨流交换后 S1: S2上下文上采样注入S1'),
        ('08_cross_exchange1_s2', 'S2 after Exchange 1 (H/2×W/2)',
         'inferno', '08_cross_exchange1_s2.png',
         '第一次跨流交换后 S2: S1细节下采样注入S2'),

        # ---- PGCA: Stage 2 ----
        ('09_stream1_stage2', 'S1 Stage 2 (H×W)',
         'inferno', '09_stream1_stage2.png',
         'Stream 1 第二个FactorizedConvBlock后'),
        ('10_stream2_stage2', 'S2 Stage 2 (H/2×W/2)',
         'inferno', '10_stream2_stage2.png',
         'Stream 2 第二个FactorizedConvBlock后'),

        # ---- PGCA: Cross-Stream Exchange 2 ----
        ('11_cross_exchange2_s1', 'S1 after Exchange 2 (H×W)',
         'inferno', '11_cross_exchange2_s1.png',
         '第二次跨流交换后 S1: 最终全分辨率细节特征'),
        ('12_cross_exchange2_s2', 'S2 after Exchange 2 (H/2×W/2)',
         'inferno', '12_cross_exchange2_s2.png',
         '第二次跨流交换后 S2: 最终上下文特征'),

        # ---- PGCA: Final Stage 3 ----
        ('13_stream1_stage3', 'S1 Final (HxW)',
         'inferno', '13_stream1_stage3.png',
         'Stream 1 final output: full-res detail features S1 (64ch, HxW)'),
        ('14_stream2_stage3', 'S2 Final (H/2xW/2)',
         'inferno', '14_stream2_stage3.png',
         'Stream 2 final output: half-res context features S2 (128ch, H/2xW/2)'),

        # ---- AGF: Attention Maps ----
        ('15_agf_attention_map', 'AGF Multi-Scale LKA Attention (A_macro)',
         'inferno', '15_agf_attention_map.png',
         'AGF Stage 1: 多尺度大核注意力图 (d=1 + d=3 空洞深度卷积)'),
        ('16_strip_pool_attention', 'Strip Pooling Attention (W_strip)',
         'inferno', '16_strip_pool_attention.png',
         'AGF Stage 2: 水平和垂直条带池化注意力掩码'),

        # ---- AGF: Fusion Weights ----
        ('17_fusion_weight_s1', 'Fusion Weight W_S1 (Stream 1)',
         'viridis', '17_fusion_weight_s1.png',
         'AGF Stage 2: Softmax门控 S1权重 W_S1∈[0,1]'),
        ('18_fusion_weight_s2', 'Fusion Weight W_S2 (Stream 2)',
         'viridis', '18_fusion_weight_s2.png',
         'AGF Stage 2: Softmax门控 S2权重 W_S2∈[0,1]'),

        # ---- Final Output ----
        ('19_final_prediction', 'Final Prediction (Y_hat)',
         'inferno', '19_final_prediction.png',
         'PHR-Net final output: segmentation logits Y_hat (1ch, HxW)'),
    ]

    # Generate individual images
    for key, title, cmap, fname, desc in feature_specs:
        if key not in all_feats:
            print(f"  [SKIP] {key} — not in captured features")
            continue

        feat = all_feats[key]
        hm = feature_to_heatmap(feat)

        # For single-channel outputs, use inferno colormap
        if 'laplacian' in key:
            actual_cmap = 'RdBu_r'  # diverging for edge maps (pos/neg gradients)
        else:
            actual_cmap = 'inferno'    # clean infernoscale, no color cast

        save_path = os.path.join(OUTPUT_DIR, fname)
        save_feature_image(hm, save_path, title=title, cmap=actual_cmap)
        try:
            print(f"  [SAVED] {fname}  -- {desc}")
        except UnicodeEncodeError:
            print(f"  [SAVED] {fname}")

    # Also save the input image
    input_save = os.path.join(OUTPUT_DIR, '00_input_image.png')
    fig_in, ax_in = plt.subplots(1, 1, figsize=(6, 6))
    ax_in.imshow(img_np_cropped)
    ax_in.set_title('Input Image (X)', fontsize=12, fontweight='bold')
    ax_in.axis('off')
    fig_in.savefig(input_save, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_in)
    print(f"  [SAVED] 00_input_image.png  -- DRIVE input image")

    # ---- 6. Generate overview grid for paper ----
    print("\nGenerating overview grid...")

    # Select a clean subset for the overview
    overview_keys = [
        '01_laplacian_prior',
        '02_enhanced_prior',
        '03_pgfe_output',
        '05_stream1_stage1',
        '06_stream2_entry',
        '07_cross_exchange1_s1',
        '09_stream1_stage2',
        '10_stream2_stage2',
        '13_stream1_stage3',
        '14_stream2_stage3',
        '15_agf_attention_map',
        '16_strip_pool_attention',
        '17_fusion_weight_s1',
        '18_fusion_weight_s2',
        '19_final_prediction',
    ]
    available = {k: all_feats[k] for k in overview_keys if k in all_feats}

    overview_path = os.path.join(OUTPUT_DIR, '20_overview_grid.png')
    save_side_by_side(available, overview_path, bg_img=img_np_cropped,
                      cols=5, figsize_per_cell=(3.5, 3.5), dpi=200)
    print(f"  [SAVED] 20_overview_grid.png  — 综合概览网格")

    # ---- 7. Write diagram insertion guide to file ----
    guide_path = os.path.join(OUTPUT_DIR, 'diagram_insertion_guide.txt')
    with open(guide_path, 'w', encoding='utf-8') as gf:
        gf.write("=" * 70 + "\n")
        gf.write("PHR-Net Feature Map Insertion Guide for Figure 2\n")
        gf.write("=" * 70 + "\n\n")
        gf.write("Data flow: Input X -> [PGFE] -> [PGCA Dual-Stream Backbone] -> [AGF] -> Output Y_hat\n\n")

        gf.write("Site 1 - PGFE Module (left side, input)\n")
        gf.write("  Insert at: inside/below PGFE box\n")
        gf.write("  Images: 01_laplacian_prior.png, 02_enhanced_prior.png, 03_pgfe_output.png\n")
        gf.write("  Purpose: Show 3-stage cascade: zero-param prior -> learnable enhancement -> fusion\n\n")

        gf.write("Site 2 - Stream 1 Full-Res Detail Stream (upper middle)\n")
        gf.write("  Insert at: above Stream 1 path, one per stage\n")
        gf.write("  Images: 05_stream1_stage1.png, 09_stream1_stage2.png, 13_stream1_stage3.png\n")
        gf.write("  Purpose: Prove full HxW resolution preserved across all stages\n\n")

        gf.write("Site 3 - Stream 2 Downsampled Context Stream (lower middle)\n")
        gf.write("  Insert at: below Stream 2 path\n")
        gf.write("  Images: 06_stream2_entry.png, 10_stream2_stage2.png, 14_stream2_stage3.png\n")
        gf.write("  Purpose: Show semantic enrichment at half resolution\n\n")

        gf.write("Site 4 - Cross-Stream Residual Exchange (middle, between streams)\n")
        gf.write("  Insert at: next to PGEB exchange arrows\n")
        gf.write("  Images: 07_cross_exchange1_s1.png, 08_cross_exchange1_s2.png\n")
        gf.write("  Purpose: Visualize bidirectional cross-scale information flow\n\n")

        gf.write("Site 5 - AGF Attention Fusion Module (right side, before output)\n")
        gf.write("  Insert at: inside/beside AGF box\n")
        gf.write("  Images: 15_agf_attention_map.png, 16_strip_pool_attention.png, 17_fusion_weight_s1.png, 18_fusion_weight_s2.png\n")
        gf.write("  Purpose: Show LKA attention refinement + strip pooling + adaptive gating\n\n")

        gf.write("Site 6 - Final Output (far right)\n")
        gf.write("  Insert at: next to Y_hat arrow\n")
        gf.write("  Images: 19_final_prediction.png\n")
        gf.write("  Purpose: Show final segmentation result\n\n")

        gf.write("-" * 70 + "\n")
        gf.write("Recommendation: Place 1-3 small images at each of the 6 sites\n")
        gf.write("Total: ~15 feature maps covering the full PGFE->PGCA->AGF data flow\n\n")
        gf.write("draw.io tips:\n")
        gf.write("  1. Drag generated PNGs onto the draw.io canvas\n")
        gf.write("  2. Use Crop to extract square regions\n")
        gf.write("  3. Add colored dashed borders + numbered labels\n")
        gf.write("  4. Connect feature maps to framework diagram with arrows\n")

    print(f"Diagram insertion guide saved to: {guide_path}")

    # ---- 8. Save a JSON mapping for programmatic use ----
    import json
    mapping = {
        "input_image": "00_input_image.png",
        "pgfe_module": {
            "laplacian_prior": "01_laplacian_prior.png",
            "enhanced_prior": "02_enhanced_prior.png",
            "pgfe_output": "03_pgfe_output.png",
            "position": "PGFE模块内部/下方, 展示三阶段级联过程",
        },
        "pgca_stream1": {
            "stage1": "05_stream1_stage1.png",
            "stage2": "09_stream1_stage2.png",
            "stage3": "13_stream1_stage3.png",
            "position": "Stream 1 路径上方, 展示全分辨率保持",
        },
        "pgca_stream2": {
            "entry": "06_stream2_entry.png",
            "stage2": "10_stream2_stage2.png",
            "stage3": "14_stream2_stage3.png",
            "position": "Stream 2 路径下方, 展示上下文语义提取",
        },
        "pgca_cross_exchange": {
            "exchange1_s1": "07_cross_exchange1_s1.png",
            "exchange1_s2": "08_cross_exchange1_s2.png",
            "position": "跨流交换箭头旁边, 展示双向信息流动",
        },
        "agf_module": {
            "lka_attention": "15_agf_attention_map.png",
            "strip_pool_attention": "16_strip_pool_attention.png",
            "fusion_weight_s1": "17_fusion_weight_s1.png",
            "fusion_weight_s2": "18_fusion_weight_s2.png",
            "position": "AGF模块内部/两侧, 展示注意力精炼和自适应融合",
        },
        "output": {
            "final_prediction": "19_final_prediction.png",
            "position": "最终输出 Ŷ 箭头旁",
        },
    }
    mapping_path = os.path.join(OUTPUT_DIR, 'feature_map_mapping.json')
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nMapping saved to: {mapping_path}")

    print("\n" + "=" * 60)
    print(f"All feature maps saved to: {OUTPUT_DIR}")
    print(f"Total: {len(all_feats) + 1} images (including input image and overview grid)")
    print("=" * 60)


if __name__ == "__main__":
    main()
