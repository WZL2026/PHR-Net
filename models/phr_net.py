import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dscnet import DSConv_pro


# ==============================================================================
# 1. 基础组件: Standard Conv Block (标准的单流残差块)
# ==============================================================================
class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.relu(out)


# ==============================================================================
# 1.0 基础组件: Factorized Conv Block (1x3 + 3x1 空间可分离残差块)
# ==============================================================================
class FactorizedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1_h = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2_h = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1_h(x)
        out = self.conv2_h(out)
        out += residual
        return self.relu(out)


# ==============================================================================
# 1.0.5 DSConvBlock: tri-branch snake conv (replaces FactorizedConvBlock)
#    Three parallel branches: regular 3x3 + DSConv_x (horizontal snake) +
#    DSConv_y (vertical snake) → fused via 1x1 conv.
#    Follows DSCNet's proven multi-branch design.
# ==============================================================================
class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9,
                 extend_scope=1.0, if_offset=True):
        super().__init__()
        # Branch 1: regular 3x3 conv
        self.branch_reg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Branch 2: DSConv along x-axis (horizontal snake)
        self.branch_dsx = DSConv_pro(
            in_channels, out_channels, kernel_size, extend_scope, 0, if_offset)
        # Branch 3: DSConv along y-axis (vertical snake)
        self.branch_dsy = DSConv_pro(
            in_channels, out_channels, kernel_size, extend_scope, 1, if_offset)
        # Fuse 3 branches
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Second conv layer (standard 3x3, matching FactorizedConvBlock depth)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        r = self.branch_reg(x)
        dx = self.branch_dsx(x)
        dy = self.branch_dsy(x)
        out = self.fuse(torch.cat([r, dx, dy], dim=1))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


def factorized_conv3x3(in_ch, out_ch, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1), bias=bias),
        nn.Conv2d(out_ch, out_ch, (3, 1), padding=(1, 0), bias=bias),
    )


# ==============================================================================
# Strip Pooling
# ==============================================================================
class StripPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), bias=False)
        self.conv2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False)
        self.conv3 = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(self.pool1(x)).expand(-1, -1, h, w)
        x2 = self.conv2(self.pool2(x)).expand(-1, -1, h, w)
        out = self.conv3(x1 + x2)
        return x * self.sigmoid(out)


class ResidualStripFusionHead(nn.Module):
    def __init__(self, in_channels_high=64, in_channels_low=128, out_classes=1, use_rsfh=True):
        super().__init__()
        self.use_rsfh = use_rsfh
        self.smooth_up = nn.Sequential(
            factorized_conv3x3(in_channels_low, in_channels_high, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels_high * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.fusion_conv = nn.Sequential(
            factorized_conv3x3(in_channels_high * 2, in_channels_high, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True)
        )
        self.strip_pool = StripPooling(in_channels_high)
        self.residual_conv = nn.Conv2d(in_channels_high, out_classes, kernel_size=1)

    def forward(self, s1, s2, out1, out2):
        s2_up_feat = F.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        s2_up_feat = self.smooth_up(s2_up_feat)
        concat_feat = torch.cat([s1, s2_up_feat], dim=1)
        weight_map = self.gate(concat_feat)
        w1 = weight_map[:, 0:1, :, :]
        w2 = weight_map[:, 1:2, :, :]
        out2_up = F.interpolate(out2, size=out1.shape[2:], mode='bilinear', align_corners=False)
        logit_base = w1 * out1 + w2 * out2_up
        if not self.use_rsfh:
            return logit_base
        fused_feat = self.fusion_conv(concat_feat)
        fused_feat = self.strip_pool(fused_feat)
        feature_residual = self.residual_conv(fused_feat)
        return logit_base + feature_residual


# ==============================================================================
# Large Kernel Attention
# ==============================================================================
class MacroscopicLKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial_d1 = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim, dilation=1)
        self.conv_spatial_d3 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_d1 = self.conv_spatial_d1(attn)
        attn_d3 = self.conv_spatial_d3(attn)
        attn = attn_d1 + attn_d3
        attn = self.conv1(attn)
        return u * attn


# ==============================================================================
# Dual-Scope Selective Block
# ==============================================================================
class DualScopeSelectiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mid_channels = out_channels
        self.input_proj = nn.Conv2d(in_channels, self.mid_channels, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(self.mid_channels)
        self.selector = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels // 4, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )
        self.scope_micro = nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1, groups=self.mid_channels, bias=False)
        self.scope_macro = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=5, padding=6, dilation=3, groups=self.mid_channels, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        self.ska = nn.Sequential(
            nn.Conv2d(self.mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        feat = self.bn_in(self.input_proj(x))
        decision_map = self.selector(feat)
        micro_weight, macro_weight = torch.chunk(decision_map, 2, dim=1)
        out_micro = self.scope_micro(feat)
        out_macro = self.scope_macro(feat)
        out = (out_micro * micro_weight) + (out_macro * macro_weight)
        return self.ska(out) + residual


# ==============================================================================
# Tri-Scope Selective Block: micro (3x3 DW) + snake (DSConv x+y) + macro (dilated)
#   3-scope variant of DSSB — adds direction-adaptive snake scope
# ==============================================================================
class TriScopeSelectiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super().__init__()
        self.mid_channels = out_channels
        self.input_proj = nn.Conv2d(in_channels, self.mid_channels, 1, bias=False)
        self.bn_in = nn.BatchNorm2d(self.mid_channels)

        # Selector outputs 3 weights (micro / snake / macro)
        self.selector = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels // 4, 3, 3, padding=1),
            nn.Softmax(dim=1)
        )

        # Micro scope: 3x3 depthwise
        self.scope_micro = nn.Conv2d(self.mid_channels, self.mid_channels, 3,
                                     padding=1, groups=self.mid_channels, bias=False)

        # Snake scope: DSConv x (horizontal) + DSConv y (vertical), each half channels
        self.scope_snake_x = DSConv_pro(self.mid_channels, self.mid_channels // 2,
                                        kernel_size, 1.0, 0, True)
        self.scope_snake_y = DSConv_pro(self.mid_channels, self.mid_channels // 2,
                                        kernel_size, 1.0, 1, True)

        # Macro scope: 5x5 dilated depthwise
        self.scope_macro = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=5,
                      padding=6, dilation=3, groups=self.mid_channels, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        self.ska = nn.Sequential(
            nn.Conv2d(self.mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        feat = self.bn_in(self.input_proj(x))
        decision_map = self.selector(feat)
        w_micro, w_snake, w_macro = torch.chunk(decision_map, 3, dim=1)

        out_micro = self.scope_micro(feat)
        out_snake_x = self.scope_snake_x(feat)
        out_snake_y = self.scope_snake_y(feat)
        out_snake = torch.cat([out_snake_x, out_snake_y], dim=1)

        out_macro = self.scope_macro(feat)
        out = (out_micro * w_micro) + (out_snake * w_snake) + (out_macro * w_macro)
        return self.ska(out) + residual


# ==============================================================================
# BlurPool
# ==============================================================================
class BlurPool(nn.Module):
    def __init__(self, channels, stride=2):
        super().__init__()
        kernel = torch.tensor([1., 2., 1.])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.groups = channels
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=self.stride, padding=1, groups=self.groups)


# ==============================================================================
# DynamicPriorEnhancement (DPE)
# ==============================================================================
class DynamicPriorEnhancement(nn.Module):
    def __init__(self, in_channels=3, use_dpe=True):
        super().__init__()
        self.use_dpe = use_dpe
        self.in_channels = in_channels
        lap_k = torch.tensor([[-1., -1., -1.],
                              [-1.,  8., -1.],
                              [-1., -1., -1.]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap_k.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        if not self.use_dpe:
            return torch.zeros_like(x)
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        lap = F.conv2d(x, self.lap_kernel, padding=0, groups=self.in_channels)
        return lap


# ==============================================================================
# Dynamic Prior Selector (DPSS) — kept for compatibility, not used in test variants
# ==============================================================================
class DynamicPriorSelector(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, prior_feat):
        weight = self.attention(prior_feat)
        return prior_feat + prior_feat * weight


# ==============================================================================
# Cross-Stream Residual Exchange
# ==============================================================================
class CrossStreamResidual(nn.Module):
    """Simple residual cross-stream exchange: S1 <-> S2 bidirectional.

    S1 -> down -> add to S2 (high-resolution detail -> low-resolution context)
    S2 -> up   -> add to S1 (low-resolution context -> high-resolution detail)
    """

    def __init__(self, in_channels_high, in_channels_low):
        super().__init__()
        self.down_conv = nn.Sequential(
            BlurPool(in_channels_high, stride=2),
            factorized_conv3x3(in_channels_high, in_channels_low, bias=False),
            nn.BatchNorm2d(in_channels_low),
            nn.ReLU(inplace=True)
        )
        self.up_conv = nn.Sequential(
            factorized_conv3x3(in_channels_low, in_channels_high, bias=False),
            nn.BatchNorm2d(in_channels_high),
            nn.ReLU(inplace=True)
        )

    def forward(self, high_feat, low_feat):
        high_to_low = self.down_conv(high_feat)
        out_low = low_feat + high_to_low
        low_up = F.interpolate(low_feat, size=high_feat.shape[2:],
                               mode='bilinear', align_corners=False)
        low_to_high = self.up_conv(low_up)
        out_high = high_feat + low_to_high
        return out_high, out_low


# ==============================================================================
# Module A: Explicit Prior Enhancement (DPE + DPSS)
# ==============================================================================
class ModuleA_PriorEnhancement(nn.Module):
    """Module A: DPE + DPSS — explicit Laplacian prior with learned amplifier.

    When use_a=True:  Laplacian prior -> project -> DPSS amplify -> merge with stem
    When use_a=False: Zero prior -> project -> merge with stem (learned-conv fallback)
    """

    def __init__(self, in_channels=3, use_a=True):
        super().__init__()
        self.use_a = use_a

        # DPE: Laplacian edge extractor
        self.dpe = DynamicPriorEnhancement(in_channels, use_dpe=use_a)

        # Prior projection: Laplacian -> 64ch feature
        self.lap_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # DPSS: residual amplifier for prior features
        self.dpss = DynamicPriorSelector(channels=64)

        # Main stem: raw image -> 64ch feature
        self.enc_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        lap_prior = self.dpe(x)                          # Laplacian or zeros
        main_feat = self.enc_stem(x)                     # stem feature
        prior_feat = self.lap_proj(lap_prior)            # project prior
        if self.use_a:
            prior_feat = self.dpss(prior_feat)           # amplify
        return main_feat + prior_feat, prior_feat        # (S1 input, prior for gate)


# ==============================================================================
# PHR-Net: Dual-Stream with Module A (prior) + Module B (attention+fusion)
#
#   Ablation (2x2):
#     use_a = DPSS+DPE    (explicit prior enhancement)
#     use_b = LKA+RSFH    (large-kernel attention + strip fusion)
#
#   Backward-compat kwargs (deprecated but accepted):
#     use_dpe, use_dpss  -> map to use_a
#     use_lka, use_rsfh  -> map to use_b
#     use_pgeb           -> silently ignored (PGEB removed)
#
#   dsconv_mode:
#     "s1"       -> DSConvBlock in Stream 1 (default)
#     "dual"     -> DSConvBlock in both streams
#     "triscope" -> TriScopeSelectiveBlock replaces DSSB
#     "none"     -> FactorizedConvBlock everywhere
#     "nodpe_s1" -> DSConv in S1 + DPE disabled
# ==============================================================================
class PHR_Net(nn.Module):
    def __init__(self, in_channels=3, out_classes=1,
                 # New API
                 use_a=None, use_b=None,
                 backbone="hrnet",  # "hrnet" (dual-stream high-res) or "unet" (encoder-decoder deep downsample)
                 # Deprecated kwargs (backward compat)
                 use_dpe=True, use_dpss=True, use_lka=True, use_rsfh=True,
                 use_pgca=True,  # Prior-Guided Channel Attention on Stream 2
                 lka_mode="deep", lka_upsample="bilinear",
                 lka_fusion="gate",
                 dsconv_mode="none", dsconv_kernel_size=9):
        super().__init__()

        # Backward compat: if use_a/use_b not explicitly set, derive from old flags
        if use_a is None:
            use_a = use_dpe and use_dpss
        if use_b is None:
            use_b = use_lka and use_rsfh
        self.use_pgca = use_pgca

        assert lka_mode in ("deep", "direct"), f"Unknown lka_mode: {lka_mode}"
        assert lka_upsample in ("bilinear", "transposed"), f"Unknown lka_upsample: {lka_upsample}"
        assert lka_fusion in ("gate", "residual"), f"Unknown lka_fusion: {lka_fusion}"
        assert dsconv_mode in ("none", "s1", "dual", "triscope", "nodpe_s1"), \
            f"Unknown dsconv_mode: {dsconv_mode}"

        # DSConv mode
        use_dsconv_s1 = dsconv_mode in ("s1", "dual", "nodpe_s1")
        use_dsconv_s2 = dsconv_mode == "dual"
        use_dsconv_triscope = dsconv_mode == "triscope"

        self.use_a = use_a
        self.use_b = use_b
        self.backbone = backbone
        self.lka_mode = lka_mode
        self.lka_upsample = lka_upsample
        self.lka_fusion = lka_fusion
        self.dsconv_mode = dsconv_mode

        # ---- Module A: Prior Enhancement (DPE + DPSS) ----
        _use_a_actual = use_a and dsconv_mode != "nodpe_s1"
        self.module_a = ModuleA_PriorEnhancement(in_channels, use_a=_use_a_actual)

        if backbone == "hrnet":
            # ==================================================================
            # HRNet: Dual-Stream High-Resolution Backbone
            #   S1: 64ch @ H×W  (high-res detail)

            # Prior-Guided Channel Attention on Stream 2 (PGCA-S2)
            #   Uses global prior statistics to recalibrate S2 channels
            #   gate ∈ [0,1]^128 — applied once at S2 entry, after stream2_conv1
            if self.use_pgca:
                self.pgca_s2 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 128),
                    nn.Sigmoid(),
                )
            else:
                self.pgca_s2 = None
            #   S2: 128ch @ H/2×W/2  (half-res context)
            #   Cross-stream residual exchange for bidirectional information flow
            # ==================================================================

            # ---- Stream 1 ----
            if use_dsconv_triscope:
                self.stream1_dssb = TriScopeSelectiveBlock(64, 64, dsconv_kernel_size)
            else:
                self.stream1_dssb = DualScopeSelectiveBlock(64, 64)

            self.stream1_conv1 = StandardConvBlock(64, 64)

            if use_dsconv_s1:
                self.stream1_conv2 = DSConvBlock(64, 64, dsconv_kernel_size)
                self.stream1_conv3 = DSConvBlock(64, 64, dsconv_kernel_size)
            else:
                self.stream1_conv2 = FactorizedConvBlock(64, 64)
                self.stream1_conv3 = FactorizedConvBlock(64, 64)

            # ---- Stream 2 transition ----
            self.transition_s2 = nn.Sequential(
                BlurPool(64, stride=2),
                factorized_conv3x3(64, 128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

            if use_dsconv_s2:
                self.stream2_conv1 = DSConvBlock(128, 128, dsconv_kernel_size)
                self.stream2_conv2 = DSConvBlock(128, 128, dsconv_kernel_size)
                self.stream2_conv3 = DSConvBlock(128, 128, dsconv_kernel_size)
            else:
                self.stream2_conv1 = FactorizedConvBlock(128, 128)
                self.stream2_conv2 = FactorizedConvBlock(128, 128)
                self.stream2_conv3 = FactorizedConvBlock(128, 128)

            # ---- Cross-Stream Residual Exchange (replaces PGEB) ----
            self.cross_1_2 = CrossStreamResidual(
                in_channels_high=64, in_channels_low=128)
            self.cross_1_2_2 = CrossStreamResidual(
                in_channels_high=64, in_channels_low=128)

            # ---- Module B (LKA part): deep attention path ----
            self.lka_strength = nn.Parameter(torch.tensor(1.0))

            if lka_mode == "deep":
                self.s2_down_deep = nn.Sequential(
                    BlurPool(128, stride=2),
                    factorized_conv3x3(128, 256, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                self.s2_bottleneck = FactorizedConvBlock(256, 256)
                self.s2_lka_deep = MacroscopicLKA(dim=256)

                if lka_upsample == "transposed":
                    self.s2_deep_upsample = nn.ConvTranspose2d(
                        256, 256, kernel_size=2, stride=2, bias=False)
                else:
                    self.s2_deep_upsample = None

                self.s2_deep_gate = nn.Sequential(
                    factorized_conv3x3(256, 128, bias=False),
                    nn.BatchNorm2d(128),
                    nn.Tanh()
                )
            else:
                self.s2_lka_direct = MacroscopicLKA(dim=128)
                self.s2_direct_gate = nn.Sequential(
                    factorized_conv3x3(128, 128, bias=False),
                    nn.BatchNorm2d(128),
                    nn.Tanh()
                )

        elif backbone == "unet":
            # ==================================================================
            # UNet: Encoder-Decoder Backbone (deep downsampling → H/8)
            #   Loses high-frequency detail through progressive pooling
            #   Used as ablation control: quantify the value of HRNet's high-res features
            #
            #   Encoder: 64→96→128→128 (H→H/2→H/4→H/8)
            #   Decoder: 128→128→64 (H/4→H/2→H) with skip connections
            #   Output:  s1=64ch@H×W, s2=128ch@H/2×W/2 (compatible with heads)
            # ==================================================================

            # ---- Encoder ----
            self.unet_enc_dssb = DualScopeSelectiveBlock(64, 64)

            self.unet_down1 = nn.Sequential(
                BlurPool(64, stride=2),
                factorized_conv3x3(64, 96, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            )
            self.unet_enc1_conv1 = StandardConvBlock(96, 96)
            self.unet_enc1_conv2 = FactorizedConvBlock(96, 96)

            self.unet_down2 = nn.Sequential(
                BlurPool(96, stride=2),
                factorized_conv3x3(96, 128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.unet_enc2_conv1 = FactorizedConvBlock(128, 128)
            self.unet_enc2_conv2 = FactorizedConvBlock(128, 128)

            self.unet_down3 = nn.Sequential(
                BlurPool(128, stride=2),
                factorized_conv3x3(128, 128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.unet_bottleneck_conv1 = FactorizedConvBlock(128, 128)
            self.unet_bottleneck_conv2 = FactorizedConvBlock(128, 128)

            # ---- Module B: LKA at bottleneck (H/8) ----
            self.unet_lka = MacroscopicLKA(dim=128)
            self.unet_lka_strength = nn.Parameter(torch.tensor(1.0))
            self.unet_lka_gate = nn.Sequential(
                factorized_conv3x3(128, 128, bias=False),
                nn.BatchNorm2d(128),
                nn.Tanh()
            )

            # ---- Decoder ----
            # dec2: H/8 → H/4 (upsample + skip enc2: 128+128=256→128)
            self.unet_dec2_up = nn.Sequential(
                factorized_conv3x3(256, 128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.unet_dec2_refine = FactorizedConvBlock(128, 128)

            # dec1: H/4 → H/2 (upsample + skip enc1: 128+96=224→128) → s2
            self.unet_dec1_up = nn.Sequential(
                factorized_conv3x3(224, 128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.unet_dec1_refine = FactorizedConvBlock(128, 128)

            # dec0: H/2 → H (upsample + skip enc0: 128+64=192→64) → s1
            self.unet_dec0_up = nn.Sequential(
                factorized_conv3x3(192, 64, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.unet_dec0_refine = FactorizedConvBlock(64, 64)

        else:
            raise ValueError(f"Unknown backbone: '{backbone}'. Expected 'hrnet' or 'unet'.")

        # ---- Heads ----
        self.head1 = nn.Conv2d(64, out_classes, kernel_size=1)
        self.head2 = nn.Conv2d(128, out_classes, kernel_size=1)

        # ---- Module B (RSFH part): strip pooling fusion ----
        self.fusion_head = ResidualStripFusionHead(
            in_channels_high=64, in_channels_low=128,
            out_classes=out_classes, use_rsfh=use_b  # RSFH follows use_b
        )

    def forward(self, x):
        # ---- 1. Module A: Prior Enhancement ----
        x_feat, prior_feat = self.module_a(x)  # (B,64,H,W), (B,64,H,W)

        if self.backbone == "hrnet":
            # ==================================================================
            # HRNet: Dual-Stream High-Resolution Path
            # ==================================================================

            s1 = x_feat
            if self.use_a:
                s1 = self.stream1_dssb(s1)
            s1 = self.stream1_conv1(s1)

            # ---- Stream 2 transition ----
            s2 = self.transition_s2(s1)
            s2 = self.stream2_conv1(s2)

            # ---- PGCA-S2: prior-guided channel attention at S2 entry ----
            if self.use_pgca:
                gate_s2 = self.pgca_s2(prior_feat)  # (B, 128)
                gate_s2 = gate_s2.view(-1, 128, 1, 1)
                s2 = s2 * gate_s2

            # ---- Cross-Stream Residual Exchange (x2) ----
            s1, s2 = self.cross_1_2(s1, s2)
            s1 = self.stream1_conv2(s1)
            s2 = self.stream2_conv2(s2)
            s1, s2 = self.cross_1_2_2(s1, s2)
            s1 = self.stream1_conv3(s1)
            s2 = self.stream2_conv3(s2)

            # ---- Module B (LKA): deep-path attention ----
            if self.use_b and self.lka_mode == "deep":
                s2_deep = self.s2_down_deep(s2)                # 1/2 -> 1/4 (256ch)
                s2_deep = self.s2_bottleneck(s2_deep)
                s2_deep = self.s2_lka_deep(s2_deep)            # LKA @ 1/4
                if self.s2_deep_upsample is not None:
                    s2_deep_up = self.s2_deep_upsample(s2_deep)
                else:
                    s2_deep_up = F.interpolate(
                        s2_deep, scale_factor=2, mode='bilinear', align_corners=False)
                s2_attn = self.s2_deep_gate(s2_deep_up)        # (B,128,H/2,W/2), tanh
                s2_attn = F.interpolate(
                    s2_attn, size=s2.shape[2:], mode='bilinear', align_corners=False)

                if self.lka_fusion == "residual":
                    s2 = s2 + self.lka_strength * s2_attn
                else:
                    s2 = s2 * (1.0 + self.lka_strength * s2_attn)
            elif self.lka_mode == "deep":
                # use_b=False: skip LKA, still run bottleneck for feature transform
                s2_deep = self.s2_down_deep(s2)
                s2_deep = self.s2_bottleneck(s2_deep)
                s2_deep_up = F.interpolate(
                    s2_deep, scale_factor=2, mode='bilinear', align_corners=False)
                s2_attn = self.s2_deep_gate(s2_deep_up)
                s2_attn = F.interpolate(
                    s2_attn, size=s2.shape[2:], mode='bilinear', align_corners=False)
                s2 = s2 * (1.0 + self.lka_strength * s2_attn)
            else:
                # Direct mode (deprecated)
                if self.use_b:
                    s2_lka = self.s2_lka_direct(s2)
                    s2_attn = self.s2_direct_gate(s2_lka)
                    if self.lka_fusion == "residual":
                        s2 = s2 + self.lka_strength * s2_attn
                    else:
                        s2 = s2 * (1.0 + self.lka_strength * s2_attn)

        elif self.backbone == "unet":
            # ==================================================================
            # UNet: Encoder-Decoder with Deep Downsampling (→ H/8 bottleneck)
            #   High-frequency detail deliberately lost through progressive pooling
            # ==================================================================

            # ---- Encoder ----
            if self.use_a:
                enc0 = self.unet_enc_dssb(x_feat)   # (B, 64, H, W)
            else:
                enc0 = x_feat                       # skip DSSB when Module A is off

            enc1 = self.unet_down1(enc0)         # (B, 96, H/2, W/2)
            enc1 = self.unet_enc1_conv1(enc1)
            enc1 = self.unet_enc1_conv2(enc1)

            enc2 = self.unet_down2(enc1)         # (B, 128, H/4, W/4)
            enc2 = self.unet_enc2_conv1(enc2)
            enc2 = self.unet_enc2_conv2(enc2)

            bottleneck = self.unet_down3(enc2)   # (B, 128, H/8, W/8)
            bottleneck = self.unet_bottleneck_conv1(bottleneck)
            bottleneck = self.unet_bottleneck_conv2(bottleneck)

            # ---- Module B: LKA at bottleneck ----
            if self.use_b:
                b_lka = self.unet_lka(bottleneck)
                b_attn = self.unet_lka_gate(b_lka)
                bottleneck = bottleneck * (1.0 + self.unet_lka_strength * b_attn)

            # ---- Decoder: upsample + skip → s2(H/2), s1(H) ----
            up2 = F.interpolate(bottleneck, size=enc2.shape[2:],
                                mode='bilinear', align_corners=False)
            cat2 = torch.cat([up2, enc2], dim=1)      # 128+128=256
            dec2 = self.unet_dec2_up(cat2)             # → 128ch @ H/4
            dec2 = self.unet_dec2_refine(dec2)

            up1 = F.interpolate(dec2, size=enc1.shape[2:],
                                mode='bilinear', align_corners=False)
            cat1 = torch.cat([up1, enc1], dim=1)      # 128+96=224
            s2 = self.unet_dec1_up(cat1)               # → 128ch @ H/2
            s2 = self.unet_dec1_refine(s2)

            up0 = F.interpolate(s2, size=enc0.shape[2:],
                                mode='bilinear', align_corners=False)
            cat0 = torch.cat([up0, enc0], dim=1)      # 128+64=192
            s1 = self.unet_dec0_up(cat0)               # → 64ch @ H
            s1 = self.unet_dec0_refine(s1)

        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        # ---- 5. Heads + Module B (RSFH): strip pooling fusion ----
        out1 = self.head1(s1)
        out2 = self.head2(s2)
        out_fused = self.fusion_head(s1, s2, out1, out2)

        if self.training:
            return out_fused, [out1, out2]
        else:
            return out_fused


# ==============================================================================
# Self-check
# ==============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("PHR-Net Variants — Architecture Check (LKA + DSConv)")
    print("=" * 60)

    # LKA variants
    lka_variants = [
        ("Baseline (deep+bilin+gate)", {}),
        ("A: transposed up",           {"lka_upsample": "transposed"}),
        ("B: direct LKA at 1/2",       {"lka_mode": "direct"}),
        ("C: residual fusion",         {"lka_fusion": "residual"}),
    ]

    # DSConv variants (all use baseline LKA config)
    dsconv_variants = [
        ("DSConv-S1",        {"dsconv_mode": "s1"}),
        ("DSConv-Dual",      {"dsconv_mode": "dual"}),
        ("DSConv-TriScope",  {"dsconv_mode": "triscope"}),
        ("DSConv-NoDPE_S1",  {"dsconv_mode": "nodpe_s1"}),
    ]

    dummy = torch.randn(2, 3, 256, 256).to(device)
    all_ok = True

    for name, kwargs in lka_variants + dsconv_variants:
        try:
            m = PHR_Net(in_channels=3, out_classes=1, **kwargs).to(device)
            m.train()
            out_fused, [out1, out2] = m(dummy)
            loss = out_fused.sum()
            loss.backward()
            m.eval()
            _ = m(dummy)
            p = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"  [PASS] {name:<25s} | {p:.2f}M | "
                  f"fused={str(out_fused.shape):22s} out1={str(out1.shape):20s} out2={str(out2.shape):20s}")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name:<25s} | {e}")
            traceback.print_exc()
            all_ok = False

    # Verify NoDPE variant produces zero prior
    m_nodpe = PHR_Net(in_channels=3, out_classes=1, dsconv_mode="nodpe_s1")
    lp = m_nodpe.module_a.dpe(dummy)
    assert lp.abs().sum() == 0, "nodpe_s1 should produce zero lap_prior"
    print(f"  [PASS] NoDPE zero-prior verified   | sum={lp.abs().sum().item():.1f}")

    # Verify TriScope selector outputs 3 weights
    m_tri = PHR_Net(in_channels=3, out_classes=1, dsconv_mode="triscope")
    assert isinstance(m_tri.stream1_dssb, TriScopeSelectiveBlock), \
        "triscope mode should use TriScopeSelectiveBlock"
    print(f"  [PASS] TriScope DSSB verified")

    print(f"\n{'All tests passed!' if all_ok else 'SOME TESTS FAILED!'}")
