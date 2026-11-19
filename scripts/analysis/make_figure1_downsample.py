# -*- coding: utf-8 -*-
"""生成论文图1：左侧"下采样导致高频细节丢失"示意 + 右侧重型模型列举（三种版式可选）。

左侧：原图（红框标出放大区）+ 该区域在 1x/2x/4x/8x 下采样后放大回原尺寸的结果，
      直观展示宽度仅数像素的末梢血管如何随下采样消失。
右侧：只做定性列举 / 参数代价展示，**不放任何精度指标**（精度对比归第 4 节）。

三种版式（--variant，默认 ladder）：
  ladder  参数阶梯：每个模型一条对数标尺横条 + 3 M 边缘预算线（当前论文采用）
  family  家族分组：按 CNN 编码器-解码器 / 全局注意力 / 状态空间模型 三组，每组一句话总括
  list    两行一条：模型 / 出处 / 参数量 / 核心设计 + 针对的问题

依赖：numpy / Pillow / matplotlib（不 import torch）。
输出：results/paper_figures/Figure1_Downsample_Loss_and_Budget.{png,pdf}
"""
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\Project\PHRNet")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

OUT = ROOT / "results" / "paper_figures" / "Figure1_Downsample_Loss_and_Budget"

IMG = ROOT.parent / "Dataset" / "STARE" / "images" / "im0082.png"
MASK_DIR = ROOT.parent / "Dataset" / "STARE" / "masks"

# ---------------------------------------------------------------- 右侧数据
# (模型, 出处（作者 + 发表处/年份）, 参数量, 核心设计, 针对的问题, 是否本文)
# 只保留参数量 >= 50 M 的重型模型（实测值，见 results/summary/summary_all_20261017_202311.csv）
ARCH_ROWS = [
    ("VM-UNet", "Ruan & Xiang, arXiv'24", "44.27 M",
     "state-space (Mamba) encoder-decoder",
     "linear-complexity global modelling", False),
    ("HM-Mamba", "Wang, Entropy'25", "51.12 M",
     "hierarchical multi-scale Mamba + Fourier blocks",
     "vessel continuity across scales", False),
    ("Swin-UMamba", "Liu, MICCAI'24", "59.89 M",
     "Swin-UNet + Mamba (VSS) blocks",
     "global context via ImageNet pretraining", False),
    ("TransUNet", "Chen, arXiv'21", "105.32 M",
     "ViT encoder + CNN decoder",
     "long-range dependency over the whole image", False),
    ("MambaVesselNet", "Xu, TOMM'25", "117.22 M",
     "CNN-Mamba hybrid encoder + bifocal decoder",
     "local texture and global vessel continuity", False),
    ("PHR-Net (ours)", "This work", "2.62 M",
     "dual-stream full-res backbone + Laplacian prior",
     "terminal-vessel detail within a 3 M budget", True),
]

# 家族分组版数据（颜色与图2 的家族配色一致：CNN 绿 / Transformer 金 / SSM 蓝）
FAMILIES = [
    dict(name="Global attention", color="#B8860B", tint="#FBF4E0",
         prange="105 M",
         note="whole-image dependency, cost grows with image size",
         models=[("TransUNet", "Chen, arXiv'21", "105.32 M")]),
    dict(name="State-space models", color="#2F5F8F", tint="#EAF0F7",
         prange="44-117 M",
         note="whole-image dependency at linear cost, still tens to hundreds of M",
         models=[("VM-UNet", "Ruan & Xiang, arXiv'24", "44.27 M"),
                 ("HM-Mamba", "Wang, Entropy'25", "51.12 M"),
                 ("Swin-UMamba", "Liu, MICCAI'24", "59.89 M"),
                 ("MambaVesselNet", "Xu, TOMM'25", "117.22 M")]),
]
OURS = ("PHR-Net (ours)", "This work", "2.62 M", "full resolution + zero-parameter prior")

FAMILY_OF = {m[0]: f["color"] for f in FAMILIES for m in f["models"]}
FAMILY_OF["PHR-Net (ours)"] = "#D62728"

RED = "#D62728"
BAR_COL = "#5B7C99"   # 基线统一色（不再按范式区分；列表里只剩一个 Transformer，颜色无信息量）


def _p(s):
    return float(s.replace("M", "").strip())


# ---------------------------------------------------------------- 左侧图像
def _thin_mask(gt):
    """近似"细血管"：GT 中不属于 3x3 全实心区域的像素（即宽度 1-2 px 的部分）。"""
    g = gt.astype(bool)
    solid = np.ones_like(g)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            solid &= np.roll(np.roll(g, dy, axis=0), dx, axis=1)
    return g & ~solid


def pick_roi(side=80):
    """选"以细血管为主"的窗口：细血管像素多、粗血管少、无黑边、避开视盘高亮区。"""
    cand = sorted(MASK_DIR.glob("im0082*.png"))
    if not cand:
        return (360, 160, side, side)
    gt = np.asarray(Image.open(cand[0]).convert("L")) > 127
    gray = np.asarray(Image.open(IMG).convert("L")).astype(np.float32)
    thin = _thin_mask(gt)
    solid = gt & ~thin
    H, W = thin.shape
    scored = []
    for y in range(60, H - side - 60, 8):
        for x in range(60, W - side - 60, 8):
            g_patch = gray[y:y + side, x:x + side]
            if (g_patch < 12).mean() > 0.01:        # 黑背景
                continue
            if g_patch.mean() > 150:                # 视盘高亮区
                continue
            n_thin = int(thin[y:y + side, x:x + side].sum())
            n_solid = int(solid[y:y + side, x:x + side].sum())
            if n_thin < 120 or n_solid > 0.35 * n_thin:
                continue                            # 要求以细血管为主
            scored.append((n_thin, x, y, n_solid))
    scored.sort(reverse=True)
    for s in scored[:5]:
        print(f"  candidate x={s[1]} y={s[2]} thin={s[0]} solid={s[3]}", flush=True)
    if not scored:
        return (360, 160, side, side)
    _, x, y, _ = scored[0]
    print(f"ROI selected: ({x}, {y}, {side}, {side})", flush=True)
    return (x, y, side, side)


ROI = pick_roi()

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})


def downsample_cycle(arr, factor):
    """下采样 factor 倍后再放大回原尺寸（模拟编码器逐级下采样的信息损失）。"""
    if factor == 1:
        return arr
    h, w = arr.shape[:2]
    small = Image.fromarray(arr).resize((max(1, w // factor), max(1, h // factor)),
                                        Image.LANCZOS)
    return np.asarray(small.resize((w, h), Image.LANCZOS))


# ---------------------------------------------------------------- 右侧三种版式
def draw_list(ax):
    """版式 1：两行一条，设计（黑）+ 针对的问题（灰斜体）。"""
    ax.text(0.0, 1.008,
            "Heavyweight architectures ($\\geq$ 25 M): source, design and the problem each targets",
            fontsize=15.0, fontweight="bold", ha="left", va="top")
    X_MODEL, X_PARAMS_R, X_DESIGN = 0.000, 0.330, 0.372
    y_head = 0.912
    for cx, txt, ha in [(X_MODEL, "Model / source", "left"),
                        (X_PARAMS_R, "Params", "right"),
                        (X_DESIGN, "Core design / targeted problem", "left")]:
        ax.text(cx, y_head, txt, fontsize=13.4, fontweight="bold", ha=ha, va="center")
    ax.plot([0.0, 1.0], [y_head - 0.044, y_head - 0.044], color="0.35", lw=1.1)

    dy, y_first = 0.109, y_head - 0.091
    for k, (name, src, params, design, problem, ours) in enumerate(ARCH_ROWS):
        yy = y_first - k * dy
        col = RED if ours else "black"
        ax.text(X_MODEL, yy + 0.022, name, fontsize=14.6, va="center", ha="left",
                color=col, fontweight="bold" if ours else "normal")
        ax.text(X_MODEL, yy - 0.029, src, fontsize=12.4, va="center", ha="left", color="#444444")
        ax.text(X_PARAMS_R, yy + 0.002, params, fontsize=13.8, va="center", ha="right",
                color=col, fontweight="bold" if ours else "normal")
        ax.text(X_DESIGN, yy + 0.022, design, fontsize=14.6, va="center", ha="left", color=col)
        ax.text(X_DESIGN, yy - 0.028, problem, fontsize=13.2, va="center", ha="left",
                color="#555555" if not ours else "#B03A3A", style="italic")


def draw_family(ax):
    """版式 2：按家族分组，每组一句总括，组内只列模型 / 出处 / 参数量。"""
    ax.text(0.0, 1.008,
            "Representative heavyweight architectures ($\\geq$ 25 M) by design family",
            fontsize=15.0, fontweight="bold", ha="left", va="top")

    y = 0.930
    head_h, char_h, row_h, pad, gap = 0.048, 0.040, 0.067, 0.012, 0.014
    for fam in FAMILIES:
        n = len(fam["models"])
        h = head_h + char_h + n * row_h + 2 * pad
        ax.add_patch(Rectangle((0.0, y - h), 1.0, h, transform=ax.transAxes,
                               facecolor=fam["tint"], edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((0.0, y - h), 0.006, h, transform=ax.transAxes,
                               facecolor=fam["color"], edgecolor="none", zorder=1))
        ax.text(0.016, y - 0.026, fam["name"], fontsize=14.2, fontweight="bold",
                color=fam["color"], va="center", zorder=2)
        ax.text(0.992, y - 0.026, fam["prange"], fontsize=12.6, color=fam["color"],
                ha="right", va="center", zorder=2)
        ax.text(0.016, y - 0.068, fam["note"], fontsize=11.8, color="#555555",
                style="italic", va="center", zorder=2)
        for j, (name, src, params) in enumerate(fam["models"]):
            yy = y - head_h - char_h - pad - 0.0335 - j * row_h
            ax.text(0.034, yy, name, fontsize=13.6, va="center", zorder=2)
            ax.text(0.300, yy, src, fontsize=11.6, color="#555555", va="center", zorder=2)
            ax.text(0.992, yy, params, fontsize=13.0, va="center", ha="right", zorder=2)
        y -= h + gap

    # 本文方法：单独一条红色
    y -= 0.002
    ax.add_patch(Rectangle((0.0, y - 0.075), 1.0, 0.075, transform=ax.transAxes,
                           facecolor="#FBECEC", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.0, y - 0.075), 0.006, 0.075, transform=ax.transAxes,
                           facecolor=RED, edgecolor="none", zorder=1))
    ax.text(0.016, y - 0.0375, OURS[0], fontsize=14.4, fontweight="bold",
            color=RED, va="center", zorder=2)
    ax.text(0.300, y - 0.0375, OURS[1], fontsize=11.6, color="#B03A3A", va="center", zorder=2)
    ax.text(0.992, y - 0.0375, OURS[2], fontsize=13.6, fontweight="bold",
            color=RED, va="center", ha="right", zorder=2)
    ax.text(0.505, y - 0.0375, OURS[3], fontsize=11.8, color="#B03A3A",
            style="italic", va="center", zorder=2)


def draw_ladder(ax):
    """版式 3：参数阶梯（横条长度与参数量成正比，线性标尺）。"""
    ax.text(0.0, 1.008,
            "Parameter cost of representative architectures ($\\geq$ 40 M)",
            fontsize=15.0, fontweight="bold", ha="left", va="top")

    X_NAME, X_SRC, X_BAR0, X_BAR1, X_VAL = 0.0, 0.215, 0.405, 0.862, 0.995
    vmax = 120.0

    def xpos(p):
        return X_BAR0 + (p / vmax) * (X_BAR1 - X_BAR0)

    rows = sorted([r for r in ARCH_ROWS if not r[5]], key=lambda r: _p(r[2]))
    rows.append(ARCH_ROWS[-1])

    y0, dy, gap = 0.860, 0.118, 0.055
    for k, (name, src, params, design, problem, ours) in enumerate(rows):
        yy = y0 - k * dy
        if ours:                       # 与基线之间留白
            yy -= gap
        col = RED if ours else BAR_COL
        ax.text(X_NAME, yy, name, fontsize=13.6, va="center",
                color=RED if ours else "black",
                fontweight="bold" if ours else "normal")
        ax.text(X_SRC, yy, src, fontsize=10.0, color="#555555", va="center")
        x1 = xpos(_p(params))
        ax.add_patch(Rectangle((X_BAR0, yy - 0.026), max(x1 - X_BAR0, 0.005), 0.052,
                               transform=ax.transAxes, facecolor=col,
                               edgecolor="none", zorder=2))
        ax.text(X_VAL, yy, params, fontsize=13.0, va="center", ha="right",
                color=RED if ours else "black",
                fontweight="bold" if ours else "normal")

    # 边缘预算标注（贴着本文方法那一条）
    y_phr = y0 - (len(rows) - 1) * dy - gap
    ax.text(xpos(3.0) + 0.009, y_phr, "edge budget 3 M", fontsize=10.0, color=RED,
            style="italic", va="center", ha="left")

    # 线性标尺
    ax.plot([X_BAR0, X_BAR1], [0.132, 0.132], transform=ax.transAxes, color="0.55", lw=1.0)
    for t in (0, 30, 60, 90, 120):
        xt = xpos(t)
        ax.plot([xt, xt], [0.124, 0.132], transform=ax.transAxes, color="0.55", lw=1.0)
        ax.text(xt, 0.112, f"{t}", fontsize=10.2, color="0.45", ha="center", va="top")
    ax.text(X_BAR1, 0.142, "parameters (M)", fontsize=10.2, color="0.45",
            ha="right", va="bottom")


PANELS = {"list": draw_list, "family": draw_family, "ladder": draw_ladder}
FOOTERS = {
    "list": "Right: representative heavyweight architectures, their source, parameter cost, "
            "core design and the problem each targets.",
    "family": "Right: representative heavyweight architectures grouped by design family, giving "
              "their source, parameter cost and what each family trades away.",
    "ladder": "Right: parameter cost of representative heavyweight architectures, with bar length "
              "proportional to the parameter count; the red bar is our method at the edge budget.",
}


def main(variant="ladder", out=None):
    img = np.asarray(Image.open(IMG).convert("RGB"))
    x, y, w, h = ROI
    roi = img[y:y + h, x:x + w]

    fig = plt.figure(figsize=(15.6, 4.95), dpi=300)
    gs = fig.add_gridspec(2, 4, width_ratios=[2.30, 1.02, 1.02, 4.86],
                          height_ratios=[1, 1], wspace=0.04, hspace=0.13,
                          left=0.004, right=0.996, top=0.945, bottom=0.055)

    # --- 原图 + 红框（跨两行）---
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(img)
    ax0.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", lw=1.4))
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("(a) Input image", fontsize=11, fontweight="bold", pad=4)

    # --- 放大区域在不同下采样倍率下的结果 ---
    labels = ["(b) x1", "(c) x1/2", "(d) x1/4", "(e) x1/8"]
    cells = [(0, 1), (0, 2), (1, 1), (1, 2)]      # 2x2 排列
    for (lab, factor), cell in zip(zip(labels, [1, 2, 4, 8]), cells):
        ax = fig.add_subplot(gs[cell[0], cell[1]])
        ax.imshow(downsample_cycle(roi, factor), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lab, fontsize=11, fontweight="bold", pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor("red" if factor == 1 else "0.55")
            sp.set_linewidth(1.0)

    # --- 右侧 ---
    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.012, pos.y0, pos.width - 0.012, pos.height])  # 与左侧留出空隙
    PANELS[variant](ax)

    # 左右两部分之间的虚线分隔
    xd = pos.x0 + 0.001
    fig.add_artist(plt.Line2D([xd, xd], [0.048, 0.955], transform=fig.transFigure,
                              color="0.45", lw=0.9, ls=(0, (5, 4)), zorder=5))

    fig.text(0.006, 0.003,
             "Left: the boxed region resampled at 1/2, 1/4 and 1/8 of its resolution and restored "
             "for display; terminal branches a few pixels wide vanish.  " + FOOTERS[variant],
             fontsize=7.4, style="italic", color="0.3")

    out = Path(out) if out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=300, facecolor="white")
        print("saved:", f"{out}.{ext}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(PANELS), default="ladder")
    ap.add_argument("--out", default=None, help="output path without extension")
    a = ap.parse_args()
    main(a.variant, a.out)
