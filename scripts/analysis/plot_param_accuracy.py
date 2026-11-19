# -*- coding: utf-8 -*-
"""论文用图：参数-精度权衡（参数效率图）。

参照作者给的示例风格：
  * 浅色点状网格（横纵都有）
  * 按模型类型区分标记与配色：CNN=绿色三角、Transformer=黄色方块、SSM/Mamba=蓝色圆、
    本文方法=红色菱形（放大）
  * 每个点旁直接标注"模型名（发表年份）"
  * 底部水平图例 + 图标题
  * 不画 Pareto 前沿、均值参考线、误差棒、阴影带与引线箭头

x = 参数量 (M, 对数轴)；y = DRIVE / CHASEDB1 / STARE 三数据集 Dice 均值。
数据：results/comparison_*_20261017_*.csv（统一协议实测）+ 表 6。
输出：results/paper_figures/Figure_ParamEfficiency.{png,pdf} 与 _data.csv
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = Path(r"F:\Project\PHRNet\results\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (名称, 参数量M, DRIVE, CHASE_DB1, STARE, 年份, 类型)
DATA = [
    ("PHR-Net (Ours)", 2.62, 0.8258, 0.8313, 0.8375, "ours", "ours"),
    ("ERFNet", 2.06, 0.7642, 0.7991, 0.7538, "2018", "cnn"),
    ("DSCNet", 2.10, 0.8137, 0.8166, 0.8024, "2023", "cnn"),
    ("FR-UNet", 7.38, 0.7973, 0.7869, 0.7685, "2022", "cnn"),
    ("DRIU", 7.86, 0.8221, 0.8327, 0.8371, "2016", "cnn"),
    ("U-Mamba", 8.26, 0.8247, 0.8326, 0.8325, "2024", "ssm"),
    ("ResUNet", 13.04, 0.8231, 0.8266, 0.8248, "2018", "cnn"),
    ("Swin-Unet", 27.17, 0.7669, 0.7512, 0.7757, "2021", "transformer"),
    ("CE-Net", 29.00, 0.8069, 0.8260, 0.8210, "2019", "cnn"),
    ("UNet", 31.04, 0.8207, 0.8295, 0.8184, "2015", "cnn"),
    ("VM-UNet", 44.27, 0.7891, 0.7941, 0.7678, "2024", "ssm"),
    ("Swin-UMamba", 59.89, 0.8241, 0.8363, 0.8383, "2024", "ssm"),
    ("TransUNet", 105.32, 0.8208, 0.8361, 0.8432, "2021", "transformer"),
    ("MambaVesselNet", 117.22, 0.8246, 0.8312, 0.8231, "2025", "ssm"),
]

STYLE = {
    "cnn": dict(marker="^", color="#8CC981", edge="#4F7D4A", label="CNN"),
    "transformer": dict(marker="s", color="#F5D76E", edge="#A68F2E", label="Transformer"),
    "ssm": dict(marker="o", color="#79C6E8", edge="#3F7F99", label="SSM / Mamba"),
    "ours": dict(marker="D", color="#E85D5D", edge="#7F1D1D", label="PHR-Net (ours)"),
}

# 标注偏移（point；dx, dy, ha），标签紧贴各自散点
OFFSET = {
    "ERFNet": (7, -8, "left"),
    "DSCNet": (7, 5, "left"),
    "FR-UNet": (7, -8, "left"),
    "DRIU": (0, 9, "center"),
    "U-Mamba": (0, -11, "center"),
    "ResUNet": (7, -9, "left"),
    "Swin-Unet": (0, -11, "center"),
    "CE-Net": (-7, -9, "right"),
    "UNet": (7, 6, "left"),
    "VM-UNet": (7, 4, "left"),
    "Swin-UMamba": (-7, 7, "right"),
    "TransUNet": (0, 9, "center"),
    "MambaVesselNet": (0, -11, "center"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "mathtext.fontset": "stix",
})

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=300)

for name, params, d, c, s, year, fam in DATA:
    y = (d + c + s) / 3
    st = STYLE[fam]
    if fam == "ours":
        ax.scatter([params], [y], s=210, marker=st["marker"], color=st["color"],
                   edgecolor=st["edge"], linewidth=1.1, zorder=5)
        ax.annotate("PHR-Net (ours)\n2.62 M, %.4f" % y, xy=(params, y), xytext=(13, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=8.6,
                    fontweight="bold", color="#B22222", zorder=6)
    else:
        ax.scatter([params], [y], s=72, marker=st["marker"], color=st["color"],
                   edgecolor=st["edge"], linewidth=0.9, zorder=4)
        dx, dy, ha = OFFSET[name]
        ax.annotate("%s (%s)" % (name, year), xy=(params, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha,
                    va="top" if dy < 0 else "bottom", fontsize=7.4, color="#1F1F1F", zorder=6)

ax.set_xscale("log")
ax.set_xlim(1.60, 300)
ax.set_ylim(0.754, 0.8455)
ax.set_xticks([2, 5, 10, 20, 50, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([0.76, 0.78, 0.80, 0.82, 0.84])
ax.set_xlabel("Parameters (M, log scale)", fontsize=10)
ax.set_ylabel("Mean Dice over DRIVE / CHASEDB1 / STARE", fontsize=10)
ax.tick_params(direction="out", length=3, width=0.8, labelsize=8.5)
ax.grid(True, which="major", linestyle=":", linewidth=0.8, color="0.82")
ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)

ax.set_title("Parameter efficiency: segmentation accuracy vs. parameters",
             fontsize=11.5, fontweight="bold", pad=10)

handles = [Line2D([], [], linestyle="", marker=STYLE[k]["marker"], markersize=8,
                  markerfacecolor=STYLE[k]["color"], markeredgecolor=STYLE[k]["edge"],
                  markeredgewidth=0.9, label=STYLE[k]["label"])
           for k in ("transformer", "cnn", "ssm", "ours")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=4, frameon=False, fontsize=9, handletextpad=0.35, columnspacing=1.4)

fig.tight_layout()
for ext in ("png", "pdf"):
    out = OUT_DIR / f"Figure_ParamEfficiency.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
    print("saved:", out)
plt.close(fig)

csv_path = OUT_DIR / "Figure_ParamEfficiency_data.csv"
with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Model", "Family", "Year", "Params_M", "DRIVE", "CHASE_DB1", "STARE", "Mean"])
    for name, params, d, c, s, year, fam in DATA:
        w.writerow([name, fam, year, params, d, c, s, round((d + c + s) / 3, 4)])
print("saved:", csv_path)
