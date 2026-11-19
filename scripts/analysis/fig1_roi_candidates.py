# -*- coding: utf-8 -*-
"""生成图1 的候选选区对比图：同一张 STARE 图上给出 K 个候选区域，供作者挑选。

左：整图（真值绿叠加）+ 编号候选框
右：每个候选一行，三列 = 真值 / U-Net（红点=漏检） / PHR-Net（绿点=检出）

用法：python scripts/analysis/fig1_roi_candidates.py --fold 5 --image 3 --k 6
输出：results/paper_figures/_fig1_roi_candidates.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\Project\PHRNet")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figure1 as M
from utils.dataset import STAREDataset
from utils.kfold import create_kfold_dataloaders

OUT = ROOT / "results" / "paper_figures" / "_fig1_roi_candidates.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=5)
    ap.add_argument("--image", type=int, default=3)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--side", type=int, default=72)
    ap.add_argument("--criterion", type=str, default="unet_vs_others",
                    choices=["unet_vs_others", "rec"],
                    help="unet_vs_others: U-Net 漏检而 TransUNet 与 PHR-Net 都检出的像素；rec: 两个基线同时漏检而 PHR-Net 检出")
    args = ap.parse_args()

    base = ROOT.parent / "Dataset" / "STARE"
    train_ds = STAREDataset(str(base), mode="training")
    val_ds = STAREDataset(str(base), mode="test")
    folds = list(create_kfold_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0))
    _, test_loader, fold_idx = folds[args.fold - 1]

    res = {}
    for m in M.MODELS:
        res[m] = M.infer_fold(args.fold, m, test_loader, only_idx=args.image)
    idx = args.image
    r = res["PHR_Net"][idx]
    gt = r["gt"] > 0.5
    ours = res["PHR_Net"][idx]["bin"] > 0.5
    unet = res["UNet"][idx]["bin"] > 0.5
    trans = res["TransUNet"][idx]["bin"] > 0.5
    miss_u = gt & ~unet
    miss_t = gt & ~trans
    rec = miss_u & miss_t & ours           # 旧判据：两基线都漏、我们检出
    focus = rec if args.criterion == "rec" else (gt & ~unet & trans & ours)  # 新判据：U-Net 漏、TransUNet+我们检出

    H, W = gt.shape
    side = args.side
    cands = []
    for y in range(0, H - side, 8):
        for x in range(0, W - side, 8):
            rec_n = int(focus[y:y + side, x:x + side].sum())
            if rec_n < 8:
                continue
            gt_n = int(gt[y:y + side, x:x + side].sum())
            miss_n = int(miss_u[y:y + side, x:x + side].sum())
            score = rec_n + 0.05 * min(gt_n, 1200)
            cands.append((score, rec_n, miss_n, gt_n, x, y))
    cands.sort(reverse=True)

    picked = []
    for c in cands:
        _, _, _, _, x, y = c
        if all((x - px) ** 2 + (y - py) ** 2 > 90 ** 2 for _, _, _, _, px, py in picked):
            picked.append(c)
        if len(picked) >= args.k:
            break
    print(f"candidates: {[(p[4], p[5], p[1], p[2]) for p in picked]}", flush=True)

    n = len(picked)
    fig = plt.figure(figsize=(9.6, 3.2 + 0.95 * n), dpi=300)
    gs = fig.add_gridspec(n, 5, width_ratios=[2.6, 1, 1, 1, 0.04],
                          hspace=0.28, wspace=0.05)

    # 左：整图 + 编号框
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(r["orig"])
    ov = np.zeros((*gt.shape, 4), dtype=np.float32)
    ov[gt] = [0.0, 1.0, 0.0, 0.45]
    ax0.imshow(ov)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(f"STARE fold {args.fold} img{idx} — candidate ROIs ({args.criterion})", fontsize=9, fontweight="bold")
    for i, (_, rec_n, miss_n, gt_n, x, y) in enumerate(picked, 1):
        ax0.add_patch(plt.Rectangle((x, y), side, side, fill=False, edgecolor="red", lw=1.2))
        ax0.text(x + 2, y - 4, f"#{i}", fontsize=9, color="red", fontweight="bold")

    titles = ["Ground truth", "U-Net (red = missed)", "PHR-Net (lime = U-Net miss, ours detects)"]
    for row, (_, rec_n, miss_n, gt_n, x, y) in enumerate(picked):
        masks = [gt, unet, ours]
        for col, (title, mk) in enumerate(zip(titles, masks)):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.imshow(r["orig"][y:y + side, x:x + side], interpolation="nearest")
            ov = np.zeros((side, side, 4), dtype=np.float32)
            sub = mk[y:y + side, x:x + side]
            ov[sub > 0.5] = [0.0, 1.0, 0.0, 0.5]
            ax.imshow(ov, interpolation="nearest")
            if col == 1:
                ys, xs = np.nonzero(miss_u[y:y + side, x:x + side])
                ax.scatter(xs, ys, s=2.2, c="red", linewidths=0, marker="o")
            if col == 2:
                ys, xs = np.nonzero(focus[y:y + side, x:x + side])
                ax.scatter(xs, ys, s=2.2, c="lime", linewidths=0, marker="o")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
            if col == 0:
                ax.set_ylabel(f"#{row + 1}\n({x},{y})\nrec={rec_n}\nmiss={miss_n}",
                              fontsize=6.5, rotation=0, ha="right", va="center", labelpad=14)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print("saved:", OUT, flush=True)


if __name__ == "__main__":
    main()
