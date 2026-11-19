# -*- coding: utf-8 -*-
"""生成论文图 1：代表性失败案例对比 + 重型模型参数量对比表。

用法：
    python make_figure1.py                     # 全 5 折扫描 -> 按准则选图 -> 渲染（结果缓存）
    python make_figure1.py --fold 5 --image 3  # 只渲染指定案例（有缓存则直接读，秒级）
    python make_figure1.py --fold 5 --image 3 --force   # 忽略缓存重算该折

选图准则（写入 meta）：
    recovered_rate = |GT ∩ PHR-Net ∩ (U-Net 漏检 ∪ TransUNet 漏检)| / |GT|
    在 "Dice_ours >= Dice_TransUNet - 0.005" 的候选里取 recovered_rate 最大者；
    若无满足约束的候选，取 recovered_rate 最大者并在 meta 中说明。

布局：左 1×5  (a) 原图 (b) 真值 (c) U-Net (d) TransUNet (e) PHR-Net
      红箭头=基线漏检/断裂的薄血管；绿箭头=PHR-Net 在同一位置正确恢复
      右：若干高精度模型的参数量与 STARE Dice 对比表（PHR-Net 高亮）
输出：results/paper_figures/Figure1_Failure_Case_im0239.{png,pdf,_meta.txt}
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"F:\Project\PHRNet")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import torch

from utils.dataset import STAREDataset
from utils.kfold import create_kfold_dataloaders
from test_comparison import get_model, load_weights_flexible, denormalize_image, DEVICE
from utils.metrics import sliding_window_inference

if os.name == "nt":  # TransUNet 官方 loader 在 Windows 下键名带反斜杠
    sys.path.insert(0, str(ROOT / "third_party" / "TransUNet-main" / "TransUNet-main"))
    try:
        import networks.vit_seg_modeling as _vsm
        import networks.vit_seg_modeling_resnet_skip as _vrs
        _vsm.pjoin = lambda *a: "/".join(str(x) for x in a)
        _vrs.pjoin = lambda *a: "/".join(str(x) for x in a)
    except Exception:  # noqa: BLE001
        pass

OUT_DIR = ROOT / "results" / "paper_figures"
CACHE = OUT_DIR / ".figure1_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

MODELS = ["PHR_Net", "UNet", "TransUNet"]
PARAM_TABLE = [
    ("Swin-Unet", "arXiv'21", "27.17 M", "0.776", "0.634", "12.607", "0.799"),
    ("CE-Net", "TMI'19", "29.00 M", "0.821", "0.696", "6.781", "0.868"),
    ("U-Net", "MICCAI'15", "31.04 M", "0.818", "0.693", "8.769", "0.861"),
    ("VM-UNet", "arXiv'24", "44.27 M", "0.768", "0.623", "13.514", "0.800"),
    ("Swin-UMamba", "MICCAI'24", "59.89 M", "0.838", "0.722", "6.182", "0.881"),
    ("TransUNet", "arXiv'21", "105.32 M", "0.843", "0.729", "5.248", "0.887"),
    ("MambaVesselNet", "JBHI'25", "117.22 M", "0.823", "0.700", "7.978", "0.867"),
    ("PHR-Net (ours)", "This work", "2.62 M", "0.838", "0.721", "6.818", "0.874"),
]

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})


def dice(pred, gt, eps=1e-6):
    tp = float((pred * gt).sum())
    return (2 * tp + eps) / (float(pred.sum() + gt.sum()) + eps)


def cache_path(fold_no, model):
    return CACHE / f"fold{fold_no}_{model}.npz"


def infer_fold(fold_no, model, test_loader, only_idx=None):
    """推断某折（或仅某一张图）并缓存。"""
    cp = cache_path(fold_no, model)
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        out = {int(k): dict(prob=z[f"prob_{k}"], bin=z[f"bin_{k}"], gt=z[f"gt_{k}"],
                            orig=z[f"orig_{k}"], dice=float(z[f"dice_{k}"]))
               for k in z["keys"]}
        print(f"    [cache] fold{fold_no} {model}: {len(out)} images", flush=True)
        if only_idx is None or only_idx in out:
            return out

    net = get_model(model).to(DEVICE)
    load_weights_flexible(net, str(ROOT / f"weights/Comparison/STARE/"
                                        f"best_{model.lower()}_stare_fold{fold_no}.pth"), DEVICE)
    net.eval()
    out = {}
    with torch.no_grad():
        for idx, (images, masks, fovs) in enumerate(test_loader):
            if only_idx is not None and idx != only_idx:
                continue
            prob = sliding_window_inference(net, images.to(DEVICE), window_size=(512, 512),
                                            stride=(256, 256), device=DEVICE)
            fov = fovs.squeeze().cpu().numpy()
            gt = masks.squeeze().cpu().numpy() * fov
            prob = prob * fov
            pb = (prob > 0.5).astype(np.float32)
            out[idx] = dict(prob=prob, bin=pb, gt=gt, orig=denormalize_image(images.squeeze(0)),
                            dice=dice(pb, gt))
            print(f"      img{idx}: Dice={out[idx]['dice']:.4f}", flush=True)
    del net
    if only_idx is None:  # 只有整折推断时才写缓存
        np.savez_compressed(cp, keys=np.array(sorted(out.keys())),
                            **{f"{k}_{i}": v[k] for i, v in out.items()
                               for k in ("prob", "bin", "gt", "orig")},
                            **{f"dice_{i}": np.array(v["dice"]) for i, v in out.items()})
        print(f"    [cache] saved {cp.name}", flush=True)
    return out


def recovered_stats(res, idx):
    gt = res["PHR_Net"][idx]["gt"] > 0.5
    ours = res["PHR_Net"][idx]["bin"] > 0.5
    miss = np.zeros_like(gt)
    for m in ("UNet", "TransUNet"):
        miss = np.maximum(miss, res[m][idx]["bin"] < 0.5)
    rec = gt & ours & miss
    return float(rec.sum()) / max(1.0, float(gt.sum())), rec


def failure_spots(res, idx, topk=3):
    _, rec = recovered_stats(res, idx)
    if rec.sum() == 0:
        return []
    H, W = rec.shape
    ys, xs = np.nonzero(rec)
    keep = (xs > 0.16 * W) & (xs < 0.62 * W) & (ys > 0.14 * H) & (ys < 0.60 * H)
    if keep.sum() == 0:
        keep = (xs > 0.10 * W) & (xs < 0.90 * W) & (ys > 0.10 * H) & (ys < 0.90 * H)
    xs, ys = xs[keep], ys[keep]
    if len(xs) == 0:
        return []
    cell, grid = 26, defaultdict(list)
    for x, y in zip(xs, ys):
        grid[(x // cell, y // cell)].append((x, y))
    centers = []
    for _, pts in sorted(grid.items(), key=lambda kv: -len(kv[1])):
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        if all((cx - ax) ** 2 + (cy - ay) ** 2 > 110 ** 2 for ax, ay in centers):
            centers.append((cx, cy))
        if len(centers) >= topk:
            break
    return centers


def draw_panel(ax, img, mask, title):
    ax.imshow(img)
    if mask is not None:
        ov = np.zeros((*mask.shape, 4), dtype=np.float32)
        ov[mask > 0.5] = [0.0, 1.0, 0.0, 0.5]
        ax.imshow(ov)
    ax.set_title(title, fontsize=9.2, fontweight="bold", pad=3)
    ax.axis("off")


def add_rect(ax, rect, color):
    """在关键区域画矩形框（白描边 + 彩色边框）。rect=(x, y, w, h)。"""
    x, y, w, h = rect
    for lw, c, z in ((3.2, "white", 29), (1.6, color, 30)):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=c, lw=lw,
                                   zorder=z, joinstyle="miter"))


def render_table_only():
    """仅渲染"重型模型参数与精度"表（不依赖推理结果）。"""
    fig = plt.figure(figsize=(7.4, 3.9), dpi=300)
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.78])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Heavyweight models on STARE ($\\geq$25 M): source and accuracy",
                 fontsize=10.5, fontweight="bold", pad=10)
    cols = [(0.00, "Model", "left"), (0.300, "Source", "left"), (0.575, "Params", "right"),
            (0.680, "Dice", "right"), (0.775, "IoU", "right"), (0.870, "HD95", "right"),
            (0.975, "clDice", "right")]
    y0 = 0.93
    for x, txt, ha in cols:
        ax.text(x, y0, txt, fontsize=9.0, fontweight="bold", ha=ha, va="center")
    ax.plot([0.0, 1.0], [y0 - 0.055, y0 - 0.055], color="0.35", lw=0.9)
    dy = 0.115
    for k, row in enumerate(PARAM_TABLE):
        y = y0 - 0.115 - k * dy
        ours = row[0].startswith("PHR-Net")
        kw = dict(fontsize=8.8, va="center",
                  color="#D62728" if ours else "black",
                  fontweight="bold" if ours else "normal")
        for (x, _, ha), val in zip(cols, row):
            ax.text(x, y, val, ha=ha, **kw)
    ax.plot([0.0, 1.0], [y0 - 0.115 - len(PARAM_TABLE) * dy + 0.03] * 2,
            color="0.75", lw=0.7)
    fig.text(0.02, 0.005,
             "STARE, mean over the 5-fold test sets; parameters in millions.",
             fontsize=7.6, style="italic", color="0.3")
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"Figure1_Heavyweight_Table.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
        print("saved:", out, flush=True)
    plt.close(fig)


def render(res, idx, spots, out_meta, roi=None):
    """两行布局：上行 (a)-(e) 整图，下行对应区域放大图，右侧跨两行放参数表。
    放大图内：红点=该基线漏检的真值血管像素，绿点=PHR-Net 在同一位置检出的像素。"""
    r = res["PHR_Net"][idx]
    fig = plt.figure(figsize=(16.2, 4.7), dpi=300)
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 2.75],
                          height_ratios=[1, 1], wspace=0.035, hspace=0.06)

    panels = [("(a) Input", None), ("(b) Ground truth", r["gt"]),
              ("(c) U-Net", res["UNet"][idx]["bin"]),
              ("(d) TransUNet", res["TransUNet"][idx]["bin"]),
              ("(e) PHR-Net (ours)", r["bin"])]
    axes = []
    for k, (title, mask) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        draw_panel(ax, r["orig"], mask, title)
        axes.append(ax)

    # 选区
    if roi is not None:
        rects = [tuple(roi)]
    else:
        rects = [(cx - 36, cy - 36, 72, 72) for cx, cy in spots[:1]] or [(0, 0, 1, 1)]
    gt_b = r["gt"] > 0.5
    # 红点：U-Net 漏检的真值血管像素（(c)(d)(e) 三个面板用同一组点，便于横向对照）
    dots = gt_b & ~(res["UNet"][idx]["bin"] > 0.5)
    edge_colors = ["0.55", "0.55", "red", "red", "green"]
    for ax, ec in zip(axes, edge_colors):
        add_rect(ax, rects[0], ec)

    # 第二行：对应区域放大
    for k, ((title, mask), ec) in enumerate(zip(panels, edge_colors)):
        axz = fig.add_subplot(gs[1, k])
        for rect in rects:
            rx, ry, rw, rh = rect
            ix, iy, iw, ih = (int(round(v)) for v in rect)
            axz.imshow(r["orig"][iy:iy + ih, ix:ix + iw], interpolation="nearest")
            if mask is not None:
                sub = mask[iy:iy + ih, ix:ix + iw]
                ov = np.zeros((ih, iw, 4), dtype=np.float32)
                ov[sub > 0.5] = [0.0, 1.0, 0.0, 0.5]
                axz.imshow(ov, interpolation="nearest")
            if k in (2, 3, 4):
                ys, xs = np.nonzero(dots[iy:iy + ih, ix:ix + iw])
                if len(xs):
                    axz.scatter(xs, ys, s=2.6, c="red", marker="o",
                                linewidths=0, zorder=33)
        axz.set_xticks([])
        axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_edgecolor(ec)
            sp.set_linewidth(0.9)

    axt = fig.add_subplot(gs[:, 5])
    axt.axis("off")
    axt.set_title("Heavyweight models ($\\geq$25 M) on STARE: source and accuracy",
                  fontsize=9.2, fontweight="bold", pad=4)
    cols = [(0.00, "Model", "left"), (0.285, "Source", "left"), (0.575, "Params", "right"),
            (0.670, "Dice", "right"), (0.760, "IoU", "right"), (0.855, "HD95", "right"),
            (0.960, "clDice", "right")]
    y0, dy = 0.95, 0.105
    for x, txt, ha in cols:
        axt.text(x, y0, txt, fontsize=8.2, fontweight="bold", ha=ha, va="center")
    axt.plot([0.0, 1.0], [y0 - 0.05, y0 - 0.05], color="0.4", lw=0.8)
    for k, row in enumerate(PARAM_TABLE):
        y = y0 - 0.105 - k * dy
        ours = row[0].startswith("PHR-Net")
        kw = dict(fontsize=7.5, va="center",
                  color="#D62728" if ours else "black",
                  fontweight="bold" if ours else "normal")
        for (x, _, ha), val in zip(cols, row):
            axt.text(x, y, val, ha=ha, **kw)
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)

    for ext in ("png", "pdf"):
        out = OUT_DIR / f"Figure1_Failure_Case_im0239.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
        print("saved:", out, flush=True)
    plt.close(fig)

    with open(out_meta, "a", encoding="utf-8") as fh:
        fh.write(f"rendered: fold {res['_fold']} image {idx} | roi {rects[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--image", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--image-name", type=str, default=None,
                    help="按文件名选图，如 im0082.png（会在 5 折测试集中查找）")
    ap.add_argument("--table-only", action="store_true",
                    help="只渲染右侧重型模型参数表（不跑推理）")
    ap.add_argument("--roi", type=str, default=None,
                    help="矩形选区 x,y,w,h（原图像素坐标），如 360,160,80,80")
    args = ap.parse_args()
    if args.table_only:
        render_table_only()
        return

    if args.force and args.fold:
        for m in MODELS:
            cp = cache_path(args.fold, m)
            if cp.exists():
                cp.unlink()

    base = ROOT.parent / "Dataset" / "STARE"
    train_ds = STAREDataset(str(base), mode="training")
    val_ds = STAREDataset(str(base), mode="test")
    folds = list(create_kfold_dataloaders(train_ds, val_ds, batch_size=1, num_workers=0))

    meta = OUT_DIR / "Figure1_Failure_Case_im0239_meta.txt"

    if args.image_name:
        target = args.image_name.lower()
        roi = None
        if args.roi:
            roi = tuple(int(float(v)) for v in args.roi.split(","))
        for _, test_loader, fold_idx in folds:
            ds = test_loader.dataset
            names = [Path(f).name.lower() for f in
                     (ds.dataset.img_files if hasattr(ds, "dataset") else ds.img_files)]
            idxs = getattr(ds, "indices", range(len(names)))
            hit = next((k for k, i in enumerate(idxs)
                        if names[i].replace(".png", "") == target.replace(".png", "")), None)
            if hit is None:
                continue
            fold_no = fold_idx + 1
            print(f"[image-name] {args.image_name} -> fold {fold_no}, test index {hit}", flush=True)
            res = {}
            for m in MODELS:
                res[m] = infer_fold(fold_no, m, test_loader, only_idx=hit)
            res["_fold"] = fold_no
            spots = failure_spots(res, hit, topk=1)
            render(res, hit, spots, meta, roi=roi)
            return
        raise SystemExit(f"[image-name] 未在 5 折测试集中找到 {args.image_name}")

    if args.fold and args.image is not None:
        _, test_loader, fold_idx = folds[args.fold - 1]
        res = {}
        for m in MODELS:
            sub = infer_fold(args.fold, m, test_loader, only_idx=args.image)
            res[m] = sub
        res["_fold"] = args.fold
        spots = failure_spots(res, args.image, topk=1)
        roi = None
        if args.roi:
            roi = tuple(int(float(v)) for v in args.roi.split(","))
        render(res, args.image, spots, meta, roi=roi)
        return

    candidates, store = [], {}
    for _, test_loader, fold_idx in folds:
        fold_no = fold_idx + 1
        print(f"\n===== STARE fold {fold_no} =====", flush=True)
        per = {}
        for m in MODELS:
            per[m] = infer_fold(fold_no, m, test_loader)
        store[fold_no] = per
        for idx in per["PHR_Net"]:
            rr, _ = recovered_stats(per, idx)
            d_o, d_u, d_t = (per["PHR_Net"][idx]["dice"], per["UNet"][idx]["dice"],
                             per["TransUNet"][idx]["dice"])
            candidates.append(dict(fold=fold_no, idx=idx, rr=rr, d_ours=d_o, d_u=d_u, d_t=d_t))
            print(f"    img{idx}: ours={d_o:.4f} U-Net={d_u:.4f} TransUNet={d_t:.4f} "
                  f"rr={rr:.4f}", flush=True)

    ok = [c for c in candidates if c["d_ours"] >= c["d_t"] - 0.005]
    best = max(ok if ok else candidates, key=lambda c: (c["rr"], c["d_ours"] - max(c["d_u"], c["d_t"])))
    print(f"\n>>> selected fold {best['fold']} img {best['idx']} | Dice {best['d_ours']:.4f} "
          f"(U {best['d_u']:.4f}, T {best['d_t']:.4f}) | rr {best['rr']:.4f} | "
          f"constraint {bool(ok)}", flush=True)

    meta.write_text(
        "selection rule: max recovered-thin-vessel rate, subject to "
        "Dice_ours >= Dice_TransUNet - 0.005\n"
        f"selected: STARE fold {best['fold']}, test image index {best['idx']}\n"
        f"  Dice: PHR-Net {best['d_ours']:.4f}, U-Net {best['d_u']:.4f}, "
        f"TransUNet {best['d_t']:.4f}\n  recovered rate {best['rr']:.4f}\n\n"
        "all candidates:\n" +
        "".join(f"  fold{c['fold']} img{c['idx']}: ours={c['d_ours']:.4f} U={c['d_u']:.4f} "
                f"T={c['d_t']:.4f} rr={c['rr']:.4f}\n"
                for c in sorted(candidates, key=lambda c: -c["rr"])),
        encoding="utf-8")

    res = store[best["fold"]]
    res["_fold"] = best["fold"]
    render(res, best["idx"], failure_spots(res, best["idx"], topk=3), meta)


if __name__ == "__main__":
    main()
