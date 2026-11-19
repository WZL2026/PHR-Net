#!/usr/bin/env python3
"""Export STARE predictions for the Table-6 model set (GPU server).

Seven Table-6 models:
    U-Net, ResUNet, DSCNet, ERFNet, TransUNet, MambaVesselNet, PHR-Net

For every STARE image:
  * sliding-window inference identical to the evaluation pipeline
    (utils.dataset.STAREDataset preprocessing + utils.metrics.sliding_window_inference),
  * probability map cached as float16 to results/preds_stare_t6/<Model>__<base>.npy,
  * per-image Dice/IoU/HD95/clDice -> <Model>_per_image.csv,
  * skeleton-width-bin tp/fp/fn -> width_bins.json  (for Figure 10).

Usage:
    python export_stare_preds.py [--only UNet,ERFNet]
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

_HERE = os.path.dirname(os.path.abspath(__file__))
_root = _HERE
while _root != os.path.dirname(_root) and not (
        os.path.isdir(os.path.join(_root, "models")) and os.path.isdir(os.path.join(_root, "utils"))):
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)
os.chdir(_root)

from utils.dataset import STAREDataset                     # noqa: E402
from utils.metrics import (sliding_window_inference,       # noqa: E402
                           compute_cldice, compute_hd95)
from baselines_ext.wrappers import EXT_INDEX, build as build_ext  # noqa: E402

_cands = [os.path.join(_root, "Dataset", "STARE"),
          os.path.join(os.path.dirname(_root), "Dataset", "STARE"),
          os.path.join(os.path.expanduser("~"), "Project", "Dataset", "STARE")]
DATA = next((p for p in _cands if os.path.isdir(os.path.join(p, "images"))), _cands[0])
W_DIR = os.path.join(_root, "weights", "Comparison", "STARE")
OUT = os.path.join(_root, "results", "preds_stare_t6")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECS = [
    ("UNet", "best_unet_stare_fold1.pth"),
    ("ResUNet", "best_resunet_stare_fold1.pth"),
    ("DSCNet", "best_dscnet_stare_fold1.pth"),
    ("ERFNet", "best_erfnet_stare_fold1.pth"),
    ("TransUNet", "best_transunet_stare_fold1.pth"),
    ("MambaVesselNet", "best_mambavesselnet_stare_fold1.pth"),
    ("PHR_Net", "best_phr_net_stare_fold1.pth"),
]

WIDTH_BINS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), (">10", 11, 999)]


def build_model(name):
    if name in EXT_INDEX:
        return build_ext(EXT_INDEX[name], in_channels=3, num_classes=1)
    if name == "UNet":
        from models.unet import UNet
        return UNet(in_channels=3, out_channels=1)
    if name == "ResUNet":
        from models.resunet import ResUnet
        return ResUnet(in_channels=3, out_channels=1)
    if name == "DSCNet":
        from models.dscnet import DSCNet_pro
        return DSCNet_pro(in_channels=3, out_channels=1)
    if name == "MambaVesselNet":
        from models.mamba_vessel_net import MambaVesselNet
        return MambaVesselNet(in_ch=3, out_ch=1)
    if name == "PHR_Net":
        from models.phr_net import PHR_Net
        return PHR_Net(in_channels=3, out_classes=1)
    raise ValueError(name)


def load_weights_flexible(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if isinstance(sd.get(k), dict):
                sd = sd[k]
                break
    msd = model.state_dict()
    clean = {k.replace("module.", ""): v for k, v in sd.items() if hasattr(v, "shape")}
    best_hit, best = 0, {}
    for pre in ("", "net.", "net.net.", "model.", "swin_unet.", "module."):
        cand = {pre + k: v for k, v in clean.items()}
        hit = {k: v for k, v in cand.items()
               if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
        if len(hit) > best_hit:
            best_hit, best = len(hit), hit
    if best_hit == 0:
        raise RuntimeError(f"{os.path.basename(path)}: no tensor matched the model")
    model.load_state_dict(best, strict=False)
    print(f"  [weights] {os.path.basename(path)}: {best_hit}/{len(msd)} tensors matched", flush=True)
    return best_hit


def skeleton_width_map(gt_mask):
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    if gt_bin.sum() == 0:
        return np.zeros_like(gt_bin, dtype=np.float32), np.zeros_like(gt_bin, dtype=bool)
    skel = skeletonize(gt_bin).astype(np.uint8)
    dist_bg = distance_transform_edt(gt_bin)
    width_map = np.zeros_like(gt_bin, dtype=np.float32)
    width_map[skel > 0] = 2.0 * dist_bg[skel > 0]
    return np.clip(width_map, 0, 50), skel > 0


def binned_tp_fp_fn(pred, gt, width_map, skeleton, fov):
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0.5).astype(np.uint8)
    fov_bin = fov > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = {}
    for label, w_min, w_max in WIDTH_BINS:
        sel = skeleton & (width_map >= w_min) & (width_map <= w_max)
        if sel.sum() == 0:
            out[label] = [0, 0, 0]
            continue
        eval_mask = cv2.dilate(sel.astype(np.uint8), kernel).astype(bool) & fov_bin
        tp = int((pred_bin & gt_bin & eval_mask).sum())
        fp = int((pred_bin & (1 - gt_bin) & eval_mask).sum())
        fn = int(((1 - pred_bin) & gt_bin & eval_mask).sum())
        out[label] = [tp, fp, fn]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]

    os.makedirs(OUT, exist_ok=True)
    ds = STAREDataset(DATA, mode="test")
    names = [os.path.basename(p).replace(".png", "") for p in ds.img_files]
    specs = [s for s in SPECS if not wanted or s[0] in wanted]
    print(f"{len(ds)} STARE images, {len(specs)} models, device={DEVICE}", flush=True)

    bins_path = os.path.join(OUT, "width_bins.json")
    all_bins = {}
    if os.path.exists(bins_path):
        with open(bins_path, encoding="utf-8") as fh:
            all_bins = json.load(fh)

    params = {"UNet": 31.04, "ResUNet": 13.04, "DSCNet": 2.10, "ERFNet": 2.06,
              "TransUNet": 105.32, "MambaVesselNet": 117.22, "PHR_Net": 2.62}

    for name, wf in specs:
        print(f"[{name}] building...", flush=True)
        model = build_model(name).to(DEVICE)
        load_weights_flexible(model, os.path.join(W_DIR, wf))
        model.eval()
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [params] {n_par:.2f} M (table: {params.get(name)} M)", flush=True)

        acc = {label: [0, 0, 0] for label, _, _ in WIDTH_BINS}
        rows = []
        for i, base in enumerate(names, 1):
            img_t, mask_t, fov_t = ds[i - 1]
            gt = mask_t.squeeze().numpy().astype(np.float32)
            fov = fov_t.squeeze().numpy().astype(np.float32)
            npy = os.path.join(OUT, f"{name}__{base}.npy")
            if os.path.exists(npy):
                prob = np.load(npy).astype(np.float32)
            else:
                with torch.no_grad():
                    prob = sliding_window_inference(model, img_t.unsqueeze(0).to(DEVICE),
                                                    window_size=(512, 512), stride=(256, 256),
                                                    device=DEVICE)
                prob = np.squeeze(prob.detach().cpu().numpy() if torch.is_tensor(prob) else prob)
                np.save(npy, prob.astype(np.float16))

            gt_bin = (gt > 0.5).astype(np.float32)
            pr = (prob > 0.5).astype(np.float32)
            tp = float((pr * gt_bin).sum())
            fp = float((pr * (1 - gt_bin)).sum())
            fn = float(((1 - pr) * gt_bin).sum())
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            hd = float(compute_hd95(pr, gt_bin))
            cld = float(compute_cldice(pr, gt_bin))
            rows.append((base, dice, iou, hd, cld))

            width_map, skel = skeleton_width_map(gt_bin)
            bt = binned_tp_fp_fn(prob, gt_bin, width_map, skel, fov)
            for label, (a, b, c) in bt.items():
                acc[label][0] += a
                acc[label][1] += b
                acc[label][2] += c
            print(f"  [{i}/{len(names)}] {base} dice={dice:.4f}", flush=True)

        with open(os.path.join(OUT, f"{name}_per_image.csv"), "w", encoding="utf-8") as fh:
            fh.write("Image,Dice,IoU,HD95,cldice\n")
            for base, dice, iou, hd, cld in rows:
                fh.write(f"{base},{dice:.6f},{iou:.6f},{hd:.4f},{cld:.6f}\n")
        all_bins[name] = acc
        with open(bins_path, "w", encoding="utf-8") as fh:
            json.dump(all_bins, fh, indent=2)
        m = float(np.mean([r[1] for r in rows]))
        print(f"[{name}] mean Dice {m:.4f}  ({n_par:.2f} M params)", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"images": names,
                   "models": [{"name": n, "weights": w, "params_m": params.get(n)} for n, w in specs],
                   "width_bins": [b[0] for b in WIDTH_BINS]}, fh, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
