#!/usr/bin/env python3
"""Scan DRIVE + CHASEDB1 + STARE for best PHR-Net capillary advantage ROIs."""
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

import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
SCRIPT_DIR = _p
sys.path.insert(0, SCRIPT_DIR)

import cv2, numpy as np, glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch, torchvision.transforms.functional as TF
from models.unet import UNet
from models.dscnet import DSCNet_pro
from models.phr_net import PHR_Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = os.path.dirname(SCRIPT_DIR)
W_DIR = os.path.join(SCRIPT_DIR, "weights")
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets to scan
DATASETS = [
    ("DRIVE", f"{BASE}/Dataset/DRIVE/test/images/*.tif",
     f"{BASE}/Dataset/DRIVE/test/1st_manual",
     f"{W_DIR}/Comparison/DRIVE",
     "best_unet_drive.pth", "best_dscnet_drive.pth", "best_phr_net_drive.pth",
     lambda f: f.split('_')[0] + "_manual1.gif"),
    ("CHASEDB1", f"{BASE}/Dataset/CHASEDB1/*.jpg",
     f"{BASE}/Dataset/CHASEDB1",
     f"{W_DIR}/Comparison/CHASEDB1",
     "best_unet_chasedb1_fold1.pth", "best_dscnet_chasedb1_fold1.pth",
     "best_phr_net_chasedb1_fold1.pth",
     lambda f: f.replace('.jpg', '_1stHO.png')),
    ("STARE", f"{BASE}/Dataset/STARE/images/*.png",
     f"{BASE}/Dataset/STARE/masks",
     f"{W_DIR}/Comparison/STARE",
     "best_unet_stare_fold1.pth", "best_dscnet_stare_fold1.pth",
     "best_phr_net_stare_fold1.pth",
     lambda f: _resolve_stare_gt_local(f"{BASE}/Dataset/STARE/masks", f.replace(".png", ""))),  # GTFIX
]

def _resolve_stare_gt_local(gt_dir, base_name):
    """STARE 真值兼容解析（GTFIX）：优先 imXXXX_ah.png，其次 imXXXX.png/_vk 等。"""
    base = base_name
    for suf in ("_ah", "_vk", "_av"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    for name_suf in ("_ah", "_vk", "", "_av"):
        for ext in (".png", ".ppm", ".gif", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
            cand = os.path.join(gt_dir, f"{base}{name_suf}{ext}")
            if os.path.exists(cand):
                return cand
    hits = sorted(glob.glob(os.path.join(gt_dir, f"{base}*")))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"STARE 真值未找到：{gt_dir}/{base}[_ah|_vk]<ext>")

def load_image(path):
    bgr = cv2.imread(path)
    if bgr is None: return None
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    H, W = img.shape[:2]
    pw = (32 - W % 32) % 32; ph = (32 - H % 32) % 32
    if pw or ph: img = np.pad(img, ((0,ph),(0,pw),(0,0)), mode='constant')
    return img

def load_gt(path):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gt is None: return None
    H, W = gt.shape
    pw = (32 - W % 32) % 32; ph = (32 - H % 32) % 32
    if pw or ph: gt = np.pad(gt, ((0,ph),(0,pw)), mode='constant')
    return gt > 128

def make_overlay(base, mask, a=0.40):
    o = base.copy(); o[mask,0]=o[mask,0]*(1-a); o[mask,1]=o[mask,1]*(1-a)+a; o[mask,2]=o[mask,2]*(1-a)
    return o

# Scan all datasets
all_candidates = []

for ds_name, img_glob, gt_dir, w_dir, w_unet, w_dsc, w_phr, gt_suffix_fn in DATASETS:
    img_files = sorted(glob.glob(img_glob))[:10]
    if not img_files: continue

    print(f"\n=== {ds_name} ({len(img_files)} images) ===")

    # Load models for this dataset
    unet = UNet(3,1).to(DEVICE)
    unet.load_state_dict(torch.load(f"{w_dir}/{w_unet}", map_location=DEVICE,
                         weights_only=False).get('model_state_dict',{}), strict=False)
    unet.eval()
    dsc = DSCNet_pro(3,1).to(DEVICE)
    dsc.load_state_dict(torch.load(f"{w_dir}/{w_dsc}", map_location=DEVICE,
                        weights_only=False).get('model_state_dict',{}), strict=False)
    dsc.eval()
    phr = PHR_Net(3,1,use_a=True,use_b=True,backbone='hrnet',dsconv_mode='none').to(DEVICE)
    phr.load_state_dict(torch.load(f"{w_dir}/{w_phr}", map_location=DEVICE,
                        weights_only=False).get('model_state_dict',{}), strict=False)
    phr.eval()

    for img_path in img_files:
        fname = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir, gt_suffix_fn(fname))
        if not os.path.exists(gt_path):
            # try alternate naming
            alt = fname.rsplit('.',1)[0] + '_manual1.gif'  # DRIVE alt
            if os.path.exists(os.path.join(gt_dir, alt)):
                gt_path = os.path.join(gt_dir, alt)
            else:
                continue

        img_np = load_image(img_path)
        gt_bin = load_gt(gt_path)
        if img_np is None or gt_bin is None: continue

        img_t = TF.normalize(torch.from_numpy(img_np.copy()).permute(2,0,1).float(),
                            mean=[0.215]*3, std=[0.151]*3).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = (torch.sigmoid(phr(img_t)).squeeze().cpu().numpy()) > 0.5
            u = (torch.sigmoid(unet(img_t)).squeeze().cpu().numpy()) > 0.5
            d = (torch.sigmoid(dsc(img_t)).squeeze().cpu().numpy()) > 0.5

        dist = distance_transform_edt(gt_bin)
        skel = skeletonize((dist > 0) & (dist < 2.5)).astype(np.uint8)
        phr_only = (p & skel) & (~(u & skel)) & (~(d & skel))

        H, W = p.shape; rhw = 80
        density = np.zeros((H, W))
        for y in range(0, H - rhw, rhw//2):
            for x in range(0, W - rhw, rhw//2):
                w_p = phr_only[y:y+rhw, x:x+rhw].sum()
                w_s = skel[y:y+rhw, x:x+rhw].sum()
                if w_s >= 15:
                    density[y+rhw//2, x+rhw//2] = w_p
        from scipy.ndimage import maximum_filter
        lm = maximum_filter(density, size=rhw//2)
        py, px = np.where((density == lm) & (density > 0))
        scores = [(density[py[i], px[i]], py[i], px[i]) for i in range(len(py))]
        scores.sort(key=lambda x: x[0], reverse=True)

        for rank, (s, cy, cx) in enumerate(scores[:2]):
            y1 = max(0, cy - rhw//2); x1 = max(0, cx - rhw//2)
            if y1 + rhw <= H and x1 + rhw <= W:
                all_candidates.append({
                    "ds": ds_name, "img": fname, "rank": rank,
                    "density": s, "y": y1, "x": x1, "h": rhw, "w": rhw,
                    "img_np": img_np, "p": p, "u": u, "d": d, "gt": gt_bin,
                })
        print(f"  {fname}: top densities {[f'{s[0]:.0f}' for s in scores[:3]]}")

# Sort and select top unique candidates
all_candidates.sort(key=lambda c: c["density"], reverse=True)
selected = []
for c in all_candidates:
    # Dedup: don't pick overlapping from same image
    dup = False
    for s in selected:
        if s["img"] == c["img"] and abs(s["y"]-c["y"]) < 60 and abs(s["x"]-c["x"]) < 60:
            dup = True; break
    if dup: continue
    # Max 2 per image
    same_img = sum(1 for s in selected if s["img"] == c["img"])
    if same_img >= 2: continue
    selected.append(c)
    if len(selected) >= 16: break

print(f"\nSelected {len(selected)} candidates from all datasets")

# Plot grid: show only PHR-Net advantage pixels (PHR-Net TP, UNet+DSCNet FN on thin skeleton)
n_cols = 4; n_rows = (len(selected) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5.5 * n_rows))
for ci, c in enumerate(selected):
    ax = axes[ci // n_cols, ci % n_cols]
    y, x, h, w = c["y"], c["x"], c["h"], c["w"]
    roi = c["img_np"][y:y+h, x:x+w].copy()
    # Compute thin-skeleton advantage pixels in this ROI
    roi_gt = c["gt"][y:y+h, x:x+w]
    roi_p = c["p"][y:y+h, x:x+w]
    roi_u = c["u"][y:y+h, x:x+w]
    roi_d = c["d"][y:y+h, x:x+w]
    dist = distance_transform_edt(roi_gt)
    skel = skeletonize((dist > 0) & (dist < 2.5)).astype(bool)
    phr_win = roi_p & skel & (~roi_u) & (~roi_d)
    # Show base image, overlay advantage pixels in bright green
    ax.imshow(roi)
    yy, xx = np.where(phr_win)
    if len(yy) > 0:
        ax.scatter(xx, yy, c='#00ff00', s=8, marker='o', edgecolors='none', alpha=0.9, zorder=10)
    ax.set_title(f'[{ci}] {c["ds"]} {c["img"]} R{c["rank"]}\ny={y} x={x} d={c["density"]:.0f}px',
                 fontsize=7.5)
    ax.axis('off')
for ci in range(len(selected), n_rows * n_cols):
    axes[ci // n_cols, ci % n_cols].axis('off')

plt.suptitle('ROI Candidates — ALL Datasets — PHR-Net correctly detects thin vessels where UNet+DSCNet miss\n'
             'Pick 2 best by visual quality', fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, "ROI_Candidates_AllDatasets.png")
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out}")

print("\n=== Coordinates ===")
for ci, c in enumerate(selected):
    print(f"[{ci}] {c['ds']} {c['img']} y={c['y']} x={c['x']} h={c['h']} w={c['w']} d={c['density']:.0f}")
