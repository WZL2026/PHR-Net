#!/usr/bin/env python3
"""Generate a grid of candidate ROIs across multiple STARE images so user can pick the best."""
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

import cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch, torchvision.transforms.functional as TF
from models.unet import UNet
from models.dscnet import DSCNet_pro
from models.phr_net import PHR_Net

DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Dataset", "STARE")
W_DIR = os.path.join(SCRIPT_DIR, "weights", "Comparison", "STARE")
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(name, cls, wf):
    if "PHR" in name:
        m = cls(3, 1, use_a=True, use_b=True, backbone="hrnet", dsconv_mode="none").to(DEVICE)
    elif name == "DSCNet":
        m = cls(3, 1).to(DEVICE)
    else:
        m = cls(3, 1).to(DEVICE)
    ckpt = torch.load(os.path.join(W_DIR, wf), map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False); m.eval()
    return m

def load(img_path, gt_path):
    bgr = cv2.imread(img_path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape[:2]
    pw = (32 - W % 32) % 32; ph = (32 - H % 32) % 32
    if pw or ph:
        img = np.pad(img, ((0,ph),(0,pw),(0,0)), mode='constant')
        gt = np.pad(gt, ((0,ph),(0,pw)), mode='constant')
    img_t = TF.normalize(torch.from_numpy(img.copy()).permute(2,0,1).float(),
                         mean=[0.215]*3, std=[0.151]*3)
    return img_t, gt > 128, img

def make_overlay(base, mask, a=0.40):
    o = base.copy(); o[mask,0]=o[mask,0]*(1-a); o[mask,1]=o[mask,1]*(1-a)+a; o[mask,2]=o[mask,2]*(1-a)
    return o

print("Loading models...")
unet = load_model("UNet", UNet, "best_unet_stare_fold1.pth")
dsc = load_model("DSCNet", DSCNet_pro, "best_dscnet_stare_fold1.pth")
phr = load_model("PHR-Net", PHR_Net, "best_phr_net_stare_fold1.pth")

img_files = sorted([f for f in os.listdir(f"{DATA_DIR}/images") if f.endswith('.png')])[:8]

candidates = []
for img_name in img_files:
    base = img_name.replace('.png','')
    img_t, gt_bin, img_np = load(f"{DATA_DIR}/images/{img_name}",
                                  f"{DATA_DIR}/masks/{base}_ah.png")
    img_b = img_t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = (torch.sigmoid(phr(img_b)).squeeze().cpu().numpy()) > 0.5
        u = (torch.sigmoid(unet(img_b)).squeeze().cpu().numpy()) > 0.5
        d = (torch.sigmoid(dsc(img_b)).squeeze().cpu().numpy()) > 0.5

    dist = distance_transform_edt(gt_bin)
    skel = skeletonize((dist > 0) & (dist < 2.5)).astype(np.uint8)
    phr_only = (p & skel) & (~(u & skel)) & (~(d & skel))

    H, W = p.shape; rhw = 80
    density = np.zeros((H, W))
    for y in range(0, H - rhw, rhw//2):
        for x in range(0, W - rhw, rhw//2):
            w_p = phr_only[y:y+rhw, x:x+rhw].sum()
            w_s = skel[y:y+rhw, x:x+rhw].sum()
            if w_s >= 20:
                density[y+rhw//2, x+rhw//2] = w_p
    from scipy.ndimage import maximum_filter
    lm = maximum_filter(density, size=rhw//2)
    py, px = np.where((density == lm) & (density > 0))
    scores = [(density[py[i],px[i]], py[i], px[i]) for i in range(len(py))]
    scores.sort(key=lambda x: x[0], reverse=True)

    for rank, (s, cy, cx) in enumerate(scores[:3]):
        y1 = max(0, cy - rhw//2); x1 = max(0, cx - rhw//2)
        candidates.append({
            "img_name": img_name, "img_np": img_np, "rank": rank,
            "density": s, "y": y1, "x": x1, "h": rhw, "w": rhw,
            "preds": {"PHR-Net": p, "UNet": u, "DSCNet": d},
            "gt_bin": gt_bin,
        })

# Keep top 12 candidates by density, ensuring image diversity
candidates.sort(key=lambda c: c["density"], reverse=True)
selected = []
used_imgs = set()
for c in candidates:
    img_key = c["img_name"]
    if img_key in used_imgs and len([s for s in selected if s["img_name"]==img_key]) >= 2:
        continue
    selected.append(c)
    used_imgs.add(img_key)
    if len(selected) >= 12: break

print(f"Selected {len(selected)} candidates from {len(used_imgs)} images")

# Plot 12 candidates in 3 rows × 4 cols
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for ci, c in enumerate(selected):
    ax = axes[ci // 4, ci % 4]
    y, x, h, w = c["y"], c["x"], c["h"], c["w"]
    roi = c["img_np"][y:y+h, x:x+w]
    gt_roi = c["gt_bin"][y:y+h, x:x+w]
    p_roi = c["preds"]["PHR-Net"][y:y+h, x:x+w]
    u_roi = c["preds"]["UNet"][y:y+h, x:x+w]

    # Show PHR-Net overlay
    disp = make_overlay(roi, p_roi)
    ax.imshow(disp)
    ax.set_title(f'{c["img_name"]} R{c["rank"]}\nd={c["density"]:.0f}px phr_only',
                 fontsize=8)
    ax.axis('off')

plt.suptitle('ROI Candidates — PHR-Net (green) correctly detects thin vessels where UNet+DSCNet miss\n'
             'Pick 2 best ROIs by visual quality', fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, "ROI_Candidates_Grid.png")
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")

# Also print coordinates for easy reference
print("\nCoordinates for selected candidates:")
for ci, c in enumerate(selected):
    print(f"  [{ci}] {c['img_name']} y={c['y']} x={c['x']} h={c['h']} w={c['w']} density={c['density']:.0f}")
