#!/usr/bin/env python3
"""ROI grid: Original / Ground Truth / PHR-Net side-by-side for visual selection."""
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
import torch, torchvision.transforms.functional as TF
from models.phr_net import PHR_Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W_DIR = os.path.join(SCRIPT_DIR, "weights", "Comparison", "STARE")
DATA_DIR = "D:/Project/Dataset/STARE"
OUT_DIR = os.path.join(SCRIPT_DIR, "results", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATES = [
    ("im0001.png", 200, 440), ("im0001.png", 160, 200),
    ("im0082.png", 160, 360), ("im0004.png", 320, 160),
    ("im0004.png", 240, 400), ("im0077.png", 160,  40),
    ("im0003.png",  40, 240), ("im0002.png", 160, 440),
    ("im0081.png", 120, 400), ("im0139.png", 200, 200),
    ("im0139.png", 160, 480), ("im0044.png",  80, 480),
    ("im0044.png", 240, 320), ("im0005.png", 360, 120),
    ("im0005.png", 200, 200), ("im0002.png", 440, 480),
]

def make_overlay(base, mask):
    o = base.copy()
    a = 0.40
    o[mask,0] = o[mask,0]*(1-a)
    o[mask,1] = o[mask,1]*(1-a)+a
    o[mask,2] = o[mask,2]*(1-a)
    return o

print("Loading PHR-Net...")
phr = PHR_Net(3,1,use_a=True,use_b=True,backbone='hrnet',dsconv_mode='none').to(DEVICE)
sd = torch.load(f"{W_DIR}/best_phr_net_stare_fold1.pth", map_location=DEVICE, weights_only=False)
phr.load_state_dict(sd.get('model_state_dict', sd), strict=False); phr.eval()

img_cache = {}
def get(iname):
    if iname in img_cache: return img_cache[iname]
    bgr = cv2.imread(f"{DATA_DIR}/images/{iname}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    gt = cv2.imread(f"{DATA_DIR}/masks/{iname.replace('.png','_ah.png')}", cv2.IMREAD_GRAYSCALE)
    H,W = img.shape[:2]; pw,ph = (32-W%32)%32, (32-H%32)%32
    if pw or ph: img=np.pad(img,((0,ph),(0,pw),(0,0)),mode='constant'); gt=np.pad(gt,((0,ph),(0,pw)),mode='constant')
    gt_bin = gt>128
    img_t = TF.normalize(torch.from_numpy(img.copy()).permute(2,0,1).float(),
                         mean=[0.215]*3,std=[0.151]*3).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = (torch.sigmoid(phr(img_t)).squeeze().cpu().numpy()) > 0.5
    img_cache[iname] = (img, gt_bin, p)
    return img, gt_bin, p

# Grid: each row = one candidate, 3 columns = Original / GT / PHR-Net
n = len(CANDIDATES)
fig, axes = plt.subplots(n, 3, figsize=(9, 3.2*n))
plt.subplots_adjust(wspace=0.03, hspace=0.08)

for ci, (img_name, y, x) in enumerate(CANDIDATES):
    img, gt, p = get(img_name)
    h = w = 80; roi = img[y:y+h, x:x+w]
    gt_roi = gt[y:y+h, x:x+w]; p_roi = p[y:y+h, x:x+w]

    axes[ci,0].imshow(roi)
    axes[ci,0].set_ylabel(f'[{ci}] {img_name}', fontsize=7, fontweight='bold')
    axes[ci,0].axis('off')
    axes[ci,1].imshow(make_overlay(roi, gt_roi))
    axes[ci,1].axis('off')
    axes[ci,2].imshow(make_overlay(roi, p_roi))
    axes[ci,2].axis('off')

    if ci == 0:
        axes[ci,0].set_title('Original', fontsize=9, fontweight='bold')
        axes[ci,1].set_title('Ground Truth', fontsize=9, fontweight='bold')
        axes[ci,2].set_title('PHR-Net', fontsize=9, fontweight='bold')

plt.suptitle('ROI Candidates — Original / GT / PHR-Net — Pick 2 where PHR-Net best matches GT',
             fontsize=12, fontweight='bold', y=1.005)
plt.tight_layout()
out = os.path.join(OUT_DIR, "ROI_Candidates_AllDatasets.png")
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
