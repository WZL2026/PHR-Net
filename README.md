# PHR-Net: Prior-Guided Parameter-Efficient High-Resolution Network for Retinal Vessel Segmentation

Official PyTorch implementation of **PHR-Net**, a 2.62 M-parameter network for
retinal vessel segmentation that keeps a single full-resolution detail stream
and replaces deep downsampling with two lightweight modules: a **prior-guided
feature enhancement** module (PGFE, zero learnable parameters) and an
**attention-guided fusion** module (AGF, +0.05 M parameters), joined by the
**dual-stream backbone PGCA**.

> Manuscript under review at *Computerized Medical Imaging and Graphics*.

- 2.62 M parameters — 17–45x fewer than the compared backbones
- best capillary-bin (1–2 px) F1 on both DRIVE and STARE
- best boundary distance (HD95 3.53 on DRIVE) among all compared models
- 2^3 full-factorial ablation isolating backbone / prior / attention effects

## Repository layout

```
.
├── train_comparison.py          # training entry point (comparison protocol)
├── test_comparison.py           # evaluation entry point (Dice/IoU/HD95/clDice)
├── models/                      # PHR-Net + U-Net, ResUNet, DSCNet, MambaVesselNet, NETracer
├── utils/                       # datasets, metrics (HD95, clDice), k-fold, logger
├── baselines_ext/               # adapters that plug in third-party baselines
├── scripts/analysis/            # 2^3 ablation, paired bootstrap / t-tests
├── scripts/visualization/       # every figure of the paper
├── weights/Comparison/          # PHR-Net checkpoints (DRIVE + STARE folds 1-5)
├── results/                     # summary + per-image CSVs behind Table 6
├── data/README.md               # how to obtain DRIVE / CHASEDB1 / STARE
└── requirements.txt
```

## Results (Dice, mean over the official split or 5-fold CV)

| Model | Params (M) | DRIVE | CHASEDB1 | STARE |
|---|---|---|---|---|
| U-Net | 31.04 | 0.821 | 0.818 | 0.818 |
| ResUNet | 13.04 | 0.823 | 0.827 | 0.825 |
| DSCNet | 2.10 | 0.814 | 0.817 | 0.802 |
| ERFNet | 2.06 | 0.764 | 0.799 | 0.754 |
| TransUNet | 105.32 | 0.821 | 0.836 | 0.843 |
| MambaVesselNet | 117.22 | 0.825 | 0.831 | 0.823 |
| **PHR-Net (ours)** | **2.62** | **0.826** | **0.831** | **0.838** |

Full metrics (IoU, HD95, clDice) and the per-image values are in
`results/summary/summary_all_20261017_202311.csv` and `results/per_image/`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11 and PyTorch 2.11 (CUDA 12.8); a CPU-only install works
for a single image but is slow.

Optional, only for the Mamba baselines: `pip install mamba_ssm causal_conv1d`.
Without them the code falls back to a pure-PyTorch selective scan
(`baselines_ext/mamba_fallback.py`), which is slower but numerically equivalent.

## Data

The datasets are **not** redistributed. Download them from the official
sources and put them in a folder next to this repository — the code resolves
`../Dataset/<name>`:

```
<parent>/
├── PHR-Net/     <- this repository
└── Dataset/
    ├── DRIVE/{training,test}/{images,1st_manual,mask}
    ├── CHASEDB1/*.jpg
    └── STARE/{images,masks,topo_gt}
```

See `data/README.md` for the download links and the exact folder layout.

## Quick start

```bash
# evaluate the shipped DRIVE checkpoint (writes results/per_image/*.csv)
python test_comparison.py --model PHR_Net --dataset DRIVE

# STARE is evaluated with 5-fold cross-validation
python test_comparison.py --model PHR_Net --dataset STARE

# train from scratch (300 epochs, batch 8, AdamW, lr 2e-4)
python train_comparison.py --model PHR_Net --dataset DRIVE

# any compared baseline uses exactly the same protocol
python train_comparison.py --model ResUNet --dataset CHASEDB1
```

The evaluation writes `results/comparison_<DATASET>_<MODEL>_<timestamp>.csv`
and, with `--save_per_image`, one row per test image in `results/per_image/`.

## Ablation and statistics

```bash
python scripts/analysis/train_phrnet_ablation.py   # 8 configurations of the 2^3 design
python scripts/analysis/test_phrnet_ablation.py
python scripts/analysis/ttest_analysis.py --per_image_dir results/per_image
```

## Figures

`scripts/visualization/` regenerates every figure of the paper, including the
qualitative panels and the per-vessel-width analysis:

```bash
python scripts/visualization/export_drive_preds.py    # cache the probability maps
python scripts/visualization/make_figures_drive.py    # Figs. 7-10
```

## Third-party baselines

`baselines_ext/` contains only adapters (weight-name mapping, patch scripts,
smoke tests). The official implementations of ERFNet, TransUNet, CE-Net,
FR-UNet, Swin-Unet, VM-UNet, Swin-UMamba, U-Mamba and M2U-Net are **not**
redistributed; clone them from their own repositories into `third_party/` and
run `python baselines_ext/apply_ext_patch.py`. Each project keeps its original
license.

## Citation

```bibtex
@article{wang2026phrnet,
  title   = {PHR-Net: Prior-Guided Parameter-Efficient High-Resolution Network
             for Retinal Vessel Segmentation},
  author  = {Wang, Zelong and Li, Lei and Zhou, Jian and Du, Xiuquan},
  journal = {Computerized Medical Imaging and Graphics},
  year    = {2026},
  note    = {under review}
}
```

## License

MIT for the code in this repository. The datasets and the third-party model
implementations are governed by their own licenses.

## Repository

https://github.com/Maxlong18/PHR-Net

## Contact

Zelong Wang — Anhui University, School of Computer Science and Technology.
Corresponding author: Xiuquan Du (dxqllp@163.com).
