# Data

The three public datasets are **not** redistributed here. Download them from
the official sources and place them next to the repository:

```
<parent>/
├── PHR-Net/            <- this repository
└── Dataset/
    ├── DRIVE/
    │   ├── training/{images,1st_manual,mask}
    │   └── test/{images,1st_manual,mask}
    ├── CHASEDB1/       <- *.jpg (images and manual annotations)
    └── STARE/
        ├── images/     <- im0001.png ...
        ├── masks/      <- im0001_ah.png ... (expert 1 annotation)
        └── topo_gt/    <- optional, used by the topology analysis
```

`test_comparison.py` resolves the data root as `../Dataset/<name>`, i.e. a
sibling folder of this repository, which matches the layout above.

| Dataset | Source |
|---|---|
| DRIVE | https://drive.grand-challenge.org/ (registration required) |
| CHASEDB1 | https://blogs.kingston.ac.uk/retinal/chasedb1/ |
| STARE | https://www.ces.clemson.edu/~ahoover/stare/ |

Notes
* DRIVE: official 20/20 train-test split; evaluation restricted to the
  field-of-view mask.
* CHASEDB1: 28 images, 4-fold cross-validation in the paper.
* STARE: 20 images, 5-fold cross-validation; the ground truth is the
  `_ah` (expert 1) annotation. Some copies of the STARE archive contain
  inconsistent annotations; `baselines_ext/apply_stare_gt_fix.py` documents
  the fix used here.
