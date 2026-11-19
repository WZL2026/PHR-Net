# Checkpoints

Only the PHR-Net checkpoints are shipped, so that the paper's numbers can be
reproduced without downloading the baselines.

| File | Dataset | Split | Dice |
|---|---|---|---|
| `Comparison/DRIVE/best_phr_net_drive.pth` | DRIVE | official test (20 images) | 0.8254 |
| `Comparison/STARE/best_phr_net_stare_fold1.pth` | STARE | fold 1 | 0.8534 |
| `Comparison/STARE/best_phr_net_stare_fold2.pth` | STARE | fold 2 | -- |
| `Comparison/STARE/best_phr_net_stare_fold3.pth` | STARE | fold 3 | -- |
| `Comparison/STARE/best_phr_net_stare_fold4.pth` | STARE | fold 4 | -- |
| `Comparison/STARE/best_phr_net_stare_fold5.pth` | STARE | fold 5 | -- |

`test_comparison.py` looks for
`weights/Comparison/<DATASET>/best_<model>_<dataset>.pth` (and
`..._stare_fold<k>.pth` for the k-fold datasets), so the files can be used as
is. Baseline checkpoints are not included; train them with
`train_comparison.py` or point the adapter at the official weights.
