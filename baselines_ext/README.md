# baselines_ext —— 第三方对比模型统一适配层

把从各论文官方仓库下载的对比模型，接成**同一套接口**，从而能直接挂到本项目的
统一训练/测试框架（`train_comparison.py` / `test_comparison.py`）里，用同一协议训练与评估。

统一接口约定：**输入 RGB 3 通道 → 输出单通道 logits、与输入同分辨率**。

## 已适配模型

| 展示名（`--model`） | 论文 | 依赖 | 备注 |
|---|---|---|---|
| `CENet` | CE-Net, TMI 2019 | 无 | 官方 PyTorch 实现（`Guzaiwang/CE-Net`） |
| `FRUNet` | FR-UNet, JBHI 2022 | 无 | 全程全分辨率，专治细血管断裂 |
| `M2UNet` | M2U-Net, CVPRW 2019 | 无 | ≈0.55 M，唯一有 ARM 端侧实测；官方权重可用 |
| `ERFNet` | ERFNet | 无 | M2U-Net 论文的端侧对照；权重同仓库 |
| `DRIU` | DRIU, 2016 | 无 | 早期血管分割深度基线 |
| `VMUNet` | VM-UNet, TOMM 2024 | `mamba-ssm` | 官方 forward 输出概率，这里改用 2 类头取 logits |
| `SwinUnet` | Swin-Unet, ECCVW 2022 | `yacs` | 需要仓库自带 yaml 配置；窗口 7，输入建议 224 的倍数 |
| `TransUNet` | TransUNet, 2021 | `ml_collections` | **需 R50-ViT-B/16 的 ImageNet 权重**，否则指标偏低 |
| `UMamba` | U-Mamba, 2024 | `nnunetv2`, `dynamic-network-architectures`, `mamba-ssm` | nnU-Net 风格 |
| `SwinUMamba` | Swin-UMamba, MICCAI 2024 | `nnunetv2`, `monai`, `mamba-ssm` | 同上 |
| `SerpMamba` | Serp-Mamba, TMI 2025 | `nnunetv2`, `dynamic-network-architectures`, `mamba-ssm`, `einops` | **最直接对标**：高分辨率视网膜血管 + 蛇形 SSM |

未适配（官方无 PyTorch 实现，论文里作文献引用即可）：**SA-UNet**（Keras 2.3.1）、**IterNet**（Keras/TF）。
**Lite-HRNet** 官方仓库是姿态估计版（主干 + mmcv/mmpose 依赖），做分割需自配解码器，暂不纳入。

## 目录要求

第三方仓库放在项目根目录的 `third_party/` 下（旧目录名 `对比模型源码/` 同样兼容），
或设置环境变量 `EXT_BASELINES_ROOT` 指向它们所在目录：

```
PHRNet/
├── train_comparison.py
├── test_comparison.py
├── baselines_ext/           # 本适配层
└── third_party/
    ├── CE-Net-master/CE-Net-master/
    ├── FR-UNet-master/FR-UNet-master/
    ├── M2U-Net-master/M2U-Net-master/
    ├── VM-UNet-main/VM-UNet-main/
    ├── Swin-Unet-main/Swin-Unet-main/
    ├── TransUNet-main/TransUNet-main/
    ├── U-Mamba-main/U-Mamba-main/
    ├── Swin-UMamba-main/Swin-UMamba-main/
    └── Serp-Mamba-main/Serp-Mamba-main/
```

## 使用步骤（新服务器）

```bash
PROJECT_DIR=/Share/home/E125211133/Project/PHRNet
PY=/Share/home/E125211133/anaconda3/envs/wzl/bin/python
cd $PROJECT_DIR

# 1) 安装依赖（按需；已有的跳过）
$PY -m pip install timm yacs ml_collections einops              # Swin-Unet / TransUNet / 通用
$PY -m pip install mamba-ssm causal-conv1d                      # VM-UNet
$PY -m pip install nnunetv2 dynamic-network-architectures monai batchgenerators   # U-Mamba / Swin-UMamba / Serp-Mamba

# 2) 注册进统一框架（幂等，自动备份为 *.bak_ext）
$PY baselines_ext/apply_ext_patch.py --project $PROJECT_DIR

# 3) 预检（CPU，不占 GPU）：逐个模型构建 + 参数量 + 前向
bash scripts/slurm/smoke_test.sh

# 4) 一键提交：每个模型一个 SLURM 作业，作业内跑 DRIVE→CHASEDB1→STARE 的训练+测试
bash scripts/slurm/submit_all_models.sh
```

只想跑其中几个模型：

```bash
bash scripts/slurm/submit_all_models.sh CENet FRUNet VMUNet SwinUnet
```

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `找不到第三方仓库目录` | 仓库没上传，或目录名/层级不对；用 `EXT_BASELINES_ROOT` 指定实际路径 |
| `ModuleNotFoundError: mamba_ssm` | 装 `mamba-ssm`（配 `causal-conv1d`）；VM-UNet 内部有 try/except，缺了会静默降级，但 U-Mamba/Serp-Mamba 会直接报错 |
| `ModuleNotFoundError: nnunetv2` / `monai` | 装 nnU-Net 生态三件套（见步骤 1） |
| `ModuleNotFoundError: yacs` / `ml_collections` | Swin-Unet / TransUNet 的配置依赖 |
| 训练能跑但指标明显偏低（TransUNet） | 没加载 ViT 预训练权重；`build_transunet(pretrained_vit=...)` 传入权重路径 |
| 输出形状不是原图大小 | 适配层会自动插值到输入分辨率；若日志里 `SIZE?`，检查该模型的 strided 结构 |
| 某个模型作业失败 | 不影响其它模型（一模型一作业）；看 `logs/cmp_<模型>_<作业号>.err` |

## 一致性提醒（写论文时要用）

- 所有第三方模型都在**同一预处理、同一损失、同一训练轮数与优化器**下训练，不做单独调参；
- 第三方模型一律改成 **3 通道 RGB 输入**，论文里需注明（部分原仓库用单通道灰度）；
- 确实无开源代码的模型（HRD-Net、LHR-Net）属于"按论文复现"，应与本表所列"官方代码"分开表述。
