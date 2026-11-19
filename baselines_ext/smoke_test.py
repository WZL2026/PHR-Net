#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一框架模型预检：逐个构建所有已注册模型并做一次 256×256 前向，报告参数量与输出形状。

在提交大批 SLURM 作业之前先跑这个（CPU 即可，不占 GPU），能一次性发现
"依赖没装 / 适配写错 / 输出形状不对" 这类问题。

用法：
    python baselines_ext/smoke_test.py                 # 全部模型
    python baselines_ext/smoke_test.py --models CENet FRUNet VMUNet
    python baselines_ext/smoke_test.py --size 256 --device cpu
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="默认跑 MODEL_NAMES 全部")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch

    import train_comparison as tc

    names = args.models or list(tc.MODEL_NAMES)
    # 依赖 mamba_ssm 的模型：其 __init__ 会 import triton，而 triton 在导入时刻就要初始化
    # CUDA 驱动，登录节点（无 GPU）会直接报 `RuntimeError: 0 active drivers`。
    # 这类模型在无 GPU 环境下标记为 SKIP，而不是 FAIL。
    mamba_models = {"MambaVesselNet", "VMUNet", "UMamba", "SwinUMamba", "SerpMamba"}
    has_gpu = torch.cuda.is_available()

    ok, failed, skipped = [], [], []
    print(f"设备={args.device}  GPU可用={has_gpu}  输入=1x3x{args.size}x{args.size}  模型数={len(names)}\n")
    for name in names:
        if name in mamba_models and not has_gpu and args.device == "cpu":
            print(f"[SKIP] {name:<14} 需要 GPU 节点（triton 在 import 时即初始化驱动）"
                  f" → 用 bash scripts/slurm/submit_smoke.sh {name}")
            skipped.append(name)
            continue
        try:
            model = tc.get_model(name).to(args.device).eval()
            total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            with torch.no_grad():
                y = model(torch.randn(1, 3, args.size, args.size, device=args.device))
            if isinstance(y, (list, tuple)):
                y = y[0]
            shape = tuple(y.shape)
            flag = "OK " if shape[-2:] == (args.size, args.size) else "SIZE?"
            print(f"[{flag}] {name:<14} {total/1e6:9.3f} M   out={shape}")
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name:<14} {type(e).__name__}: {str(e)[:160]}")
            failed.append(name)
            if args.device == "cpu":
                traceback.print_exc(limit=2)
    print(f"\n通过 {len(ok)}/{len(names)}  跳过 {len(skipped)}  失败 {len(failed)}")
    if skipped:
        print("跳过（需 GPU 节点）：", ", ".join(skipped))
    if failed:
        print("失败：", ", ".join(failed))
        print("提示：第三方模型失败常见原因——依赖未装（timm/yacs/ml_collections/einops/mamba-ssm/nnunetv2/monai）、"
              "第三方仓库未上传或 EXT_BASELINES_ROOT 未指向 third_party 目录。")
        sys.exit(1)


if __name__ == "__main__":
    main()
