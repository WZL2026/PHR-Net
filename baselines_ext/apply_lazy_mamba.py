#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 MambaVesselNet 的导入改成惰性导入（幂等，自动备份 .bak_lazy）。

原因：`train_comparison.py` / `test_comparison.py` 原本在模块顶层就
`from models.mamba_vessel_net import MambaVesselNet`，而 mamba_ssm 在 import 时会加载 triton
的 layer_norm，triton 的 @autotune 在"导入时刻"就要初始化 CUDA 驱动 —— 于是在**登录节点**
（无 GPU）连 `import train_comparison` 都会直接抛 `RuntimeError: 0 active drivers`，
CPU 预检、参数统计等纯 CPU 操作全部无法进行。

改法：顶层 import 去掉，改为在 get_model() 命中 MambaVesselNet 分支时才导入 —— 行为不变，
只是把"需要 GPU 的时刻"推迟到真正要用这个模型的时候。

用法：
    python baselines_ext/apply_lazy_mamba.py --project .
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMPORT_OLD = "from models.mamba_vessel_net import MambaVesselNet"
BRANCH_OLD = "    if model_name == 'MambaVesselNet':  return MambaVesselNet(in_ch=3, out_ch=1)"
BRANCH_NEW = (
    "    if model_name == 'MambaVesselNet':\n"
    "        # 惰性导入：mamba_ssm 在 import 时会初始化 triton 驱动，登录节点(无 GPU)直接 import 会失败\n"
    "        from models.mamba_vessel_net import MambaVesselNet\n"
    "        return MambaVesselNet(in_ch=3, out_ch=1)"
)
MARK = "MODEL_IMPORT_LAZY"


def patch(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARK in text:
        print(f"[skip] {path.name} 已改过")
        return False
    if IMPORT_OLD not in text or BRANCH_OLD not in text:
        print(f"[warn] {path.name} 未找到预期代码，跳过（顶层 import 或 get_model 分支已被改动？）")
        return False

    bak = path.with_suffix(path.suffix + ".bak_lazy")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")

    text = text.replace(IMPORT_OLD, f"# {IMPORT_OLD}  # {MARK}：改为 get_model() 内惰性导入")
    text = text.replace(BRANCH_OLD, BRANCH_NEW)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"[patch] {path.name}：MambaVesselNet 已改为惰性导入")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    n = 0
    for name in ("train_comparison.py", "test_comparison.py"):
        p = root / name
        if p.exists():
            n += int(patch(p))
        else:
            print(f"[warn] 未找到 {p}")
    print(f"\n完成，修改 {n} 个文件。回滚：用同目录 *.bak_lazy 覆盖回去。")


if __name__ == "__main__":
    main()
