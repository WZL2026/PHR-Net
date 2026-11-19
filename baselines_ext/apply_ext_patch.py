#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 baselines_ext 的第三方对比模型注册进统一训练/测试框架（幂等，自动备份）。

作用（对 train_comparison.py 与 test_comparison.py 各做两处修改）：
  1. 在 MODEL_NAMES 里追加第三方模型名，使 `--model CENet` 之类的参数合法；
  2. 在 get_model() 开头插入 baselines_ext 分发，命中即返回适配后的模型。

用法（新服务器项目根目录下）：
    python baselines_ext/apply_ext_patch.py            # 默认 --project .
    python baselines_ext/apply_ext_patch.py --project /Share/home/E125211133/Project/PHRNet
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

EXT_NAMES = ["CENet", "FRUNet", "M2UNet", "ERFNet", "DRIU", "VMUNet",
             "SwinUnet", "TransUNet", "UMamba", "SwinUMamba", "SerpMamba"]

MARKER = "# === EXT BASELINES (baselines_ext) ==="
CONST_LINE = ("EXT_MODEL_NAMES = " + repr(EXT_NAMES) +
              "  # 第三方官方仓库模型，统一适配见 baselines_ext/wrappers.py\n")
HOOK = (
    "    " + MARKER + "\n"
    "    if model_name in EXT_MODEL_NAMES:\n"
    "        from baselines_ext.wrappers import EXT_INDEX, build as _build_ext\n"
    "        if model_name not in EXT_INDEX:\n"
    "            raise KeyError(f\"{model_name} 未在 baselines_ext/wrappers.py 的 BUILDERS 中注册\")\n"
    "        return _build_ext(EXT_INDEX[model_name], in_channels=3, num_classes=1)\n"
)

TARGETS = ["train_comparison.py", "test_comparison.py"]


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[skip] {path.name} 已打过补丁")
        return False

    bak = path.with_suffix(path.suffix + ".bak_ext")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[backup] {path.name} -> {bak.name}")

    # 1) MODEL_NAMES 追加第三方模型名
    m = re.search(r"^MODEL_NAMES = \[(.*?)\]", text, re.M)
    if not m:
        raise SystemExit(f"[FAIL] 在 {path.name} 里找不到 MODEL_NAMES")
    names = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
    added = [n for n in EXT_NAMES if n not in names]
    new_names = names + added
    new_line = "MODEL_NAMES = [" + ", ".join(f"'{n}'" for n in new_names) + "]"
    text = text[:m.start()] + new_line + "\n" + CONST_LINE + text[m.end():]
    print(f"[patch] {path.name} MODEL_NAMES += {added}")

    # 2) get_model() 开头插入分发
    m2 = re.search(r"^def get_model\(.*?\):\n", text, re.M)
    if not m2:
        raise SystemExit(f"[FAIL] 在 {path.name} 里找不到 get_model()")
    text = text[:m2.end()] + HOOK + text[m2.end():]
    print(f"[patch] {path.name} 已在 get_model() 插入 baselines_ext 分发")

    path.write_text(text, encoding="utf-8", newline="\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".", help="项目根目录（含 train_comparison.py）")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    print(f"项目目录：{root}")
    changed = 0
    for name in TARGETS:
        p = root / name
        if not p.exists():
            print(f"[warn] 未找到 {p}")
            continue
        changed += int(patch_file(p))
    print(f"\n完成：修改 {changed} 个文件。回滚方法：用同目录下的 *.bak_ext 覆盖回去。")


if __name__ == "__main__":
    main()
