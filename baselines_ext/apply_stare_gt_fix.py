#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复 STARE 真值命名不匹配（幂等，自动备份 .bak_gtfix）。

背景：`utils/dataset.py::STAREDataset.__getitem__` 硬编码找 `masks/imXXXX_ah.png`，
而常见的 STARE 数据组织是 `masks/imXXXX.png`（无后缀）或 `_vk.ppm` 等，于是训练时
DataLoader 报 `FileNotFoundError: ../Dataset/STARE/masks/im0082_ah.png`。

改法：新增 `_resolve_stare_gt(gt_dir, base_name)`，按 `_ah → 无后缀 → _vk → …` 与多种扩展名
依次探测，并自动剥掉 base_name 末尾的 `_ah/_vk`（避免 `im0082_ah_ah.png` 这类二次拼接）。
对 `utils/dataset.py` 与 5 个出图脚本同样生效。

用法：
    python baselines_ext/apply_stare_gt_fix.py --project .          # 预览
    python baselines_ext/apply_stare_gt_fix.py --project . --apply  # 写入
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

HELPER = '''

def _resolve_stare_gt(gt_dir: str, base_name: str) -> str:
    """STARE 真值文件名兼容解析（GTFIX）。

    STARE 的真值在不同数据源里可能是 imXXXX_ah.png、imXXXX.png、imXXXX_vk.ppm 等，
    这里按常见命名与扩展名依次探测，并把 base_name 末尾的 _ah/_vk 先剥掉，
    避免出现 imXXXX_ah_ah.png 这种二次拼接。
    """
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
    import glob as _glob
    hits = sorted(_glob.glob(os.path.join(gt_dir, f"{base}*")))
    if hits:
        return hits[0]
    try:
        sample = sorted(os.listdir(gt_dir))[:6]
    except OSError:
        sample = []
    raise FileNotFoundError(
        f"STARE 真值未找到：{gt_dir}/{base}[_ah|_vk]<ext>；该目录示例：{sample}")

'''

# (旧字符串, 新字符串)
REPL = [
    # utils/dataset.py
    ('        gt_path = os.path.join(self.gt_dir, f"{base_name}_ah.png")',
     '        gt_path = _resolve_stare_gt(self.gt_dir, base_name)  # GTFIX'),
    # 出图脚本：细血管分析
    ('    gt_path = os.path.join(mask_dir, f"{base}_ah.png")',
     '    gt_path = _resolve_stare_gt_local(mask_dir, base)  # GTFIX'),
    # 出图脚本：分割对比
    ('    gt_path = os.path.join(DATA_DIR, "masks", f"{iname.replace(\'.png\',\'\')}_ah.png")',
     '    gt_path = _resolve_stare_gt_local(os.path.join(DATA_DIR, "masks"), iname.replace(".png", ""))  # GTFIX'),
    # 出图脚本：ROI 候选（两个版本 + 最终版）
    ('        f"{DATA_DIR}/masks/{base}_ah.png")',
     '        _resolve_stare_gt_local(f"{DATA_DIR}/masks", base))  # GTFIX'),
    ('               lambda f: f.replace(\'.png\', \'_ah.png\')),',
     '               lambda f: _resolve_stare_gt_local(f"{DATA_DIR}/masks", f.replace(".png", ""))),  # GTFIX'),
    ('    gt = cv2.imread(f"{DATA_DIR}/masks/{iname.replace(\'.png\',\'_ah.png\')}", cv2.IMREAD_GRAYSCALE)',
     '    gt = cv2.imread(_resolve_stare_gt_local(f"{DATA_DIR}/masks", iname.replace(".png", "")), cv2.IMREAD_GRAYSCALE)  # GTFIX'),
]

LOCAL_HELPER = '''

def _resolve_stare_gt_local(gt_dir, base_name):
    """STARE 真值兼容解析（GTFIX，出图脚本用，逻辑同 utils/dataset.py）。"""
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
    hits = sorted(glob.glob(os.path.join(gt_dir, f"{base}*"))) if "glob" in globals() else []
    if hits:
        return hits[0]
    raise FileNotFoundError(f"STARE 真值未找到：{gt_dir}/{base}[_ah|_vk]<ext>")

'''

TARGETS = [
    "utils/dataset.py",
    "scripts/visualization/visualize_fine_vessel_analysis.py",
    "scripts/visualization/visualize_segmentation_comparison.py",
    "scripts/visualization/visualize_roi_candidates.py",
    "scripts/visualization/visualize_roi_candidates_v2.py",
    "scripts/visualization/visualize_roi_grid_final.py",
]


def patch_file(p: Path, apply: bool) -> str:
    if not p.exists():
        return f"[skip] 不存在 {p}"
    text = p.read_text(encoding="utf-8")
    if "GTFIX" in text:
        return f"[skip] {p.name} 已修复过"
    hits = [(o, n) for o, n in REPL if o in text]
    if not hits:
        return f"[skip] {p.name} 未找到需要替换的行"

    new = text
    for o, n in hits:
        new = new.replace(o, n, 1)

    # 注入 helper：放在第一个 class/def 之前
    if "utils/dataset.py" in str(p).replace("\\", "/"):
        m = re.search(r"^class ", new, re.M)
        idx = m.start() if m else len(new)
        new = new[:idx] + HELPER.lstrip("\n") + "\n" + new[idx:]
    else:
        m = re.search(r"^(class |def |if __name__)", new, re.M)
        idx = m.start() if m else len(new)
        new = new[:idx] + LOCAL_HELPER.lstrip("\n") + "\n" + new[idx:]

    if apply:
        bak = p.with_suffix(p.suffix + ".bak_gtfix")
        if not bak.exists():
            shutil.copy2(p, bak)
        p.write_text(new, encoding="utf-8", newline="\n")
        return f"[fix] {p.name}（替换 {len(hits)} 处，备份 {bak.name}）"
    return f"[预览] {p.name} 将替换 {len(hits)} 处"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    for rel in TARGETS:
        print(patch_file(root / rel, args.apply))
    if not args.apply:
        print("\n（预览模式；加 --apply 才会写入）")


if __name__ == "__main__":
    main()
