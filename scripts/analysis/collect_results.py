#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 results/comparison_*.csv → 论文主表（Markdown + LaTeX + 合并 CSV）。

特性：
  - 同一数据集可能有多次运行的 CSV：按 (数据集, 模型, 折) 取**最新**那次的结果；
  - 多折（CHASEDB1 / STARE 的 5 折）自动聚合为 mean±std；
  - 输出 Markdown 表格、LaTeX 行、以及合并后的 CSV；
  - 报告缺失模型（预期列表 vs 实际结果），避免"某个模型没跑完却不知道"。

用法：
    python scripts/analysis/collect_results.py
    python scripts/analysis/collect_results.py --latex
    python scripts/analysis/collect_results.py --expected PHR_Net UNet ResUNet DSCNet
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# --- 定位项目根目录（脚本归档在 scripts/analysis/ 下）---
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not (
        os.path.isdir(os.path.join(_p, "models")) and os.path.isdir(os.path.join(_p, "utils"))):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
os.chdir(_p)

METRICS = ["Dice", "IoU", "Precision", "Recall", "Spec", "FPR", "HD95", "clDice"]
LOWER_BETTER = {"FPR", "HD95"}
DATASETS = ["DRIVE", "CHASEDB1", "STARE"]

# 主表默认预期模型（NETracer 已移出；LHRNet/HRDNet 跑过可加进来）
DEFAULT_EXPECTED = ["PHR_Net", "UNet", "ResUNet", "DSCNet", "MambaVesselNet",
                    "CENet", "FRUNet", "M2UNet", "ERFNet", "DRIU",
                    "VMUNet", "SwinUnet", "TransUNet", "UMamba", "SwinUMamba", "SerpMamba"]


def to_float(s) -> float | None:
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def parse_cell(s) -> tuple[float | None, float | None]:
    """解析单元格，支持两种格式：

      - 单折原始值：      "0.8260"                -> (0.8260, None)
      - 已聚合的 5 折值： "0.8337+-0.0095"        -> (0.8337, 0.0095)
        （也兼容 "0.8337±0.0095"、"0.8337 +- 0.0095"）
    """
    if s is None:
        return None, None
    txt = str(s).strip().replace("±", "+-")
    if not txt or txt in {"-", "nan", "None"}:
        return None, None
    if "+-" in txt:
        a, b = txt.split("+-", 1)
        m = to_float(a)
        sd = to_float(b)
        return (m, sd if sd is not None else 0.0)
    return to_float(txt), None


def load_all(results_dir: Path, since: str = "", drop: tuple[str, ...] = ()) -> tuple[dict, list[str]]:
    """返回 {(dataset, model, fold): {metric: (mean, std|None), 'Params': str, 'src': 文件名}}。

    - 同名文件按修改时间升序覆盖，因此每个 (数据集,模型,折) 取最新一次运行的结果；
    - since 形如 "20261015"：只采用文件名时间戳 >= since 的运行（用于排除历史运行混入）；
    - drop：要排除的模型名（例如已移出主表的 NETracer）；
    - 指标为 N/A 的模型（评估未产出结果）不纳入。
    """
    records: dict[tuple, dict] = {}
    files = sorted(results_dir.glob("comparison_*.csv"), key=lambda p: p.stat().st_mtime)
    for f in files:
        m = re.match(r"comparison_(drive|chasedb1|stare)_(\d{8})_", f.name)
        if not m:
            m = re.match(r"comparison_(drive|chasedb1|stare)_", f.name)
            stamp = ""
        else:
            stamp = m.group(2)
        if since and stamp and stamp < since:
            continue
        ds = m.group(1).upper()
        with f.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                model = (row.get("Model") or "").strip()
                if not model or model in drop:
                    continue
                fold = (row.get("Fold") or "-").strip() or "-"
                rec = {"Params": (row.get("Params") or "").strip(), "src": f.name,
                       "fold_label": fold}
                for k in METRICS:
                    rec[k] = parse_cell(row.get(k))
                records[(ds, model, fold)] = rec
    return records, [f.name for f in files]


def agg(pairs: list[tuple[float, float | None]]) -> tuple[float, float]:
    """pairs 为 [(mean, std|None)]。

    若其中存在自带 std 的**已聚合**记录（CSV 里 5 折那种），直接采用它；
    否则把多折原始值聚合成 mean±std。
    """
    agg_rows = [(m, s) for m, s in pairs if m is not None and s is not None]
    if agg_rows:
        m, s = agg_rows[-1]          # 最新一次的聚合结果
        return m, float(s)
    vals = [m for m, _ in pairs if m is not None]
    if not vals:
        return float("nan"), 0.0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var ** 0.5


def build_table(records: dict, dataset: str) -> dict:
    """{model: {'params': str, 'n_folds': int, metrics: (mean, std)}}"""
    per_model: dict[str, dict] = defaultdict(lambda: {"params": "", "folds": 0,
                                                      "vals": {k: [] for k in METRICS}})
    for (ds, model, fold), rec in records.items():
        if ds != dataset:
            continue
        e = per_model[model]
        e["params"] = rec["Params"] or e["params"]
        e["folds"] += 1
        for k in METRICS:
            e["vals"][k].append(rec[k])
    out = {}
    for model, e in per_model.items():
        metrics = {k: agg(e["vals"][k]) for k in METRICS}
        # 全部指标都是 NaN（评估未产出结果，如 N/A）→ 不纳入表格
        if all(v[0] != v[0] for v in metrics.values()):
            continue
        out[model] = {"params": e["params"], "n_folds": e["folds"], "metrics": metrics}
    return out


def fmt(mean: float, std: float, multi: bool) -> str:
    if mean != mean:  # NaN
        return "-"
    if multi and std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def is_best(table: dict, model: str, metric: str) -> bool:
    vals = [v["metrics"][metric][0] for v in table.values()
            if v["metrics"][metric][0] == v["metrics"][metric][0]]
    if not vals:
        return False
    target = table[model]["metrics"][metric][0]
    best = min(vals) if metric in LOWER_BETTER else max(vals)
    return abs(target - best) < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/summary")
    ap.add_argument("--expected", nargs="*", default=None)
    ap.add_argument("--latex", action="store_true", help="额外打印 LaTeX 行")
    ap.add_argument("--sort", default="params", choices=["params", "dice", "name"])
    ap.add_argument("--since", default="", help='只采用文件名时间戳不早于该日期的运行，如 --since 20261015')
    ap.add_argument("--drop", nargs="*", default=["NETracer"], help="排除的模型名（默认 NETracer）")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = args.expected if args.expected else DEFAULT_EXPECTED
    drop = tuple(args.drop or ())

    records, files = load_all(results_dir, since=args.since, drop=drop)
    scope = f"（仅 {args.since} 之后的运行）" if args.since else ""
    print(f"扫描到 {len(files)} 个 comparison_*.csv{scope}，共 {len(records)} 条 (数据集,模型,折) 记录")
    if drop:
        print(f"已排除模型：{', '.join(drop)}")
    print()
    if not records:
        print("没有可用结果。等训练产出 results/comparison_*.csv 后再跑。")
        return

    seen_models: dict[str, set] = defaultdict(set)
    merged_rows = []
    for ds in DATASETS:
        table = build_table(records, ds)
        if not table:
            continue
        for m in table:
            seen_models[m].add(ds)
        # 排序
        if args.sort == "params":
            def pnum(m):
                s = table[m]["params"].replace("M", "").strip()
                try:
                    return float(s)
                except ValueError:
                    return 1e9
            order = sorted(table, key=lambda m: (m != "PHR_Net", pnum(m)))
        elif args.sort == "dice":
            order = sorted(table, key=lambda m: -table[m]["metrics"]["Dice"][0])
        else:
            order = sorted(table)

        multi = any(table[m]["metrics"][k][1] > 0 for m in table for k in METRICS)
        print("=" * 104)
        print(f"  {ds}   （{len(order)} 个模型{'，多折 mean±std' if multi else ''}）")
        print("=" * 104)
        head = f"| {'Model':<14} | {'Params':>9} | " + " | ".join(f"{k:>9}" for k in METRICS) + " |"
        print(head)
        print("|" + "-" * 16 + "|" + "-" * 11 + "|" + "|".join(["-" * 11] * len(METRICS)) + "|")
        for m in order:
            cells = []
            for k in METRICS:
                mean, std = table[m]["metrics"][k]
                txt = fmt(mean, std, multi)
                if is_best(table, m, k) and txt != "-":
                    txt = f"**{txt}**"
                cells.append(f"{txt:>9}")
            print(f"| {m:<14} | {table[m]['params']:>9} | " + " | ".join(cells) + " |")
            merged_rows.append([ds, m, table[m]["params"], table[m]["n_folds"]]
                               + [fmt(*table[m]["metrics"][k], multi) for k in METRICS])
        print()

        if args.latex:
            print(f"% ---- {ds} ----")
            for m in order:
                cells = " & ".join(fmt(*table[m]["metrics"][k], multi) for k in METRICS)
                print(f"{m} & {table[m]['params']} & {cells} \\\\")
            print()

    # 合并 CSV
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_csv = out_dir / f"summary_all_{stamp}.csv"
    with merged_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Dataset", "Model", "Params", "Folds"] + METRICS)
        w.writerows(merged_rows)
    print(f"合并结果已写入：{merged_csv}")

    # 缺失模型报告
    missing = []
    for ds in DATASETS:
        table = build_table(records, ds)
        for m in expected:
            if m not in table:
                missing.append(f"{ds}:{m}")
    if missing:
        print(f"\n[!] 尚无结果的模型（{len(missing)} 项）：")
        for x in missing:
            print("   ", x)
        print("   -> 可能还在排队/训练中；等作业结束再跑一次本脚本即可。")
    else:
        print("\n[OK] 预期模型在三个数据集上都有结果。")


if __name__ == "__main__":
    main()
