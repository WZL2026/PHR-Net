#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把汇总结果整理成「论文分支可直接引用的数字表」。

输入：results/summary/summary_all_<最新时间戳>.csv（由 collect_results.py 生成）
输出：results/summary/论文数字交接_<时间戳>.md

表格内容：
  * 实测值（Dice / IoU / HD95 / clDice，折平均口径，含折数）
  * 论文报告值（含出处类型：原文 / 第三方 / 原文未报告）
  * 差距与结论

用法：
    python scripts/analysis/make_paper_numbers.py
    python scripts/analysis/make_paper_numbers.py --summary results/summary/summary_all_20261016_072810.csv
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUM_DIR = REPO / "results" / "summary"
DATASETS = ("DRIVE", "CHASEDB1", "STARE")

# 论文报告值（来自 docs/对比模型复现审查计划.md §八，已逐篇查证）
# 值统一写成 Dice/F1 口径；括号注明口径或出处类型。
PAPER = {
    "PHR_Net":   {"DRIVE": "—（本文方法）", "CHASEDB1": "—", "STARE": "—"},
    "CENet":     {"DRIVE": "无 Dice（原文 Sen 0.8309 / Acc 0.9545 / AUC 0.9779）",
                  "CHASEDB1": "原文未报告", "STARE": "原文未报告"},
    "FRUNet":    {"DRIVE": "0.8316（原文 F1）", "CHASEDB1": "0.8151（原文 F1）", "STARE": "0.8330（原文 F1）"},
    "DSCNet":    {"DRIVE": "—", "CHASEDB1": "—", "STARE": "—"},
    "UNet":      {"DRIVE": "—（常规基线）", "CHASEDB1": "—", "STARE": "—"},
    "ResUNet":   {"DRIVE": "—（常规基线）", "CHASEDB1": "—", "STARE": "—"},
    "ERFNet":    {"DRIVE": "0.8022（第三方，预训练+544²）", "CHASEDB1": "0.7994（第三方）", "STARE": "无任何来源"},
    "SwinUnet":  {"DRIVE": "0.7324（第三方）", "CHASEDB1": "0.6676（第三方）", "STARE": "0.5541（第三方）"},
    "M2UNet":    {"DRIVE": "0.8091（原文）", "CHASEDB1": "0.8006（原文）", "STARE": "原文未报告"},
    "DRIU":      {"DRIVE": "0.822（原文 Fig.2）", "CHASEDB1": "原文未报告", "STARE": "0.831（原文 Fig.2）"},
    "TransUNet": {"DRIVE": "0.8039（第三方）", "CHASEDB1": "0.7964（第三方）", "STARE": "0.8106（第三方）"},
    "MambaVesselNet": {"DRIVE": "原文++版评过，数值未取到", "CHASEDB1": "原文未报告", "STARE": "原文未报告"},
    "UMamba":    {"DRIVE": "无来源", "CHASEDB1": "0.7785（第三方）", "STARE": "无来源"},
    "VMUNet":    {"DRIVE": "无来源", "CHASEDB1": "无来源", "STARE": "无来源"},
    "SwinUMamba": {"DRIVE": "无来源", "CHASEDB1": "无来源", "STARE": "无来源"},
    "NETracer":  {"DRIVE": "—（已移出主表）", "CHASEDB1": "—", "STARE": "—"},
    "SerpMamba": {"DRIVE": "—（域不同，已移出主表）", "CHASEDB1": "—", "STARE": "—"},
}

METRICS = ("Dice", "IoU", "HD95", "clDice")


def latest_summary() -> Path | None:
    files = sorted(SUM_DIR.glob("summary_all_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None, help="指定 summary_all_*.csv；默认取最新")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    src = Path(args.summary) if args.summary else latest_summary()
    if not src or not src.exists():
        print("[FAIL] 找不到 results/summary/summary_all_*.csv；先跑 collect_results.py")
        return 1
    rows = load_rows(src)
    print(f"[OK] 读入 {src.name}：{len(rows)} 行")

    # index: (dataset, model) -> row
    idx: dict[tuple[str, str], dict] = {}
    models: list[str] = []
    for r in rows:
        d = (r.get("Dataset") or "").strip()
        m = (r.get("Model") or "").strip()
        if not d or not m:
            continue
        idx[(d, m)] = r
        if m not in models:
            models.append(m)

    # 排序：PHR_Net 在最前，其余按参数量
    def params(r) -> float:
        p = (r.get("Params") or "").strip().rstrip("M")
        try:
            return float(p)
        except ValueError:
            return 1e9

    first = [m for m in models if m == "PHR_Net"]
    rest = sorted([m for m in models if m != "PHR_Net"],
                  key=lambda m: params(idx.get((DATASETS[0], m), idx.get(("DRIVE", m), {}))))
    ordered = first + rest

    def cell(model: str, dataset: str, metric: str) -> str:
        r = idx.get((dataset, model))
        if not r:
            return "—"
        v = (r.get(metric) or "").strip()
        return v if v else "—"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else SUM_DIR / f"论文数字交接_{ts}.md"
    L: list[str] = []
    L.append(f"# 论文数字交接表（{datetime.now().strftime('%Y-%m-%d %H:%M')}）")
    L.append("")
    L.append(f"- 数据来源：`{src.relative_to(REPO).as_posix()}`（{len(rows)} 行）")
    L.append("- 口径：Dice/IoU/clDice 为折平均（DRIVE 单次评估，CHASE_DB1/STARE 为 5 折均值）；"
             "HD95 单位像素，越小越好。")
    L.append("- 论文报告值来自 `docs/对比模型复现审查计划.md` 第八节（逐篇查证，注明原文/第三方）。")
    L.append("- **本表只做数据搬运，不改论文**；论文分支引用时请保留“协议差异”说明。")
    L.append("")

    L.append("## 1. 主表（Dice）")
    L.append("")
    L.append("| 模型 | 参数量 | DRIVE | CHASE_DB1 | STARE |")
    L.append("|---|---|---|---|---|")
    for m in ordered:
        p = (idx.get(("DRIVE", m)) or idx.get(("CHASEDB1", m)) or {}).get("Params", "—")
        L.append(f"| {m} | {p} | {cell(m,'DRIVE','Dice')} | {cell(m,'CHASEDB1','Dice')} | {cell(m,'STARE','Dice')} |")
    L.append("")

    L.append("## 2. 论文报告值对照（口径见括号）")
    L.append("")
    L.append("| 模型 | DRIVE | CHASE_DB1 | STARE |")
    L.append("|---|---|---|---|")
    for m in ordered:
        pp = PAPER.get(m)
        if not pp:
            continue
        L.append(f"| {m} | {pp.get('DRIVE','—')} | {pp.get('CHASEDB1','—')} | {pp.get('STARE','—')} |")
    L.append("")

    L.append("## 3. 其它指标")
    for metric in ("IoU", "HD95", "clDice"):
        L.append("")
        L.append(f"**{metric}**")
        L.append("")
        L.append("| 模型 | DRIVE | CHASE_DB1 | STARE |")
        L.append("|---|---|---|---|")
        for m in ordered:
            L.append(f"| {m} | {cell(m,'DRIVE',metric)} | {cell(m,'CHASEDB1',metric)} | {cell(m,'STARE',metric)} |")
    L.append("")

    na = [(d, m) for (d, m), r in idx.items() if (r.get("Dice") or "").strip().upper() in ("N/A", "NA", "")]
    if na:
        L.append("## 4. N/A / 未跑通清单（论文里写 N/A 并说明原因）")
        L.append("")
        for d, m in sorted(na):
            L.append(f"- {m} / {d}")
        L.append("")

    L.append("## 5. 缺口与结论（实验分支填写）")
    L.append("")
    L.append("> 待服务器跑完后补充：每个模型与目标分的差距、是否达标、协议差异说明。")
    L.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"[OK] 已生成 {out.relative_to(REPO).as_posix()}")
    print(f"     主表 {len(ordered)} 个模型；N/A {len(na)} 项")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
