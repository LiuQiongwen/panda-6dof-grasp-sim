#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 YCB 多物体 6-DoF 几何抓取结果：
- 从 data/grasps_val/*_geom_val.json 里读取每个抓取的 success
- 按 score 降序计算 SR@k（k = 1,3,5,10,20,32,64）
- 汇总成 results/ycb_sr_table.csv
- 可选：画一张 SR@k 曲线图 results/ycb_sr_curves.png
"""

import os
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 想看的 k 值
K_LIST = [1, 3, 5, 10, 20, 32, 64]


def compute_sr_for_file(json_path: str) -> dict:
    """对单个 *_geom_val.json 计算整体成功率和 SR@k"""
    with open(json_path, "r", encoding="utf-8") as f:
        grasps = json.load(f)

    if not isinstance(grasps, list):
        raise ValueError(f"{json_path} is not a list")

    n = len(grasps)
    if n == 0:
        return {"n_grasps": 0, "success_rate_json": 0.0, **{f"SR@{k}": 0.0 for k in K_LIST}}

    # 按 score 降序（如果有）
    if "score" in grasps[0]:
        grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)

    succ = np.array([1.0 if g.get("success") else 0.0 for g in grasps], dtype=float)
    overall = float(succ.mean())

    res = {"n_grasps": int(n), "success_rate_json": overall}
    for k in K_LIST:
        kk = min(k, n)
        res[f"SR@{k}"] = float(succ[:kk].mean()) if kk > 0 else 0.0
    return res


def main():
    # 1) 找到所有 *_geom_val.json
    val_files = sorted(glob.glob("data/grasps_val/*_geom_val.json"))
    if not val_files:
        print("[ERROR] No files matched data/grasps_val/*_geom_val.json")
        return

    rows = []
    for f in val_files:
        stem = Path(f).stem  # e.g. "cracker_box_geom_val"
        obj_id = stem.replace("_geom_val", "")
        stats = compute_sr_for_file(f)
        stats["obj_id"] = obj_id
        stats["file"] = f
        rows.append(stats)

    df_sr = pd.DataFrame(rows)
    df_sr = df_sr.sort_values("obj_id").reset_index(drop=True)

    # 2) 如果有 summary_ycb.csv，把 summary 的整体成功率也 merge 进来
    summary_path = "results/summary_ycb.csv"
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        # 从 obj 路径里提取简单名字，比如 "YCB_Dataset/ycb/cracker_box.urdf" -> "cracker_box"
        df_sum["obj_id"] = df_sum["obj"].apply(
            lambda s: Path(str(s)).stem if isinstance(s, str) else str(s)
        )
        df_sum_small = df_sum[["obj_id", "success_rate"]].rename(
            columns={"success_rate": "success_rate_summary"}
        )
        df_sr = df_sr.merge(df_sum_small, on="obj_id", how="left")

    os.makedirs("results", exist_ok=True)
    out_csv = "results/ycb_sr_table.csv"
    df_sr.to_csv(out_csv, index=False)
    print(f"[INFO] Saved SR table to {out_csv}\n")
    print(df_sr.to_string(index=False))

    # 3) 画 SR@k 曲线图（所有物体共用一张）
    try:
        plt.figure()
        for _, row in df_sr.iterrows():
            y = [row[f"SR@{k}"] for k in K_LIST]
            plt.plot(K_LIST, y, marker="o", label=row["obj_id"])
        plt.xlabel("k")
        plt.ylabel("SR@k (success rate)")
        plt.title("YCB objects: SR@k curves (geometric grasps)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        out_fig = "results/ycb_sr_curves.png"
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved SR@k curves figure to {out_fig}")
    except Exception as e:
        print(f"[WARN] Failed to plot SR@k curves: {e}")


if __name__ == "__main__":
    main()

