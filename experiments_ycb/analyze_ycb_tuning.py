#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析 results/summary_ycb_tune.csv：
- 对每个 YCB 物体，找到 success_rate 最大的那一组 (ds, vc, sq) 作为 tuned
- 找到默认参数组 (0.0015, 0.8, 0.8) 的 success_rate 作为 default
- 输出 ycb_tuning_results.csv，包含：obj_id, SR_default, SR_tuned, best_ds, best_vc, best_sq, delta
"""

import pandas as pd
from pathlib import Path

SUMMARY_TUNE = "results/summary_ycb_tune.csv"

# 默认参数（要和脚本里用的一致）
DS_DEFAULT = 0.0015
VC_DEFAULT = 0.8
SQ_DEFAULT = 0.8

def main():
    df = pd.read_csv(SUMMARY_TUNE)

    # 从 obj 路径中抽取简单对象名（例如 "YCB_Dataset/ycb/cracker_box.urdf" -> "cracker_box"）
    df["obj_id"] = df["obj"].apply(lambda s: Path(str(s)).stem.replace(".urdf", ""))

    # 保留我们关心的列
    # 你可以先 print(df.columns) 看一眼，如果列名不同就稍微改一下
    cols_needed = ["obj_id", "success_rate", "descent_step", "vel_close", "squeeze"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {SUMMARY_TUNE}, actual columns: {df.columns.tolist()}")

    df_small = df[cols_needed].copy()

    results = []

    for obj_id, group in df_small.groupby("obj_id"):
        # 找 tuned：success_rate 最大的那一行
        best_idx = group["success_rate"].idxmax()
        best_row = group.loc[best_idx]

        SR_tuned = float(best_row["success_rate"])
        best_ds = float(best_row["descent_step"])
        best_vc = float(best_row["vel_close"])
        best_sq = float(best_row["squeeze"])

        # 找 default：参数等于默认那一行
        cond = (
            (abs(group["descent_step"] - DS_DEFAULT) < 1e-9) &
            (abs(group["vel_close"]   - VC_DEFAULT) < 1e-9) &
            (abs(group["squeeze"]     - SQ_DEFAULT) < 1e-9)
        )
        g_def = group[cond]
        if len(g_def) == 0:
            # 万一没扫到默认参数（理论上不会），就设成 NaN
            SR_default = float("nan")
        else:
            SR_default = float(g_def["success_rate"].iloc[0])

        results.append({
            "obj_id": obj_id,
            "SR_default": SR_default,
            "SR_tuned": SR_tuned,
            "best_ds": best_ds,
            "best_vc": best_vc,
            "best_sq": best_sq,
            "delta": SR_tuned - SR_default
        })

    df_res = pd.DataFrame(results).sort_values("obj_id").reset_index(drop=True)
    print(df_res.to_string(index=False))

    out_path = "results/ycb_tuning_results.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved tuning results to {out_path}")

if __name__ == "__main__":
    main()

