#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把几何 (summary_ycb.csv) 和 随机 (summary_ycb_random.csv) 的 success_rate 合并成一张表
"""

import pandas as pd

geom = pd.read_csv("results/summary_ycb.csv")
rand = pd.read_csv("results/summary_ycb_random.csv")

# 提取 obj_id：从 obj 路径里取文件名
geom["obj_id"] = geom["obj"].apply(lambda s: str(s).split("/")[-1].replace(".urdf",""))
rand["obj_id"] = rand["obj"].apply(lambda s: str(s).split("/")[-1].replace(".urdf",""))

g = geom[["obj_id", "success_rate"]].rename(columns={"success_rate": "SR_geom"})
r = rand[["obj_id", "success_rate"]].rename(columns={"success_rate": "SR_random"})

df = g.merge(r, on="obj_id", how="inner")
df["delta"] = df["SR_geom"] - df["SR_random"]

df = df.sort_values("obj_id").reset_index(drop=True)
print(df.to_string(index=False))

df.to_csv("results/ycb_geom_vs_random.csv", index=False)
print("\n[INFO] Saved comparison to results/ycb_geom_vs_random.csv")

