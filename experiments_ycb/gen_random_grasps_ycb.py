#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成与几何 grasp 同格式的随机 6-DoF 抓取，用于 YCB baseline：
- 读取 objects_ycb.txt
- 每个物体生成 N=256 个随机 grasps
- 保存到 data/grasps_raw/{obj_id}_random_K256.json
"""

import json
import math
import os
import random
from pathlib import Path

N_PER_OBJ = 256
OBJ_LIST = "objects_ycb.txt"

# 随机范围，可根据需要微调
X_MIN, X_MAX = 0.35, 0.45
Y_MIN, Y_MAX = -0.10, 0.10
Z_OFFSET_MIN, Z_OFFSET_MAX = 0.02, 0.10
WIDTH_MIN, WIDTH_MAX = 0.02, 0.08

def sample_random_grasp(target_h: float):
    """采样一个随机 grasp，返回与 validate_grasps_panda 兼容的 dict"""
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    z = random.uniform(target_h + Z_OFFSET_MIN, target_h + Z_OFFSET_MAX)
    roll = random.uniform(-math.pi, math.pi)
    pitch = random.uniform(-math.pi, math.pi)
    yaw = random.uniform(-math.pi, math.pi)
    width = random.uniform(WIDTH_MIN, WIDTH_MAX)

    return {
        "position": [x, y, z],
        "rpy": [roll, pitch, yaw],
        "width": width,
        # 打个占位分数，后面不会用，只是字段要齐全
        "score": 0.0,
        "meta": {"type": "random_baseline"}
    }


def main():
    os.makedirs("data/grasps_raw", exist_ok=True)

    with open(OBJ_LIST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            oid, urdf_path, target_h = line.split()
            target_h = float(target_h)

            out_path = Path("data/grasps_raw") / f"{oid}_random_K{N_PER_OBJ}.json"
            print(f"[INFO] Generating random grasps for {oid} -> {out_path}")

            grasps = [sample_random_grasp(target_h) for _ in range(N_PER_OBJ)]

            with open(out_path, "w", encoding="utf-8") as fout:
                json.dump(grasps, fout, indent=2)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

