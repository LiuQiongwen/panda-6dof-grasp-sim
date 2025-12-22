# -*- coding: utf-8 -*-
"""
针对不同物体，集中管理：
- 默认的 grasp 候选文件（已用 validator + rerank 处理）
- 默认的抓取控制参数（descend_clear / descent_step / vel_close / pos_close / squeeze ...）

后续 OWG 在执行时，可以统一通过这里取配置。
"""

import os
from pathlib import Path
import json

# 你刚才 select_best_params.py 输出的 Rank 1 参数
_DEFAULT_CTRL_CYLINDER = dict(
    cube_scale=0.08,
    ee_index=11,
    descent_step=0.0006,
    descend_clear=0.02,
    vel_close=0.8,
    pos_close=700.0,
    squeeze=1.0,
    fast=True,
    fast_scale=0.9,
)

# 可以继续给其他物体加，先留个 default
_DEFAULT_CTRL_GENERIC = dict(
    cube_scale=0.08,
    ee_index=11,
    descent_step=0.0006,
    descend_clear=0.02,
    vel_close=0.6,
    pos_close=540.0,
    squeeze=1.0,
    fast=True,
    fast_scale=0.9,
)

# 物体名（stem）到配置的映射
GRASP_LIBRARY = {
    "cylinder": {
        "grasp_json": "grasp_6dof/dataset/gen_cylinder_validated_seed1_tuned2_top8.json",
        "ctrl": _DEFAULT_CTRL_CYLINDER,
    },
    # 后续比如： "mug": {...}, "bottle": {...}
}


def _obj_key_from_urdf(obj_urdf_path: str) -> str:
    """例如 grasp_6dof/assets/cylinder.urdf -> cylinder"""
    return Path(obj_urdf_path).stem


def get_default_ctrl_params(obj_urdf_path: str) -> dict:
    key = _obj_key_from_urdf(obj_urdf_path)
    entry = GRASP_LIBRARY.get(key)
    if entry is not None:
        return dict(entry["ctrl"])
    # fallback：用 generic
    return dict(_DEFAULT_CTRL_GENERIC)


def load_reranked_grasps(obj_urdf_path: str) -> list:
    """
    优先尝试加载为该物体准备的“高质量 grasps JSON”。
    如果不存在，就返回空 list（由上层决定 fallback 策略）。
    """
    key = _obj_key_from_urdf(obj_urdf_path)
    entry = GRASP_LIBRARY.get(key)
    if entry is None:
        return []

    grasp_path = entry["grasp_json"]
    if not os.path.exists(grasp_path):
        print(f"[WARN] Reranked grasp file not found: {grasp_path}")
        return []

    with open(grasp_path, "r", encoding="utf-8") as f:
        grasps = json.load(f)
    if not isinstance(grasps, list):
        print(f"[WARN] Unexpected format in {grasp_path}, expect list.")
        return []
    return grasps

