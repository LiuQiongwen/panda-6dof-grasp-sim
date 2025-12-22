# -*- coding: utf-8 -*-
"""
把 validate_grasps_panda.py 的输出 JSON
转换成用于 LG-GSN 训练的 CSV 表格。

输入 JSON 每个元素类似：
{
  "position": [x,y,z],
  "rpy": [r,p,y],
  "score": float,
  "width": float,           # 某些生成器会写入，没有就默认 0.04
  "success": bool,
  "contact_seen": bool,
  "dz": float,
  "fell_off": bool,
  "_metrics": {
      "dz_lift": float,
      "need_dz": float,
      "H": float,
      "still_touch": bool,
      ...
  }
}

输出 CSV 列大致为：
x,y,z,roll,pitch,yaw,width,score,dz,dz_lift,need_dz,H,still_touch,
success,fell_off,contact_seen,obj_type,source,label
"""

import os
import json
import csv
import argparse


def infer_obj_type(path: str, override: str | None) -> str:
    """根据文件名简单猜一下物体类型。"""
    if override:
        return override
    name = os.path.basename(path).lower()
    if "cyl" in name:
        return "cylinder"
    if "sphere" in name:
        return "sphere"
    if "cube" in name or "box" in name:
        return "cube"
    if "can" in name:
        return "can"
    if "bottle" in name:
        return "bottle"
    return "unknown"


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_files", nargs="+", required=True,
        help="一个或多个 validate_grasps_panda 输出 JSON 路径"
    )
    ap.add_argument(
        "--out", type=str, required=True,
        help="输出 CSV 路径，例如 grasp_6dof/dataset/all_lggsn.csv"
    )
    ap.add_argument(
        "--obj-type", type=str, default=None,
        help="统一指定物体类型（可选）。若不设则从文件名猜。"
    )
    ap.add_argument(
        "--min-good-dz", type=float, default=0.01,
        help="dz_lift 或 dz 低于这个阈值，即使 success 也视为差抓取(label=0)"
    )
    return ap


def convert_files(args):
    rows = []

    for path in args.in_files:
        if not os.path.isfile(path):
            print(f"[WARN] file not found, skip: {path}")
            continue

        obj_type = infer_obj_type(path, args.obj_type)
        source = os.path.basename(path)

        print(f"[INFO] loading {path} (obj_type={obj_type})")
        with open(path, "r", encoding="utf-8") as f:
            grasps = json.load(f)

        for g in grasps:
            pos = g.get("position") or g.get("pos")
            rpy = g.get("rpy")
            if pos is None or rpy is None:
                continue

            x, y, z = map(float, pos)
            roll, pitch, yaw = map(float, rpy)

            width = float(g.get("width", 0.04))
            score = float(g.get("score", 0.0))

            success = bool(g.get("success", False))
            fell_off = bool(g.get("fell_off", False))
            contact_seen = bool(g.get("contact_seen", False))

            dz = float(g.get("dz", 0.0))

            metrics = g.get("_metrics", {}) or {}
            dz_lift = float(metrics.get("dz_lift", dz))
            need_dz = float(metrics.get("need_dz", 0.0))
            H = float(metrics.get("H", 0.0))
            still_touch = bool(metrics.get("still_touch", contact_seen))

            # ------- 打 label 的规则（可以之后再调）-------
            # 1) 明确失败 or 掉落 → 坏抓
            # 2) 虽然 success=True，但抬起高度太小 → 也当坏抓
            label = 1
            if (not success) or fell_off:
                label = 0
            elif dz_lift < args.min_good_dz:
                label = 0

            row = {
                "x": x,
                "y": y,
                "z": z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "width": width,
                "score": score,
                "dz": dz,
                "dz_lift": dz_lift,
                "need_dz": need_dz,
                "H": H,
                "still_touch": int(still_touch),
                "success": int(success),
                "fell_off": int(fell_off),
                "contact_seen": int(contact_seen),
                "obj_type": obj_type,
                "source": source,
                "label": int(label),
            }
            rows.append(row)

    if not rows:
        print("[WARN] No rows collected, nothing to write.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fieldnames = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "width", "score",
        "dz", "dz_lift", "need_dz", "H", "still_touch",
        "success", "fell_off", "contact_seen",
        "obj_type", "source",
        "label",
    ]

    print(f"[INFO] writing {len(rows)} rows → {args.out}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 简单打印一下正负例分布
    pos = sum(r["label"] == 1 for r in rows)
    neg = sum(r["label"] == 0 for r in rows)
    print(f"[INFO] label distribution: pos={pos}, neg={neg}")


def main():
    args = build_argparser().parse_args()
    convert_files(args)


if __name__ == "__main__":
    main()

