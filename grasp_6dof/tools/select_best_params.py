# -*- coding: utf-8 -*-
"""
从 summary.csv / summary_tune.csv 中自动挑出“平均成功率最高”的参数组合。

用法示例：

  python grasp_6dof/tools/select_best_params.py \
    --csv grasp_6dof/out/summary_tune.csv \
    --obj grasp_6dof/assets/cylinder.urdf \
    --topk 5

如果不加 --obj，就会对 CSV 里出现过的每个 obj 分别给出前 topk 组参数。
"""

import csv
import argparse
from collections import defaultdict


def parse_float(d, key, default=0.0):
    v = d.get(key, "")
    try:
        return float(v)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="grasp_6dof/out/summary_tune.csv",
        help="参数 sweep 写入的 summary CSV 路径",
    )
    ap.add_argument(
        "--obj",
        type=str,
        default=None,
        help="只分析某一个 obj（比如 grasp_6dof/assets/cylinder.urdf），不填则所有 obj 都分析",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=3,
        help="每个 obj 输出前 topk 个参数组合",
    )
    args = ap.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("[WARN] CSV 为空，没有任何记录。")
        return

    # 先按 obj 分桶，方便后面对每个物体分别选最优参数
    rows_by_obj = defaultdict(list)
    for r in rows:
        obj = r.get("obj", "")
        if (args.obj is not None) and (obj != args.obj):
            continue
        rows_by_obj[obj].append(r)

    if not rows_by_obj:
        print(f"[WARN] CSV 中没有找到 obj = {args.obj!r} 的记录。")
        return

    # 对每个 obj，按 (关键参数组合) 分组
    for obj, obj_rows in rows_by_obj.items():
        print("\n" + "=" * 80)
        print(f"[INFO] Object: {obj}  (共 {len(obj_rows)} 条记录)")

        groups = defaultdict(list)

        for r in obj_rows:
            # 这些是你在 validate_grasps_panda.py 里写入 summary 的字段
            key = (
                r.get("obj", ""),
                r.get("cube_scale", ""),
                r.get("ee_index", ""),
                r.get("descent_step", ""),
                r.get("descend_clear", ""),
                r.get("vel_close", ""),
                r.get("pos_close", ""),
                r.get("squeeze", ""),
                # 这里可以选择要不要把 fast/fast_scale 也纳入 key
                r.get("fast", ""),
                r.get("fast_scale", ""),
            )
            groups[key].append(r)

        summary_list = []

        for key, grp in groups.items():
            # 同一 key 下，通常是不同 seed 的多次实验
            sr_list = []
            seeds = []
            for r in grp:
                sr = parse_float(r, "success_rate", 0.0)
                sr_list.append(sr)
                seeds.append(r.get("seed", ""))

            if not sr_list:
                continue

            mean_sr = sum(sr_list) / len(sr_list)
            min_sr = min(sr_list)
            max_sr = max(sr_list)

            summary_list.append(
                {
                    "key": key,
                    "n_runs": len(sr_list),
                    "mean_sr": mean_sr,
                    "min_sr": min_sr,
                    "max_sr": max_sr,
                    "seeds": list(sorted(set(seeds))),
                }
            )

        # 按 mean success_rate 从高到低排序
        summary_list.sort(key=lambda x: x["mean_sr"], reverse=True)

        topk = max(1, args.topk)
        print(f"[INFO] 找到 {len(summary_list)} 组不同参数组合，取前 {topk} 组：\n")

        for i, item in enumerate(summary_list[:topk], start=1):
            (
                obj_name,
                cube_scale,
                ee_index,
                descent_step,
                descend_clear,
                vel_close,
                pos_close,
                squeeze,
                fast,
                fast_scale,
            ) = item["key"]

            print(f"--- Rank {i} ---")
            print(
                f"mean_sr={item['mean_sr']:.3f}  "
                f"(min={item['min_sr']:.3f}, max={item['max_sr']:.3f}, "
                f"n_runs={item['n_runs']}, seeds={item['seeds']})"
            )
            print(
                f"  cube_scale = {cube_scale}\n"
                f"  ee_index   = {ee_index}\n"
                f"  descent_step  = {descent_step}\n"
                f"  descend_clear = {descend_clear}\n"
                f"  vel_close     = {vel_close}\n"
                f"  pos_close     = {pos_close}\n"
                f"  squeeze       = {squeeze}\n"
                f"  fast          = {fast}\n"
                f"  fast_scale    = {fast_scale}\n"
            )

        # 你也可以在这里顺手打印一行“推荐默认参数”
        best = summary_list[0]
        (
            _obj_name,
            cube_scale,
            ee_index,
            descent_step,
            descend_clear,
            vel_close,
            pos_close,
            squeeze,
            fast,
            fast_scale,
        ) = best["key"]

        print("[RECOMMEND] 推荐默认参数（该 obj 上 mean success_rate 最高）：")
        print(
            f"  --cube-scale {cube_scale} "
            f"--ee-index {ee_index} "
            f"--descent-step {descent_step} "
            f"--descend-clear {descend_clear} "
            f"--vel-close {vel_close} "
            f"--pos-close {pos_close} "
            f"--squeeze {squeeze} "
            f"--fast {fast} "
            f"--fast-scale {fast_scale}"
        )
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

