# -*- coding: utf-8 -*-
"""
根据 validate_grasps_panda 的输出 JSON，
用 _metrics 计算 grasp 质量分，筛选出高质量子集。

用法示例：
  python grasp_6dof/tools/rank_validated_grasps.py \
    --in grasp_6dof/dataset/gen_cylinder_validated_seed1_tuned2.json \
    --out grasp_6dof/dataset/gen_cylinder_validated_seed1_top8.json \
    --topk 8
"""

import json
import argparse
import os


def compute_quality(g):
    """给单个 grasp 计算质量分数."""
    m = g.get("_metrics", {})

    # dz_lift: 抬起高度；H: 物体高度；still_touch: 抬起后是否仍接触
    dz = float(m.get("dz_lift", g.get("dz", 0.0)))
    H = float(m.get("H", 0.08))  # 没有就给个默认高度
    still = bool(m.get("still_touch", g.get("success", False)))

    # 质量分：抬起高度占物体高度的比例 + 是否保持接触的奖励
    # 你可以之后自己再调这个公式
    quality = dz / max(H, 1e-6) + 0.5 * (1.0 if still else 0.0)
    return float(quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True,
                    help="输入: validate_grasps_panda 输出的 *_validated*.json")
    ap.add_argument("--out", dest="output", required=True,
                    help="输出: 只保留高质量 grasps 的 json")
    ap.add_argument("--topk", type=int, default=None,
                    help="保留前 topk 个 grasp（按 quality 排序）")
    ap.add_argument("--min-quality", type=float, default=None,
                    help="如果设置，只保留 quality >= 这个阈值的 grasp")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        grasps = json.load(f)

    # 计算 quality
    for g in grasps:
        g["quality"] = compute_quality(g)

    # 按质量排序（高→低）
    grasps.sort(key=lambda g: g.get("quality", 0.0), reverse=True)

    # 按阈值过滤
    if args.min_quality is not None:
        grasps = [g for g in grasps if g.get("quality", 0.0) >= args.min_quality]

    # 再按 topk 截断
    if args.topk is not None and args.topk > 0:
        grasps = grasps[:args.topk]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(grasps, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Loaded {len(grasps)} grasps after filtering.")
    if grasps:
        qs = [g["quality"] for g in grasps]
        print(f"[INFO] quality: min={min(qs):.3f}, max={max(qs):.3f}, mean={sum(qs)/len(qs):.3f}")
    print(f"[INFO] Saved → {args.output}")


if __name__ == "__main__":
    main()
