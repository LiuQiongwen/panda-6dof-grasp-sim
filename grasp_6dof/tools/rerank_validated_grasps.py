# -*- coding: utf-8 -*-
"""
根据 validate_grasps_panda.py 的输出 JSON
对每个 grasp 进行重打分和排序，并可选只保留 Top-K。

使用示例：

  python grasp_6dof/tools/rerank_validated_grasps.py \
    --in grasp_6dof/dataset/gen_cylinder_validated_seed1_tuned2.json \
    --out grasp_6dof/dataset/gen_cylinder_validated_seed1_tuned2_top8.json \
    --topk 8

如果不指定 --out，则默认在文件名后加 _reranked。
"""

import json
import argparse
import os
from statistics import mean


def compute_quality(g, dz_weight=1.0, success_bonus=1.0, felloff_penalty=0.5):
    """
    简单的质量评分：
      quality = dz_weight * max(dz, 0) + success_bonus * I(success) - felloff_penalty * I(fell_off)
    后面你可以随时改这个公式。
    """
    dz = float(g.get("dz", 0.0))
    success = bool(g.get("success", False))
    fell_off = bool(g.get("fell_off", False))

    q = dz_weight * max(dz, 0.0)
    if success:
        q += success_bonus
    if fell_off:
        q -= felloff_penalty
    return float(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, required=True,
                    help="validate_grasps_panda 输出的 *_validated*.json 路径")
    ap.add_argument("--out", dest="out_path", type=str, default=None,
                    help="重排后输出的 JSON 路径，不填则自动加 _reranked 后缀")
    ap.add_argument("--topk", type=int, default=0,
                    help="只保留前 K 个 grasp；<=0 表示保留全部")
    ap.add_argument("--dz-weight", type=float, default=1.0,
                    help="dz 权重")
    ap.add_argument("--success-bonus", type=float, default=1.0,
                    help="成功抓取的额外加分")
    ap.add_argument("--felloff-penalty", type=float, default=0.5,
                    help="fell_off=True 的扣分")
    args = ap.parse_args()

    in_path = args.in_path
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if args.out_path is None:
        root, ext = os.path.splitext(in_path)
        out_path = root + "_reranked" + ext
    else:
        out_path = args.out_path

    with open(in_path, "r", encoding="utf-8") as f:
        grasps = json.load(f)

    if not isinstance(grasps, list):
        raise ValueError("输入 JSON 格式错误：顶层应为 list。")

    # 给每个 grasp 计算 quality
    for g in grasps:
        q = compute_quality(
            g,
            dz_weight=args.dz_weight,
            success_bonus=args.success_bonus,
            felloff_penalty=args.felloff_penalty,
        )
        g["_quality"] = round(float(q), 6)

    # 排序
    grasps_sorted = sorted(grasps, key=lambda g: g.get("_quality", 0.0), reverse=True)

    # 可选截断 Top-K
    if args.topk and args.topk > 0:
        grasps_sorted = grasps_sorted[: args.topk]

    # 统计 & 打印信息
    n_all = len(grasps)
    n_keep = len(grasps_sorted)
    succ_all = sum(1 for g in grasps if g.get("success"))
    succ_keep = sum(1 for g in grasps_sorted if g.get("success"))

    sr_all = succ_all / max(1, n_all)
    sr_keep = succ_keep / max(1, n_keep)

    qs_all = [g["_quality"] for g in grasps]
    qs_keep = [g["_quality"] for g in grasps_sorted]

    print("=== Rerank Summary ===")
    print(f"Input file : {in_path}")
    print(f"Output file: {out_path}")
    print(f"Total grasps    : {n_all}")
    print(f"Kept grasps     : {n_keep}")
    print(f"Success (all)   : {succ_all}/{n_all} = {sr_all:.3f}")
    print(f"Success (kept)  : {succ_keep}/{n_keep} = {sr_keep:.3f}")
    print(f"Quality avg(all): {mean(qs_all):.3f}")
    print(f"Quality avg(kept): {mean(qs_keep):.3f}")
    print()
    print("Top 5 (after rerank):")
    for i, g in enumerate(grasps_sorted[:5], start=1):
        print(
            f"#{i}: quality={g['_quality']:.3f}, "
            f"success={bool(g.get('success'))}, dz={g.get('dz')}, "
            f"fell_off={bool(g.get('fell_off'))}, score={g.get('score')}"
        )

    # 写回 JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grasps_sorted, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Saved reranked grasps to:", out_path)


if __name__ == "__main__":
    main()

