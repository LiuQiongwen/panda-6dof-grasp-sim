# -*- coding: utf-8 -*-
import argparse
import json
import os
import random

import numpy as np
import open3d as o3d  # 只在这里 import，一律不要在 main 里再改 o3d

from grasp_sampler import sample_grasps_from_mesh, pack_for_json



def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description="6-DoF grasp generator using Open3D & grasp_sampler.py"
    )
    parser.add_argument(
        "--obj",
        type=str,
        required=True,
        help="mesh file path (.ply/.obj/.stl, etc.)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="output json file for sampled grasps",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2048,
        help="number of raw grasp samples to generate",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=512,
        help="keep top-K grasps by score",
    )
    parser.add_argument(
        "--table_z",
        type=float,
        default=0.0,
        help="table top height used when sampling grasps",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.002,
        help="voxel size for down-sampling the mesh/pcd",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )

    args = parser.parse_args()
    set_global_seed(args.seed)

    mesh_path = args.obj
    print(f"[INFO] Loading mesh: {mesh_path}")

    # 这里的调用只是为了检查文件是否能被 Open3D 正常读取
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(
            f"[ERROR] Mesh is empty or has no triangles: {mesh_path}.\n"
            f"请确认这是一个三角网格，而不是点云。"
        )

    print(
        f"[INFO] Sampling grasps: n={args.n}, table_z={args.table_z}, "
        f"down_sample={args.voxel:.4f}, seed={args.seed}"
    )

    # 真正的抓取采样，由 grasp_sampler.py 负责
    grasps = sample_grasps_from_mesh(
        mesh_path=mesh_path,
        n_samples=args.n,
        down_sample_voxel=args.voxel,
        table_z=args.table_z,
        seed=args.seed,
    )

    # 打包为 JSON
    data = pack_for_json(grasps, topk=args.topk)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {len(data)} grasps → {args.out}")


if __name__ == "__main__":
    main()

