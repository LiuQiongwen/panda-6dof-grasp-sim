# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np
import argparse, os, json, time

def load_with_target_h(urdf_path, target_h, table_top_z, xy):
    """按目标高度统一缩放并放到桌面上"""
    probe_id = p.loadURDF(urdf_path, basePosition=[0,0,1.0], globalScaling=1.0)
    a0 = p.getAABB(probe_id, -1)
    h0 = max(1e-6, a0[1][2] - a0[0][2])
    p.removeBody(probe_id)

    sf = float(target_h) / h0
    base_z = table_top_z + 0.002 + 0.5 * float(target_h)
    obj_id = p.loadURDF(urdf_path, basePosition=[xy[0], xy[1], base_z], globalScaling=sf)
    return obj_id, sf

def random_on_cone(center_dir, angle_rad, n):
    """围绕中心方向的圆锥均匀采样方向"""
    center_dir = center_dir / (np.linalg.norm(center_dir) + 1e-9)
    # 找到任意与 center_dir 垂直的基向量
    if abs(center_dir[2]) < 0.9:
        tmp = np.array([0,0,1.0])
    else:
        tmp = np.array([1.0,0,0])
    u = np.cross(center_dir, tmp); u /= (np.linalg.norm(u) + 1e-9)
    v = np.cross(center_dir, u)

    ang = np.random.rand(n) * angle_rad
    az  = np.random.rand(n) * 2*np.pi
    dirs = (np.cos(ang)[:,None]*center_dir +
            np.sin(ang)[:,None]*(np.cos(az)[:,None]*u + np.sin(az)[:,None]*v))
    return dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)

def write_ply_xyz(path, pts):
    """最简单的 ASCII PLY 写出 (无需依赖 open3d)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x,y,z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=str)
    ap.add_argument("--out",  required=True, type=str)
    ap.add_argument("--target-h", type=float, default=0.08)
    ap.add_argument("--table-z", type=float, default=-0.004)
    ap.add_argument("--xy", type=float, nargs=2, default=[0.38, 0.0])
    ap.add_argument("--radius", type=float, default=0.55, help="扫描半径（摄像球半径）")
    ap.add_argument("--views", type=int, default=24, help="环向视角数量（>=12）")
    ap.add_argument("--rays-per-view", type=int, default=3000, help="每个视角的射线数")
    ap.add_argument("--cone-deg", type=float, default=20, help="每个视角锥角（度）")
    args = ap.parse_args()

    physics = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)

    # 可选：放个桌子（不强制）
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, args.table_z-0.63+0.002])

    obj_id, sf = load_with_target_h(args.urdf, args.target_h, args.table_z, args.xy)
    p.stepSimulation()

    # 目标中心 & 高度，用 AABB 估计
    aabb = p.getAABB(obj_id, -1)
    center = (np.array(aabb[0]) + np.array(aabb[1])) * 0.5
    size   = (np.array(aabb[1]) - np.array(aabb[0]))
    height = float(size[2])

    # 半球视角：方位角 [0, 2π)，固定仰角（从水平向上）
    views = max(8, int(args.views))
    azs   = np.linspace(0, 2*np.pi, views, endpoint=False)
    el    = np.deg2rad(35.0)  # 仰角 35°
    R     = float(args.radius)
    cone  = np.deg2rad(args.cone_deg)
    rays_per_view = max(500, int(args.rays_per_view))

    all_hits = []
    for k, az in enumerate(azs, 1):
        # 视角相机中心（绕物体一圈，略高于中线）
        cam = center + np.array([R*np.cos(az), R*np.sin(az), max(0.12, 0.35*height)])
        center_dir = center - cam
        center_dir = center_dir / (np.linalg.norm(center_dir)+1e-9)

        dirs = random_on_cone(center_dir, cone, rays_per_view)
        FROM = cam[None, :].repeat(rays_per_view, axis=0)
        TO   = FROM + 2.0 * dirs  # 2m 的投射长度，够用了

        # 批量射线（分块以免太大）
        batch = 4096
        hpts = []
        for i in range(0, rays_per_view, batch):
            sub_from = FROM[i:i+batch]
            sub_to   = TO[i:i+batch]
            res = p.rayTestBatch(sub_from.tolist(), sub_to.tolist(), reportHitNumber=0)
            for r in res:
                # r = (hitObjectUid, linkIndex, hitFraction, hitPosition, hitNormal)
                if r[0] == obj_id:
                    hpts.append(r[3])
        print(f"[{k}/{views}] hits={len(hpts)}")
        all_hits.extend(hpts)

    pts = np.array(all_hits, dtype=np.float32)
    if pts.shape[0] == 0:
        p.disconnect()
        raise RuntimeError("没有采到点（raycast）。检查 URDF 路径、半径/视角是否覆盖到物体。")

    # 去重/下采样（可选）
    # 1) 简单体素格下采样
    voxel = 0.002  # 2mm
    if voxel > 0:
        q = np.floor(pts / voxel)
        _, idx = np.unique(q, axis=0, return_index=True)
        pts = pts[idx]

    write_ply_xyz(args.out, pts)
    print(f"[OK] write {pts.shape[0]} points → {args.out}")
    p.disconnect()

if __name__ == "__main__":
    main()

