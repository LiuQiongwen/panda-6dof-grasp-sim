"""
Minimal 6-DoF grasp generator (Open3D + geometric scoring)
Outputs JSON with fields compatible to validate_grasps_panda.py:
[
  {"position":[x,y,z], "rpy":[roll,pitch,yaw], "width":float, "score":float, "meta":{...}},
  ...
]
"""
import json, os, math, argparse, random
import numpy as np

try:
    import open3d as o3d
except:
    raise RuntimeError("Please: pip install open3d")

# -------------------------- Utils --------------------------

def set_seed(seed: int | None):
    if seed is None: return
    random.seed(seed); np.random.seed(seed)

def load_pointcloud(path: str):
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".ply", ".pcd", ".xyz"]:
        pc = o3d.io.read_point_cloud(path)
        if len(pc.points) == 0:
            raise ValueError(f"Empty point cloud: {path}")
        return pc
    elif ext == ".npz":
        arr = np.load(path)
        pts = arr["points"] if "points" in arr else arr["xyz"]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))
        if "normals" in arr:
            pc.normals = o3d.utility.Vector3dVector(np.asarray(arr["normals"], float))
        return pc
    else:
        raise ValueError(f"Unsupported point cloud: {path}")

def ensure_normals(pc, radius=0.02, max_nn=40):
    if not pc.has_normals():
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pc.orient_normals_consistent_tangent_plane(50)
    return pc

def rpy_from_R(R):
    """Intrinsic XYZ (roll, pitch, yaw)."""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    else:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    return [float(roll), float(pitch), float(yaw)]

def build_R_from_normal_and_yaw(n_out: np.ndarray, yaw_rad: float):
    """
    We set gripper +Z to align with outward surface normal (n_out).
    In validator, approach_dir == -R[:,2], i.e., approach along -n_out (into surface).
    Then rotate around +Z (n_out) by yaw.
    """
    z = n_out / (np.linalg.norm(n_out) + 1e-9)
    # pick any vector not colinear with z
    tmp = np.array([1,0,0], float) if abs(z[0]) < 0.9 else np.array([0,1,0], float)
    x0 = np.cross(tmp, z); x0 /= (np.linalg.norm(x0) + 1e-9)  # one tangent
    y0 = np.cross(z, x0)  # the other tangent
    # yaw around z
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    x =  c * x0 + s * y0
    y = -s * x0 + c * y0
    R = np.stack([x, y, z], axis=0).T  # columns are axes
    return R

def pca_cov_eigs(pts: np.ndarray):
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 0.0])
    cov = np.cov(pts.T)
    w,_ = np.linalg.eigh(cov)
    return np.sort(w)[::-1]  # λ1 >= λ2 >= λ3

def radius_from_eigs(lam1, lam2):
    # surface patch: lam3 << lam1≈lam2; we take radial extent ~2*sqrt(lam2)
    # keep positive
    lam2 = max(lam2, 1e-12)
    return 2.0 * math.sqrt(lam2)

# --------------------- Scoring (MVP) ------------------------

def score_grasp(pc_tree, pts_np, pos, R, table_z: float, gripper_max=0.080):
    """
    Simple geometric scoring:
    - stability: prefer larger neighborhood support & flatter patch (λ2 - λ3)
    - collision/table penalty: if pre-approach goes below table
    - width penalty: needed_width beyond gripper limit
    """
    # neighborhood
    idx = pc_tree.search_radius_vector_3d(pos, 0.02)[1]  # 2cm radius
    neigh = pts_np[idx] if len(idx) > 5 else pts_np[:0]
    if len(neigh) < 5:
        return -1e3, 0.04  # poor

    lam = pca_cov_eigs(neigh)  # λ1≥λ2≥λ3
    lam1, lam2, lam3 = lam[0], lam[1], lam[2]
    # local "radius" as a proxy of needed opening along finger closing
    local_radius = radius_from_eigs(lam1, lam2)  # ~diameter in tangent plane
    needed_width = min(max(0.02, local_radius + 0.008), 0.10)  # +8mm margin

    # stability: large support + flatness (λ2 - λ3), normalized
    flat = max(0.0, lam2 - lam3)
    support = min(1.0, len(neigh)/150.0)
    stability = (0.6*support + 0.4*min(1.0, flat / (lam1 + 1e-9)))  # [0,1] ish

    # table penalty: pre-approach point 1cm above current pos along +Z of gripper?
    # we simulate pre_pos = pos + (+Z)*0.02 (go up 2cm along +Z then descend)
    pre_pos = pos + R[:,2]*0.02
    table_pen = 0.0 if pre_pos[2] > (table_z + 0.015) else ( (table_z + 0.015 - pre_pos[2]) * 50.0 )

    # width penalty if beyond limit
    width_pen = max(0.0, needed_width - gripper_max) * 200.0

    # center prior: prefer x in [0.30,0.60], y in [-0.25,0.25]
    cx, cy = pos[0], pos[1]
    reach_pen = 0.0
    if not (0.28 <= cx <= 0.62): reach_pen += 10.0 * (abs(cx-0.45))
    if not (-0.27 <= cy <= 0.27): reach_pen += 10.0 * (abs(cy-0.0))

    score = 2.0*stability - 1.0*(table_pen>0) - 0.5*table_pen - 0.5*width_pen - 0.2*reach_pen
    return float(score), float(needed_width)

# ------------------ Sampling (normals + yaw) -----------------

def sample_grasps_from_normals(pc, K=64, yaw_bins=8, offset_mm=(2, 8), table_z=-0.004, seed=None):
    """
    For each sampled surface point:
      - take outward normal n_out
      - set gripper +Z = n_out (so approach_dir == -n_out)
      - apply yaw around +Z
      - place TCP at pos = pt + n_out * offset
    """
    set_seed(seed)
    pc = ensure_normals(pc)
    pts = np.asarray(pc.points, float)
    nms = np.asarray(pc.normals, float)
    N = pts.shape[0]
    if N < 20: 
        print(f"[WARN] Point cloud very small (N={N}), results may be poor.")

    # make KD-tree once
    pc_tree = o3d.geometry.KDTreeFlann(pc)

    # orient normals to point OUTWARD w.r.t. centroid
    ctr = pts.mean(axis=0)
    to_out = pts - ctr
    flip = np.sum(to_out * nms, axis=1) < 0
    nms[flip] *= -1.0

    # subsample candidate indices
    cand = np.random.choice(N, size=min(2000, N), replace=False)
    offsets = np.linspace(offset_mm[0]/1000.0, offset_mm[1]/1000.0, 3)  # 2mm~8mm
    yaws = np.linspace(-math.pi, math.pi, yaw_bins, endpoint=False)

    results = []
    for idx in cand:
        p0 = pts[idx]; n_out = nms[idx]
        # reject points too close to table (avoid grazing)
        if p0[2] < table_z + 0.01: 
            continue
        for yaw in yaws:
            R = build_R_from_normal_and_yaw(n_out, yaw)
            for off in offsets:
                pos = p0 + n_out * off  # TCP slightly outside surface
                # basic sanity: don't start below table
                if pos[2] < table_z + 0.008:
                    continue
                score, need_w = score_grasp(pc_tree, pts, pos, R, table_z)
                results.append({
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "rpy": rpy_from_R(R),
                    "width": float(need_w),
                    "score": float(score),
                    "meta": {
                        "yaw": float(yaw),
                        "offset_m": float(off),
                        "normal": [float(n_out[0]), float(n_out[1]), float(n_out[2])]
                    }
                })

    if len(results) == 0:
        return []

    # NMS-like diversity on (pos,yaw): keep top by score with distance/yaw spreading
    results.sort(key=lambda g: g["score"], reverse=True)
    picked = []
    def too_close(a, b):
        pa = np.array(a["position"]); pb = np.array(b["position"])
        da = np.linalg.norm(pa - pb) < 0.008  # <8mm
        ya = a["meta"]["yaw"]; yb = b["meta"]["yaw"]
        dy = abs((ya - yb + math.pi)%(2*math.pi) - math.pi) < (math.pi/12)  # <15deg
        return da and dy

    for g in results:
        if len(picked) >= K: break
        if any(too_close(g, q) for q in picked): 
            continue
        picked.append(g)
    return picked

# ======= PCA 主轴 + 横向评分 =======
import numpy as np

def principal_axis_xy(points):
    pts = np.asarray(points)[:, :2]
    pts = pts - pts.mean(axis=0, keepdims=True)
    # SVD 第一主成分（桌面平面上的“长轴”）
    v = np.linalg.svd(pts, full_matrices=False)[2][0]
    v = v / (np.linalg.norm(v) + 1e-8)
    return v  # shape (2,)

def yaw_vec(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)])

def angle_perp_score(yaw, axis_xy):
    v = yaw_vec(yaw)
    cosang = np.clip(np.dot(v, axis_xy) / (np.linalg.norm(v)*np.linalg.norm(axis_xy)+1e-8), -1, 1)
    # 期望与主轴垂直（≈90°），用“离 90° 越近越好”打分
    ang = np.arccos(cosang)
    return 1.0 - abs(ang - np.pi/2) / (np.pi/2)  # ∈[0,1]

def mid_height_score(z, zmin, zmax):
    mid = 0.5*(zmin+zmax)
    span = max(1e-6, 0.5*(zmax-zmin))
    return max(0.0, 1.0 - abs(z - mid)/span)  # 越接近中腰越高



# --------------------------- I/O -----------------------------

def save_grasps(grs, out_json):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(grs, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(grs)} grasps → {out_json}")

# --------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc", type=str, required=True, help="Point cloud path (.ply/.pcd/.npz)")
    parser.add_argument("--out", type=str, required=True, help="Output JSON")
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--yaw-bins", type=int, default=12)
    parser.add_argument("--offset-mm", type=float, nargs=2, default=[2.0, 8.0])
    parser.add_argument("--table-z", type=float, default=-0.004)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    pc = load_pointcloud(args.pc)
    grasps = sample_grasps_from_normals(
        pc, K=args.topk, yaw_bins=args.yaw_bins,
        offset_mm=tuple(args.offset_mm), table_z=args.table_z, seed=args.seed
    )
    
    # ===== 可选：基于主轴 + 高度的几何重排序 =====
    pts = np.asarray(pc.points)
    axis_xy = principal_axis_xy(pts[:, :2])
    zvals = pts[:, 2]
    zmin, zmax = float(zvals.min()), float(zvals.max())

    scored = []
    for g in grasps:
        yaw = g.get("rpy", [np.pi, 0, 0])[-1] if "rpy" in g else g.get("yaw", 0.0)
        z   = float(g["position"][2])
        s_perp = angle_perp_score(yaw, axis_xy)
        s_mid  = mid_height_score(z, zmin, zmax)
        s = 0.7 * s_perp + 0.3 * s_mid
        g["score_geo"] = float(s)
        scored.append(g)

    kept = [g for g in scored if g["score_geo"] >= 0.55]
    if len(kept) < max(24, int(0.4 * len(scored))):
        kept = scored

    grasps = kept[:args.topk]
    # ===== 几何重排序结束 =====
    
    if len(grasps) == 0:
        print("[WARN] No grasps proposed.")
    save_grasps(grasps, args.out)

if __name__ == "__main__":
    main()

