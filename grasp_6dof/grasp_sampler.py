# -*- coding: utf-8 -*-
"""
Grasp Sampler: 从网格/点云生成 6-DoF 抓取候选，并给出几何评分
输出：[{position:[x,y,z], rpy:[r,p,y], score:float}, ...]
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import open3d as o3d

# -------------------------
# 数据结构
# -------------------------
@dataclass
class GraspPose:
    position: np.ndarray    # (3,)
    rpy: np.ndarray         # (3,) in radians
    score: float

# -------------------------
# 工具函数
# -------------------------
def euler_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """从旋转矩阵取 ZYX euler (roll, pitch, yaw)"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-8
    if not singular:
        roll  = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return float(roll), float(pitch), float(yaw)

def rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """轴角到旋转矩阵"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)
    return R

def orthonormal_basis_from_z(z_dir: np.ndarray) -> np.ndarray:
    """
    给定 z 方向（抓取法向朝 -z 或 z），构造 (x,y,z) 正交基。
    这里我们让手爪的 -z 朝向物体法向（即手爪向下“压”向物体）。
    """
    z = z_dir / (np.linalg.norm(z_dir) + 1e-12)
    helper = np.array([1,0,0]) if abs(z[0]) < 0.9 else np.array([0,1,0])
    x = np.cross(helper, z); x /= (np.linalg.norm(x)+1e-12)
    y = np.cross(z, x);      y /= (np.linalg.norm(y)+1e-12)
    R = np.stack([x,y,z], axis=1)  # 列向量为基
    return R

# -------------------------
# 打分（几何启发式）
# -------------------------
def score_grasp(
    p: np.ndarray, n: np.ndarray,
    R: np.ndarray,
    workspace: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]],
    table_z: float,
    clearance: float = 0.01,
    friction_angle_deg: float = 35.0,
) -> float:
    """
    纯几何快速评分（0~1），越高越好
    - 工作空间可达性（AABB 约束）
    - 与桌面/边界间隙
    - 法向一致性（摩擦圆锥近似）
    """
    (xr, yr, zr) = workspace

    # 1) 可达性 AABB
    if not (xr[0] <= p[0] <= xr[1] and yr[0] <= p[1] <= yr[1] and zr[0] <= p[2] <= zr[1]):
        return 0.0

    # 2) 与桌面间隙（抓取点离桌面至少 clearance）
    if (p[2] - table_z) < clearance:
        return 0.0

    # 3) 法向一致性：让 R 的 -z 方向与物体法向 n 对齐越好
    #   手爪 -z 朝向法向 n，夹角越小越好
    gripper_minus_z = -R[:,2]  # R 的第三列是 z_dir，这里取其反向
    cosang = np.clip(np.dot(gripper_minus_z, n) / (np.linalg.norm(n)+1e-12), -1, 1)
    ang = np.arccos(cosang)
    # 允许角度阈值（与摩擦锥相关）
    friction_rad = np.deg2rad(friction_angle_deg)
    # 线性衰减到 0
    align_score = max(0.0, 1.0 - (ang / (friction_rad + 1e-6)))

    # 4) 离边缘的“留量”（简单起见：用 p投影到平面后与包围盒边界的距离，越远越好）
    #   这里不计算真实几何边缘，给一个恒定值作为占位
    boundary_score = 0.7

    # 5) 汇总（可调权重）
    w_align, w_clear, w_bound = 0.5, 0.3, 0.2
    # clear_score：高于桌面越多越安全，> 2*clearance 后饱和
    clear_score = np.clip((p[2]-table_z)/(2*clearance), 0, 1)

    total = w_align*align_score + w_clear*clear_score + w_bound*boundary_score
    return float(np.clip(total, 0, 1))

# -------------------------
# 主流程：采样 + 评分
# -------------------------
def sample_grasps_from_mesh(
    mesh_path: str,
    n_samples: int = 500,
    down_sample_voxel: float = 0.003,
    table_z: float = 0.0,
    approach_offset: float = 0.02,
    yaw_bins: int = 8,
    pitch_jitter_deg: float = 7.5,
    roll_jitter_deg: float = 7.5,
    workspace: Tuple[Tuple[float,float], Tuple[float,float], Tuple[float,float]] = ((0.30,0.70), (-0.25,0.25), (0.02,0.40)),
    seed: Optional[int] = 19,
) -> List[GraspPose]:
    """
    从 mesh 采样点和法向，围绕法向形成一簇候选姿态并打分。
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) 载入 mesh（支持 .obj/.stl/.ply 等）
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    is_pc_like = (not mesh.has_triangles())

    # === 情况 A：正常的三角网格 ===
    if not is_pc_like:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # 从表面均匀采样点（Poisson disk）
        pcd = mesh.sample_points_poisson_disk(min(n_samples * 2, 5000))
        if down_sample_voxel is not None and down_sample_voxel > 0:
            pcd = pcd.voxel_down_sample(voxel_size=down_sample_voxel)
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )

    # === 情况 B：当前文件其实是点云（没有三角形） ===
    else:
        print(f"[WARN] '{mesh_path}' has no triangles, treat as point cloud.")
        # 用点云方式重新读取
        pcd = o3d.io.read_point_cloud(mesh_path)
        if down_sample_voxel is not None and down_sample_voxel > 0:
            pcd = pcd.voxel_down_sample(voxel_size=down_sample_voxel)
        # 给点云估计法向
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )

    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    if pts.shape[0] == 0:
        raise RuntimeError("点云为空，检查 mesh 路径或采样参数。")

    # 随机选 n_samples 个点
    sel_idx = np.random.choice(pts.shape[0], size=min(n_samples, pts.shape[0]), replace=False)

    grasps: List[GraspPose] = []
    for idx in sel_idx:
        p = pts[idx]
        n = nrm[idx]
        n = n / (np.linalg.norm(n) + 1e-12)

        # 基础朝向：让手爪 -z 与 n 对齐（即 z = -n）
        z_dir = -n
        R0 = orthonormal_basis_from_z(z_dir)

        # 沿着 z 轴离散多个 yaw，且在 roll/pitch 上加小扰动
        for b in range(yaw_bins):
            yaw = (2*np.pi) * (b / yaw_bins)
            R_yaw = rodrigues(z_dir, yaw)

            # 小扰动
            pj = np.deg2rad(np.random.uniform(-pitch_jitter_deg, pitch_jitter_deg))
            rj = np.deg2rad(np.random.uniform(-roll_jitter_deg,  roll_jitter_deg))
            # 扰动轴：在 x、y 上做小旋转
            R_pitch = rodrigues(R0[:,0], pj)
            R_roll  = rodrigues(R0[:,1], rj)

            R = R0 @ R_yaw @ R_pitch @ R_roll

            # 末端位置：在法向反方向（朝外）退一点作为 approach
            pos = p + (-n) * approach_offset

            # 评分并记录
            s = score_grasp(pos, n, R, workspace=workspace, table_z=table_z)
            if s <= 0.0:
                continue

            rpy = np.array(euler_from_R(R))  # roll,pitch,yaw
            grasps.append(GraspPose(position=pos, rpy=rpy, score=s))

    # 排序（score 从高到低）
    grasps.sort(key=lambda g: g.score, reverse=True)
    return grasps

def pack_for_json(grasps: List[GraspPose], topk: Optional[int] = None):
    out = []
    K = len(grasps) if topk is None else min(topk, len(grasps))
    for i in range(K):
        g = grasps[i]
        out.append({
            "position": [float(g.position[0]), float(g.position[1]), float(g.position[2])],
            "rpy":      [float(g.rpy[0]), float(g.rpy[1]), float(g.rpy[2])],
            "score":    float(g.score),
        })
    return out

