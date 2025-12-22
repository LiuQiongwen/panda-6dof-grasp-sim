# vis_grasps.py
# -*- coding: utf-8 -*-
import os, json, time, math, argparse
import numpy as np
import pybullet as p
import pybullet_data

# ---------------- Camera (仅用于 headless 截图) ----------------
class CamCfg:
    def __init__(self, width=1280, height=720, fov=40.0, znear=0.2, zfar=6.0):
        self.width  = width
        self.height = height
        self.fov    = fov
        self.znear  = znear
        self.zfar   = zfar
        # 对准桌子中心偏方块的视角（和你之前截图接近）
        self.center = np.array([0.50, -0.35, 1.60], dtype=float)
        self.target = np.array([0.38,  0.00, 0.02], dtype=float)
        self.up     = np.array([0.00,  1.00, 0.00], dtype=float)

def snapshot_png(out_path, cam: CamCfg):
    view = p.computeViewMatrix(cam.center.tolist(), cam.target.tolist(), cam.up.tolist())
    
    proj = p.computeProjectionMatrixFOV(
        fov=cam.fov, aspect=float(cam.width)/cam.height,
        nearVal=0.05, farVal=3.0
    )

    w,h,rgba,depth,seg = p.getCameraImage(
        cam.width, cam.height, view, proj, renderer=p.ER_TINY_RENDERER
    )
    import imageio.v2 as imageio
    img = np.uint8(rgba)[:,:,:3]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.imwrite(out_path, img)
    print(f"[INFO] saved snapshot → {out_path}")

# ---------------- Math utils ----------------
def euler_to_R(rpy):
    r, pch, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(pch), np.sin(pch)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def rot_to_quat(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-8:
        roll  = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = 0.0
    else:
        roll  = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw   = math.atan2(R[1,0], R[0,0])
    return p.getQuaternionFromEuler([roll, pitch, yaw])

# ---------------- Geometry axes (可见于 headless/GUI) ----------------
def add_cylinder(p0, dir_world, length=0.06, radius=0.0025, rgba=(1,0,0,1)):
    d = np.array(dir_world, dtype=float)
    n = np.linalg.norm(d) + 1e-12
    d = d / n
    half = (length/2.0) * d
    center = np.array(p0, dtype=float) + half
    # 对齐 +Z 到 d
    z = np.array([0,0,1.0], dtype=float)
    v = np.cross(z, d); c = float(np.dot(z, d))
    if np.linalg.norm(v) < 1e-10:
        R = np.eye(3) if c > 0 else np.diag([1,-1,-1])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx@vx * (1.0/(1.0+c+1e-12))
    orn = rot_to_quat(R)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                      basePosition=center.tolist(), baseOrientation=orn)

def add_axes_geom(p0, R, L=0.06, rad=0.0025):
    """X红、Y绿、Z蓝；另外把 -Z（常作为抓取法线）加粗画出来"""
    x = (R @ np.array([1,0,0]))*L
    y = (R @ np.array([0,1,0]))*L
    z = (R @ np.array([0,0,1]))*L
    add_cylinder(p0,  x, length=L,   radius=rad,       rgba=(1,0,0,1))
    add_cylinder(p0,  y, length=L,   radius=rad,       rgba=(0,1,0,1))
    add_cylinder(p0,  z, length=L,   radius=rad,       rgba=(0,0,1,1))
    add_cylinder(p0, -z, length=L*1.1, radius=rad*1.6, rgba=(0.2,0.4,1.0,1))

# ---------------- Scene ----------------
TABLE_POS = [0.5, 0.0, -0.63]
TABLE_TOP_Z_APPROX = -0.004  # validate 中观测值
def cube_pos_from_scale(scale: float):
    return [0.38, 0.00, 0.002 + 0.5*scale]

def setup_scene(headless: bool, cube_scale: float, obj: str):
    if headless:
        cid = p.connect(p.DIRECT)
    else:
        cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 可视化阶段关闭重力，防止方块掉到桌下
    p.setGravity(0, 0, 0)
    # 桌子 & 物体
    table_id = p.loadURDF("table/table.urdf", basePosition=TABLE_POS, useFixedBase=True)
    cube_id  = p.loadURDF(obj, basePosition=cube_pos_from_scale(cube_scale),
                          globalScaling=cube_scale)
                          
    # --- 让方块更显眼：上色 + 画 AABB 线框 ---
    try:
        p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0.2, 0.2, 1])  # 染成红色
    except Exception:
        pass

    aabb_min, aabb_max = p.getAABB(cube_id, -1)
    # 画紫色 AABB 线框
    x0,y0,z0 = aabb_min; x1,y1,z1 = aabb_max
    purple = [0.6, 0.2, 1.0]
    W = 2.0  # 线宽
    corners = [
        (x0,y0,z0),(x0,y1,z0),(x1,y1,z0),(x1,y0,z0),
        (x0,y0,z1),(x0,y1,z1),(x1,y1,z1),(x1,y0,z1),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7)   # pillars
    ]
    for i,j in edges:
        p.addUserDebugLine(corners[i], corners[j], purple, lineWidth=W)
                      
    # 给 GUI 一点初始化时间
    if not headless:
        time.sleep(0.2)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2, cameraYaw=50, cameraPitch=-35,
            cameraTargetPosition=[0.38, 0.0, 0.0]
        )
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        except Exception:
            pass
    return cid, table_id, cube_id
    
    # 让方块显眼
    try:
        p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])
    except Exception:
        pass

    # 画 AABB 线框（紫色）
    aabb_min, aabb_max = p.getAABB(cube_id)
    def draw_aabb(a, b, color=[1,0,1], width=2):
        x0,y0,z0 = a; x1,y1,z1 = b
        corners = [
            [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
            [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]
        ]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i,j in edges:
            p.addUserDebugLine(corners[i], corners[j], color, width)
    draw_aabb(aabb_min, aabb_max)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True, help="path to grasps json")
    ap.add_argument("--cube-scale", type=float, default=0.08)
    ap.add_argument("--obj", type=str, default="cube.urdf")
    ap.add_argument("--show-n", type=int, default=20, help="最多显示前 N 个抓取")
    ap.add_argument("--headless", action="store_true", help="无 GUI 渲染并导出 PNG")
    ap.add_argument("--out-png", type=str, default="grasp_6dof/out/vis_grasps.png")
    ap.add_argument("--axes-length", type=float, default=0.06)
    ap.add_argument("--axes-radius", type=float, default=0.0025)
    args = ap.parse_args()

    headless = bool(args.headless)
    cid, table_id, cube_id = setup_scene(headless, args.cube_scale, args.obj)

    # 读抓取
    with open(args.grasps, "r", encoding="utf-8") as f:
        gs = json.load(f)
    N = min(args.show_n, len(gs))

    # 画坐标轴
    
    TABLE_Z = TABLE_TOP_Z_APPROX  # 约 -0.004
    SAFE_CLEAR = 0.020            # 相对桌面至少抬 2cm

    for g in gs[:N]:
        p0 = np.array(g["position"], dtype=float)

        # 如果 z 太低（<= 桌面+2cm），把它抬到方块上方（桌面 + 2mm + 半高 + 2cm）
        min_safe_z = TABLE_Z + 0.002 + 0.5*args.cube_scale + SAFE_CLEAR
        if p0[2] < min_safe_z:
            p0[2] = min_safe_z

        Rg = euler_to_R(g["rpy"])
        add_axes_geom(p0, Rg, L=args.axes_length, rad=args.axes_radius)

    cube_z = cube_pos_from_scale(args.cube_scale)[2] - TABLE_POS[2]  # 相对桌基座；仅用于打印
    print(f"Showing {N} grasps (table_top_z={TABLE_TOP_Z_APPROX:.3f}, cube_z={cube_z:.3f}).")

    if headless:
        cube_xyz = cube_pos_from_scale(args.cube_scale)
        # 相机位置：在方块“右后上方”一点点，且离得更近
        center = np.array(cube_xyz) + np.array([0.60, -0.60, 1.20], dtype=float)
        target = np.array(cube_xyz) + np.array([0.00,  0.00, 0.02], dtype=float)

        cam = CamCfg(width=1280, height=720, fov=40.0, znear=0.2, zfar=6.0)
        cam.center = center
        cam.target = target

        snapshot_png(args.out_png, cam)
        p.disconnect()
        return


    # GUI 循环
    try:
        while True:
            p.stepSimulation()      # 仅用于触发 GUI 刷新
            time.sleep(1.0/60.0)
    except KeyboardInterrupt:
        pass
    p.disconnect()

if __name__ == "__main__":
    main()

