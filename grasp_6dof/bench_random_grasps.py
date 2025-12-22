# grasp_6dof/bench_random_grasps.py
import pybullet as p
import pybullet_data
import numpy as np
import csv, os, time, argparse, random

# ----------------- 参数与范围 -----------------
X_MIN, X_MAX = 0.35, 0.50
Y_MIN, Y_MAX = -0.12, 0.12
SCALE_MIN, SCALE_MAX = 0.06, 0.12       # 立方体缩放范围（更小更易抓）
GRASP_OFFSET = 0.02                      # 距桌面的目标抓取高度
FINGER_FORCE = 900
FINGER_CLOSE_ITERS = 1200
DT = 1.0/480.0

def detect_table_top_z(table_id):
    """射线测量：在桌面中心上方向下打一条射线，返回命中 z。"""
    from_pt = [0.5, 0.0, 1.5]
    to_pt   = [0.5, 0.0, -1.0]
    hit = p.rayTest(from_pt, to_pt)[0]
    if hit[0] == table_id:
        return hit[3][2]
    # 兜底：用 AABB 的上表面
    aabb = p.getAABB(table_id)
    return aabb[1][2]

def setup_world(vis):
    cid = p.connect(p.GUI if vis else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.8)
    p.setPhysicsEngineParameter(numSolverIterations=200)
    p.setTimeStep(DT)

    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5,0,-0.63])
    table_z  = detect_table_top_z(table_id)
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # 手指 link index（Franka Panda 常见是 9 和 10）
    finger_ids = [9, 10]

    return cid, table_id, table_z, panda_id, finger_ids

def reset_arm(panda_id):
    quat_down = p.getQuaternionFromEuler([np.pi, 0, 0])
    home = [0, -0.5, 0, -1.7, 0, 1.3, 0.8]
    for i in range(7):
        p.resetJointState(panda_id, i, home[i])
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, home[i], force=600)
    for _ in range(240): p.stepSimulation()
    return quat_down, home

def ik_move(panda_id, ee_idx, pos, orn, home, steps=240, force=700):
    ll = [-2.8]*7; ul=[2.8]*7; jr=[5.6]*7
    js = p.calculateInverseKinematics(
        panda_id, ee_idx, pos, orn,
        lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=home,
        maxNumIterations=200, residualThreshold=1e-4
    )
    for i in range(7):
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, js[i], force=force)
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(DT)

def place_cube(table_z, xy, scale):
    half = 0.5*scale           # cube.urdf 的单位立方体，缩放后半高≈scale/2
    z = table_z + half + 0.002
    obj_id = p.loadURDF("cube.urdf", basePosition=[xy[0], xy[1], z], globalScaling=scale)
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, rollingFriction=0.05, restitution=0.0)
    # 静置
    for _ in range(240): p.stepSimulation()
    return obj_id, z

def do_one_grasp(table_z, panda_id, obj_id, finger_ids, grasp_offset=GRASP_OFFSET):
    ee_idx = 11  # panda_grasptarget
    quat_down, home = reset_arm(panda_id)

    # 物块当前位姿
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    grasp_z = table_z + grasp_offset

    # 1) 接近与下探
    above = [obj_pos[0], obj_pos[1], grasp_z + 0.25]
    ik_move(panda_id, ee_idx, above, quat_down, home, steps=360)
    target = [obj_pos[0], obj_pos[1], grasp_z]
    ik_move(panda_id, ee_idx, target, quat_down, home, steps=360)

    # 2) 合爪 + 接触检测
    contact_step = None
    contact_points = 0
    for it in range(FINGER_CLOSE_ITERS):
        for fid in finger_ids:
            p.setJointMotorControl2(
                panda_id, fid, p.POSITION_CONTROL,
                targetPosition=0.0, force=FINGER_FORCE
            )
        p.stepSimulation()
        time.sleep(DT)
        cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        if cps:
            fp = [c for c in cps if c[3] in finger_ids or c[2] in finger_ids]
            if fp:
                contact_step = it
                contact_points = len(fp)
                break

    # 3) 抬升目标
    start_pos, _ = p.getBasePositionAndOrientation(obj_id)
    lift_goal = [obj_pos[0], obj_pos[1], grasp_z + 0.25]
    ik_move(panda_id, ee_idx, lift_goal, quat_down, home, steps=240)

    # 3.1) 抬升后“保持阶段”测峰值：记录 0.5s 的高度轨迹
    peak_z = start_pos[2]
    hold_frames = 0
    HIGH_Z = table_z + 0.03   # 判定“离桌”的阈值
    REQUIRED_HOLD = int(0.2/DT)  # 至少保持 0.2s

    for _ in range(int(0.5/DT)):   # 0.5s 采样
        p.stepSimulation()
        time.sleep(DT)
        z = p.getBasePositionAndOrientation(obj_id)[0][2]
        if z > peak_z:
            peak_z = z
        if z > HIGH_Z:
            hold_frames += 1

    lift_dz = peak_z - start_pos[2]

    # 4) 松爪
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
    for _ in range(120):
        p.stepSimulation()

    # 5) 成功判定（更靠谱）
    success = (peak_z >= table_z + 0.06) and (hold_frames >= REQUIRED_HOLD)

    return {
        "success": bool(success),
        "contact_step": (int(contact_step) if contact_step is not None else -1),
        "contact_points": int(contact_points),
        "lift_dz": float(lift_dz),
        "peak_z": float(peak_z),
        "start_z": float(start_pos[2]),
        "hold_frames": int(hold_frames),
        "grasp_z": float(grasp_z)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--vis", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", type=str, default="grasp_6dof/out/grasp_bench.csv")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    cid, table_id, table_z, panda_id, finger_ids = setup_world(args.vis)
    print(f"[INFO] table_z = {table_z:.3f}, trials = {args.trials}, vis={args.vis}")

    rows = []
    succ = 0
    t0 = time.time()

    for t in range(1, args.trials+1):
        # 随机方块参数
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        scale = np.random.uniform(SCALE_MIN, SCALE_MAX)

        # 放置方块
        obj_id, nominal_z = place_cube(table_z, (x,y), scale)

        # 执行抓取
        m = do_one_grasp(table_z, panda_id, obj_id, finger_ids)

        row = {
            "trial": t,
            "x": round(x,3), "y": round(y,3), "scale": round(scale,3),
            "success": int(m["success"]),
            "contact_step": m["contact_step"],
            "contact_points": m["contact_points"],
            "lift_dz": round(m["lift_dz"], 4),
            "grasp_z": round(m["grasp_z"], 4)
        }
        rows.append(row); succ += m["success"]
        print(f"[{t}/{args.trials}] succ={m['success']}  pos=({x:.3f},{y:.3f}) scale={scale:.3f}  "
              f"contact_step={m['contact_step']} pts={m['contact_points']}  lift_dz={m['lift_dz']:.3f}")

        # 清理该 trial 的物体
        p.removeBody(obj_id)

    dur = time.time()-t0
    rate = succ/args.trials if args.trials>0 else 0.0
    mean_lift = np.mean([r["lift_dz"] for r in rows]) if rows else 0.0

    # 写 CSV
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames = [
            "trial","x","y","scale",
            "success","contact_step","contact_points",
            "lift_dz","peak_z","start_z","hold_frames","grasp_z"
        ])
        w.writeheader(); w.writerows(rows)

    print("\n===== SUMMARY =====")
    print(f"Trials         : {args.trials}")
    print(f"Successes      : {succ} ({rate*100:.1f}%)")
    print(f"Mean lift_dz   : {mean_lift:.3f} m")
    print(f"CSV saved      : {args.csv}")
    print(f"Elapsed        : {dur:.1f}s (vis={args.vis})")

    p.disconnect()

if __name__ == "__main__":
    main()

