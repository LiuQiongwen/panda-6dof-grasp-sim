# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np
import json
import time
import argparse
import os
import random
from datetime import datetime

# ---------------------------- Utils ----------------------------

def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

def save_env_snapshot(out_dir="grasp_6dof/out"):
    os.makedirs(out_dir, exist_ok=True)
    snap = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pybullet_build": p.getAPIVersion(),
    }
    path = os.path.join(out_dir, f"env_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Environment snapshot saved → {path}")

def get_table_top_z(table_id: int) -> float:
    top_z = -1e9
    for ji in range(-1, p.getNumJoints(table_id)):  # -1: base
        aabb = p.getAABB(table_id, ji)
        if aabb:
            top_z = max(top_z, aabb[1][2])
    return top_z

# ------------------------- Grasp Routine -----------------------

def grasp_with_panda(
    obj_id,
    grasp_pose,
    panda_id,
    end_effector_index=8,
    finger_ids=None,
    table_top_z=0.0,
    init_base_z=None,
    open_width_m=0.04, 
    descent_step=0.002,
    descend_clear=0.020,
    vel_close=0.25,
    pos_close=900,
    squeeze=0.35,
    step_fn=None,
    ik_iters=400,
    ik_attempts=5,
    joint_force=900.0,
):
    """
    稳定 + 高速：渐降 -> 速度找物 -> 位置夹紧 -> 轻抬二次挤压 -> 抬升判定
    使用 IK 做 7DoF 手臂控制。
    """
    # ---------- 常量 ----------
    MIN_CLEAR = 0.005
    DESCENT_STEP = float(descent_step)
    VEL_CLOSE_TIME = float(vel_close)
    POS_CLOSE_FORCE = float(pos_close)
    SQUEEZE_EXTRA_TIME = float(squeeze)
    LIFT_UP = 0.25
    LIFT_SUCCESS_DZ = 0.05
    JOINT_FORCE = float(joint_force)
    IK_ITERS = int(ik_iters)
    IK_ATTEMPTS = max(1, int(ik_attempts))

    # ---------- step 函数 ----------
    if step_fn is None:
        def step_fn(n):
            for _ in range(int(n)):
                p.stepSimulation()
                time.sleep(1/480.0)

    # ---------- 物理参数 ----------
    p.setPhysicsEngineParameter(numSolverIterations=200)
    p.setTimeStep(1.0/480.0)

    home = [0, -0.5, 0, -1.7, 0, 1.3, 0.8]
    for i in range(7):
        p.resetJointState(panda_id, i, home[i])
        p.setJointMotorControl2(panda_id, i, p.POSITION_CONTROL, home[i], force=500)
    step_fn(int(0.25 * 480))

    # 摩擦
    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    for fid in finger_ids:
        p.changeDynamics(panda_id, fid,
                         lateralFriction=2.5,
                         rollingFriction=0.05,
                         spinningFriction=0.02)
    
    # ---------- 使用 grasp 的位姿 ----------
    # ---- 读取 grasp 并强制“末端朝下 + 仅保留 yaw” ----
    target_pos = np.array(
        grasp_pose.get("position", [0.38, 0.0, table_top_z + 0.12]), dtype=float
    )

    # 兼容两种写法：rpy 或 yaw 字段；默认 yaw=0
    if "rpy" in grasp_pose and isinstance(grasp_pose["rpy"], (list, tuple)) and len(grasp_pose["rpy"]) >= 3:
        yaw = float(grasp_pose["rpy"][2])
    else:
        yaw = float(grasp_pose.get("yaw", 0.0))

    # ⭐ 关键：强制朝下（pitch=π），只用 yaw 做绕Z旋转
    quat_target = p.getQuaternionFromEuler([np.pi, 0.0, yaw])

    # 由四元数取末端 -Z 方向（渐降用）
    R = np.array(p.getMatrixFromQuaternion(quat_target)).reshape(3, 3)
    minus_z = -R[:, 2]
    minus_z = minus_z / (np.linalg.norm(minus_z) + 1e-8)

    # 可达性小兜底：把 XY 夹到手臂工作区
    target_pos[0] = float(np.clip(target_pos[0], 0.30, 0.60))
    target_pos[1] = float(np.clip(target_pos[1], -0.25, 0.25))


    # 目标与高度（AABB 顶面更鲁棒）
    obj_pos0, _ = p.getBasePositionAndOrientation(obj_id)
    obj_aabb = p.getAABB(obj_id, -1)
    cube_top_z = obj_aabb[1][2]
    cube_bot_z = obj_aabb[0][2]
    cube_mid_z = 0.5 * (cube_bot_z + cube_top_z)

    # --- 判定是否近似“球/圆”类：三维尺寸相差很小 ---
    dims = np.array(obj_aabb[1]) - np.array(obj_aabb[0])   # [dx, dy, dz]
    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
    ratio_xy = abs(dx - dy) / max(1e-6, max(dx, dy))
    ratio_xz = abs(dx - dz) / max(1e-6, max(dx, dz))
    ratio_yz = abs(dy - dz) / max(1e-6, max(dy, dz))
    nearly_sphere = (ratio_xy < 0.15 and ratio_xz < 0.15 and ratio_yz < 0.15)

    cx = float(np.clip(obj_pos0[0], 0.32, 0.55))
    cy = float(np.clip(obj_pos0[1], -0.18, 0.18))
    init_z = float(init_base_z) if init_base_z is not None else float(obj_pos0[2])

    approach_z   = cube_top_z + 0.10
    descend_from = cube_top_z + float(descend_clear)

    # ⭐ 关键：球类允许探到“赤道以下”一些（更容易卡住）
    if nearly_sphere:
        # 允许到中线以下 ~0.35*高度，但不撞桌
        below_mid = cube_mid_z - 0.35 * dz
        min_target_z = max(below_mid, table_top_z + MIN_CLEAR)
    else:
        min_target_z = max(cube_mid_z, table_top_z + MIN_CLEAR)
    
    print(f"[DEBUG] shape={'sphere_like' if nearly_sphere else 'other'} "
          f"table_z={table_top_z:.3f}, top={cube_top_z:.3f}, mid={cube_mid_z:.3f}, "
          f"dz={dz:.3f}, approach={approach_z:.3f}, from={descend_from:.3f}, "
          f"min_target_z={min_target_z:.3f}")

    # ---------- IK 移动封装（默认用 grasp 朝向；不再引用不存在的 quat_down） ----------
    def move_to(pos, orn=None, steps=240):
        if orn is None:
            orn = quat_target
        target = np.array(pos, dtype=float)

        joints_full = p.calculateInverseKinematics(
            bodyUniqueId=panda_id,
            endEffectorLinkIndex=end_effector_index,
            targetPosition=target.tolist(),
            targetOrientation=orn,
            solver=p.IK_DLS,
            maxNumIterations=int(IK_ITERS),
            residualThreshold=1e-4
        )
        for j in range(7):
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, joints_full[j], force=JOINT_FORCE)

        step_fn(steps)
        ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
        cur = np.array(ee[0])
        err = float(np.linalg.norm(cur - target))
        near_table = target[2] < (table_top_z + 0.10)
        tol = 0.008 if not near_table else 0.012
        if err > tol:
            joints_full = p.calculateInverseKinematics(
                panda_id, end_effector_index, target.tolist(), orn,
                solver=p.IK_DLS, maxNumIterations=max(int(IK_ITERS*2), 400),
                residualThreshold=5e-5
            )
            for j in range(7):
                p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, joints_full[j], force=JOINT_FORCE)
            step_fn(max(steps//2, 120))
            ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
            cur = np.array(ee[0])
            err = float(np.linalg.norm(cur - target))
            if err >= 0.02:
                print(f"[WARN] move_to 未完全到位，残差={err:.3f} m@{target.tolist()}")

    # ---------- 手指先张到目标开口 ----------
    half = max(0.0, float(open_width_m) * 0.5)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=half, force=200)
    step_fn(int(0.20 * 480))

    # ---------- 先到“抓取点上方”（沿末端 -Z 提前 descend_clear），再到抓取点 ----------
    pre_pos = target_pos + minus_z * float(descend_clear)
    move_to(pre_pos,  orn=quat_target, steps=200)
    move_to(target_pos, orn=quat_target, steps=140)

    # ---------- 渐降找接触（强制步进 + 正确的 linkIndex 检测） ----------
    def fingers_contact_object() -> bool:
        # getContactPoints 返回: (..., bodyA, bodyB, linkIndexA, linkIndexB, ...)
        # 我们传的是 bodyA=panda(hand), bodyB=obj；手指的 linkIndex 在 c[3]
        cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        return any((cp[3] in finger_ids) for cp in cps)

    contact = False
    z = float(target_pos[2])
    z_eps = 1e-4
    step = max(0.0015, DESCENT_STEP)  # 球面更细一些
    print(f"[DEBUG] descend z: start={z:.3f} -> min_target_z={min_target_z:.3f} (step={step:.4f})")
    while z > (min_target_z - z_eps):
        z -= step
        move_to([target_pos[0], target_pos[1], z], orn=quat_target, steps=int(60))
        hit = fingers_contact_object()
        print(f"[TRACE] z={z:.3f}, contact={hit}")
        if hit:
            contact = True
            break

    # ---------- 合爪：速度找物 -> 位置夹紧到 0.0 ----------
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.VELOCITY_CONTROL, targetVelocity=-0.2, force=40)
    step_fn(int(VEL_CLOSE_TIME * 480))
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=POS_CLOSE_FORCE)
    step_fn(int(0.20 * 480))

    if not contact:
        # 合爪后再检查一次（用正确的 link 索引）
        contact = fingers_contact_object()

    # ---------- 轻抬 + 二次挤压 ----------
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = np.array(ee[0])
    move_to([ee_pos[0], ee_pos[1], ee_pos[2] + 0.015], orn=quat_target, steps=120)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=POS_CLOSE_FORCE)
    step_fn(int(SQUEEZE_EXTRA_TIME * 480))
    # 对球体尤其有用：在安全高度做一次微探，帮助手指“卡住”物体
    probe_z = max(min_target_z, table_top_z + MIN_CLEAR + 0.001)
    move_to([target_pos[0], target_pos[1], probe_z], orn=quat_target, steps=90)    

    # ---------- 抬升并判定 ----------
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = np.array(ee[0])
    move_to([ee_pos[0], ee_pos[1], ee_pos[2] + LIFT_UP], orn=quat_target, steps=360)

    now_z = p.getBasePositionAndOrientation(obj_id)[0][2]
    lifted = (now_z - (init_base_z if init_base_z is not None else obj_pos0[2])) > LIFT_SUCCESS_DZ
    print(f"[DEBUG] contact={contact}, Δz={now_z - (init_base_z if init_base_z is not None else obj_pos0[2]):.3f}, success={lifted}")

    # ---------- 松手 ----------
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
    step_fn(int(0.20 * 480))

    return lifted


def load_obj_with_target_height(urdf_path: str, target_h: float, table_top_z: float, xy=(0.38, 0.0)):
    """按目标高度 target_h（米）自适应计算 globalScaling，任何 URDF 都能统一尺寸。"""
    # 先探测原始高度 h0
    probe_id = p.loadURDF(urdf_path, basePosition=[0,0,1.0], globalScaling=1.0)
    a0 = p.getAABB(probe_id, -1)
    h0 = max(1e-6, a0[1][2] - a0[0][2])
    p.removeBody(probe_id)

    sf = float(target_h) / h0
    base_z = table_top_z + 0.002 + 0.5 * float(target_h)
    obj_id = p.loadURDF(urdf_path,
                        basePosition=[xy[0], xy[1], base_z],
                        globalScaling=sf)
    return obj_id, sf, base_z, float(target_h)


# ------------------------------ Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="cube.urdf")
    parser.add_argument("--grasps", type=str, default="grasp_6dof/dataset/sample_grasps.json")
    parser.add_argument("--out", type=str, default="grasp_6dof/dataset/validated_grasps_panda.json")
    parser.add_argument("--vis", type=int, default=1)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--fast", action="store_true", help="关可视化/不sleep/缩短steps")
    parser.add_argument("--fast-scale", type=float, default=0.9, help="steps 缩放系数(0.6~0.95)")
    parser.add_argument("--cube-scale", type=float, default=0.08)
    parser.add_argument("--reset-each-trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)

    # IK/动力学可调参数
    parser.add_argument("--ee-index", type=int, default=11, help="8: hand, 11: gripper center")
    parser.add_argument("--ik-iters", type=int, default=400)
    parser.add_argument("--ik-attempts", type=int, default=5)
    parser.add_argument("--joint-force", type=float, default=900.0)
    parser.add_argument("--descent-step", type=float, default=0.002,
                        dest="descent_step", help="渐降步长(m)，默认 0.002")
    parser.add_argument("--descend-clear", type=float, default=0.020,
                        dest="descend_clear", help="从方块顶面上方多少米开始渐降，默认 0.020")
    parser.add_argument("--vel-close", type=float, default=0.25,
                        dest="vel_close", help="速度合爪阶段时长(s)，默认 0.25")
    parser.add_argument("--pos-close", type=float, default=900,
                        dest="pos_close", help="位置夹紧的力(牛)，默认 900")
    parser.add_argument("--squeeze", type=float, default=0.35,
                        dest="squeeze", help="二次挤压时长(s)，默认 0.35")
    parser.add_argument("--summary-csv", type=str, default="grasp_6dof/out/summary.csv",
                    help="将本次实验的配置与结果附加写入该 CSV")


    args = parser.parse_args()
    set_global_seed(args.seed)
    
    def append_summary_row(path, fields, values):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import csv
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(fields)
            w.writerow(values)
    
    # 连接仿真
    physicsClient = p.connect(p.GUI if (args.vis and not args.fast) else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    save_env_snapshot()

    # 步进函数：fast 模式关闭 sleep 且按 scale 减少步数
    def step(n):
        steps = int(max(1, n * (args.fast_scale if args.fast else 1.0)))
        if args.fast or not args.vis:
            for _ in range(steps):
                p.stepSimulation()
        else:
            for _ in range(steps):
                p.stepSimulation()
                time.sleep(1/480.0)

    # 桌面
    table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
    TABLE_TOP_Z = get_table_top_z(table_id)
    print(f"[INFO] Detected table top z = {TABLE_TOP_Z:.3f}")

    # 物体：静置在桌面
    CUBE_SCALE = float(args.cube_scale)
    CUBE_HALF_Z = 0.5 * CUBE_SCALE
    CUBE_XY = (0.38, 0.00)
    obj_z = TABLE_TOP_Z + CUBE_HALF_Z + 0.002
    obj_id, sf, obj_z, TARGET_H = load_obj_with_target_height(args.obj, args.cube_scale,    TABLE_TOP_Z, xy=CUBE_XY)
    print(f"[INFO] 对象放置：target_h={TARGET_H:.3f}m, auto_scale={sf:.3f}, z={obj_z:.3f}")

    p.changeDynamics(obj_id, -1, lateralFriction=1.6, restitution=0.0, rollingFriction=0.05)
    print(f"[INFO] 方块已放在桌面上 (x={CUBE_XY[0]:.2f}, y={CUBE_XY[1]:.2f}, z={obj_z:.3f}, scale={CUBE_SCALE}).")
    
    init_obj_pos, init_obj_orn = p.getBasePositionAndOrientation(obj_id)

    # 先稳定
    step(int(0.5 * 480))

    # 读取 grasps；为空则兜底
    def set_gripper_width(width_m, speed=0.3, max_force=120):
        # Panda: finger joints usually 9 (left) & 10 (right)
        half = max(0.0, width_m * 0.5)
        p.setJointMotorControl2(panda_id, 9,  p.POSITION_CONTROL, targetPosition=half,
                                force=max_force, maxVelocity=speed)
        p.setJointMotorControl2(panda_id, 10, p.POSITION_CONTROL, targetPosition=half,
                                force=max_force, maxVelocity=speed)
        for _ in range(60):  # ~1s @60Hz
            p.stepSimulation()
            
    try:
        with open(args.grasps, "r") as f:
            grasps = json.load(f)
            if not isinstance(grasps, list):
                grasps = []
    except Exception:
        grasps = []
    if args.topk is not None and len(grasps) > 0:
        if "score" in grasps[0]:
            grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)
        grasps = grasps[:args.topk]

    if len(grasps) == 0:
        cx, cy, cz = p.getBasePositionAndOrientation(obj_id)[0]
        z_above = cz + 0.12
        yaw_list = np.linspace(-np.pi, np.pi, 12, endpoint=False)
        grasps = [{
            "position": [float(cx), float(cy), float(z_above)],
            "rpy": [float(np.pi), 0.0, float(yaw)]
        } for yaw in yaw_list]
        print(f"[WARN] No grasps in JSON. Generated {len(grasps)} top-down fallback grasps.")
    print(f"[INFO] Loaded {len(grasps)} grasps for validation.")

    # Panda
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    finger_ids = [9, 10]
    END_EFFECTOR_INDEX = int(args.ee_index)

    # 统计
    validated, success_count = [], 0
    init_base_z = p.getBasePositionAndOrientation(obj_id)[0][2]

    for i, grasp in enumerate(grasps):
        # 每轮重置：臂+手指
        if args.reset_each_trial:
            for j in range(7):
                p.resetJointState(panda_id, j, 0.0)
            for fid in finger_ids:
                # 手指摩擦再提一档（避免小球打滑）
                p.changeDynamics(panda_id, fid, lateralFriction=2.5, rollingFriction=0.05, spinningFriction=0.02)
                p.resetJointState(panda_id, fid, 0.04)

            # ⭐ 复位方块到初始位姿（很关键）
            p.resetBasePositionAndOrientation(obj_id, init_obj_pos, init_obj_orn)
            step(int(0.15 * 480))  # 稍微稳定一下

        # ⭐ 每轮都重新读取“抬升成功”的基准高度
        init_base_z = p.getBasePositionAndOrientation(obj_id)[0][2]
        
        # --- 以 AABB 估算该物体在夹爪方向所需最小开口（再加 1cm 安全余量）---
        aabb = p.getAABB(obj_id, -1)
        need_open = float(aabb[1][0] - aabb[0][0]) + 0.010     # 物体X向宽度 + 1cm
        # 来自 grasp 的建议宽度（+4mm 装配余量）
        suggest_open = float(grasp.get("width", 0.04)) + 0.004
        # 取更大的那个，但不超过 Panda 机械极限（≈0.08m）
        target_open = min(max(need_open, suggest_open), 0.080)

        
        ok = grasp_with_panda(
            obj_id, grasp, panda_id,
            end_effector_index=END_EFFECTOR_INDEX,
            finger_ids=finger_ids,
            table_top_z=TABLE_TOP_Z,
            init_base_z=init_base_z,
            open_width_m=target_open,           # ← 传进去
            descent_step=args.descent_step,
            descend_clear=args.descend_clear,
            vel_close=args.vel_close,
            pos_close=args.pos_close,
            squeeze=args.squeeze,
        )
        out_g = dict(grasp); out_g["success"] = bool(ok)
        validated.append(out_g)
        if ok: success_count += 1
        print(f"[{i+1}/{len(grasps)}] Grasp success = {ok}")

    # 结果
    total = max(1, len(grasps))
    print(f"[INFO] Success rate = {success_count / total:.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved validated grasps → {args.out}")

    fields = [
        "time","obj","cube_scale","topk","seed",
        "ee_index","ik_iters","ik_attempts","joint_force",
        "descent_step","descend_clear","vel_close","pos_close","squeeze",
        "fast","fast_scale",
        "n_trials","success_count","success_rate"
    ]
    values = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.obj, args.cube_scale, args.topk, args.seed,
        args.ee_index, args.ik_iters, args.ik_attempts, args.joint_force,
        args.descent_step, args.descend_clear, args.vel_close, args.pos_close, args.squeeze,
        int(args.fast), args.fast_scale,
        len(grasps), success_count, round(success_count/max(1,len(grasps)), 4)
    ]
    append_summary_row(args.summary_csv, fields, values)
    print(f"[INFO] Appended summary → {args.summary_csv}")

    if args.vis and not args.fast:
        input("Press Enter to exit simulation...")
    p.disconnect()

if __name__ == "__main__":
    main()

