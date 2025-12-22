# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np
import json
import time
import math
import argparse
import os
import pathlib
import random
from datetime import datetime
import csv

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

def load_obj_with_target_height(urdf_path: str, target_h: float, table_top_z: float, xy=(0.38, 0.0)):
    """
    按目标高度 target_h（米）自适应计算 globalScaling，让任意 URDF 都按同一高度落到桌面上。
    返回: (obj_id, scale_factor, base_z, target_h)
    """
    # 1) 量原始高度
    probe_id = p.loadURDF(urdf_path, basePosition=[0, 0, 1.0], globalScaling=1.0)
    aabb = p.getAABB(probe_id, -1)
    h0 = max(1e-6, aabb[1][2] - aabb[0][2])
    p.removeBody(probe_id)

    # 2) 计算缩放 & 放置到桌面
    sf = float(target_h) / h0
    base_z = float(table_top_z) + 0.002 + 0.5 * float(target_h)
    obj_id = p.loadURDF(
        urdf_path,
        basePosition=[float(xy[0]), float(xy[1]), base_z],
        globalScaling=sf,
    )
    return obj_id, sf, base_z, float(target_h)

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
    稳定 + 高速的抓取例程（适配 Panda + PyBullet）
    流程：对齐→张开→渐降触碰→速度找物→位置夹紧(留缝)→轻抬→零缝二次挤压→抬升判定
    返回：bool (是否抓取成功)
    """
    # --------- 默认参数/常量 ---------
    finger_ids = list(finger_ids or [9, 10])   # 用来打电机的关节（9/10）
    nJ = p.getNumJoints(panda_id)

    # 可能发生接触的 link：手指本体 + 指尖（11,13）+ 手掌(8)
    candidate_links = set(finger_ids + [8, 11, 13])
    contact_links = sorted([fid for fid in candidate_links if 0 <= fid < nJ])

    # 抬升高度 & 成功阈值
    LIFT_UP = 0.12
    LIFT_SUCCESS_DZ = 0.035


    # 统一步进（兼容外部 step_fn）
    def _step(n):
        n = int(max(1, n))
        if step_fn is None:
            for _ in range(n):
                p.stepSimulation()
                time.sleep(1.0 / 480.0)
        else:
            step_fn(n)
    
    # IK 位姿移动
    def move_to(pos, orn, steps=200):
        joints = p.calculateInverseKinematics(
            panda_id,
            endEffectorLinkIndex=end_effector_index,
            targetPosition=list(pos),
            targetOrientation=orn,
            solver=p.IK_DLS,
            maxNumIterations=int(ik_iters),
            residualThreshold=1e-4,
        )
        for j in range(7):
            p.setJointMotorControl2(
                panda_id, j, p.POSITION_CONTROL, targetPosition=joints[j], force=float(joint_force)
            )
        _step(steps)

    def _has_touch_or_near(panda_id, obj_id, link_ids, near_th=0.010, min_nforce=0.5):
        hits = []
        for fid in link_ids:
            cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id, linkIndexA=int(fid))
            strong = [cp for cp in cps if cp[9] >= float(min_nforce)]
            hits.append(len(strong) > 0)

        if hits.count(True) >= 1:
            return True

        for fid in link_ids:
            near = p.getClosestPoints(bodyA=panda_id, bodyB=obj_id,
                                    distance=float(near_th), linkIndexA=int(fid))
            if near:
                return True
        return False


    def log_axis_alignment(quat_target):
        R = np.array(p.getMatrixFromQuaternion(quat_target)).reshape(3,3)
        ee_down = +R[:,2]      # 若你的爪法线定义为 -z；若是 -x 请改为 -R[:,0]
        world_down = np.array([0,0,-1.0])
        cs = np.clip(ee_down.dot(world_down)/(np.linalg.norm(ee_down)+1e-9), -1, 1)
        ang = math.degrees(math.acos(cs))
        print(f"[DEBUG] ee-down vs world-down angle = {ang:.1f}°")
        return ang

    # ---------- 解析 grasp 位姿 ----------

    try:
        gx, gy, gz = map(float, grasp_pose.get("position", []))
    except Exception:
        gx, gy, gz = p.getBasePositionAndOrientation(obj_id)[0]

    # 直接使用数据集中提供的 rpy（通常类似 [π, 0, yaw]）
    rpy = grasp_pose.get("rpy", [math.pi, 0.0, 0.0])
    rpy = [float(r) for r in rpy[:3]] + [0.0] * (3 - len(rpy))
    quat_target = p.getQuaternionFromEuler(rpy)
    yaw = float(rpy[2])
    
    # 物体 AABB & 顶面/中面
    aabb_obj = p.getAABB(obj_id, -1)
    top_z = aabb_obj[1][2]
    mid_z = 0.5 * (aabb_obj[0][2] + aabb_obj[1][2])
    dx = aabb_obj[1][0] - aabb_obj[0][0]
    dy = aabb_obj[1][1] - aabb_obj[0][1]
    rx = abs(dx - dy) / max(1e-6, max(dx, dy))

    # 放宽圆柱/圆盘类判据：25% 以内视作“圆柱类”
    is_cyl_like_xy = (rx < 0.25)

    # 统一采用“中面+2.5mm”为最低目标，避免贴顶面/桌面
    PAD = 0.0025  # 2.5 mm
    min_target_z = max(float(table_top_z) + 0.010, mid_z - PAD)

    # 渐降起点（顶面上方）
    DESCENT_STEP = float(descent_step)
    start_z = top_z + float(descend_clear)
    safe_clear_from_table = max(0.010, float(descend_clear) * 0.2)

    # —— 圆柱类：仅略放低最低目标，不做 XY 外退（避免离物体太远）
    is_cyl_like_xy = (abs(dx - dy) / max(1e-6, max(dx, dy)) < 0.12)
    if is_cyl_like_xy:
        target_mid = mid_z
        PAD = 0.001
        min_target_z = max(float(table_top_z) + 0.008, target_mid - PAD)
    # —— 低摩擦找位阶段（在速度找物/留缝之前确保是低摩擦）
    for fid in finger_ids:
        p.changeDynamics(panda_id, fid, lateralFriction=2.2, rollingFriction=0.02,  spinningFriction=0.03)

    # === 先张开 ===
    open_width_m = min(0.080, max(0.022, float(open_width_m or 0.04)))
    half = max(0.0, min(open_width_m * 0.5, 0.040))  # Panda: 单指最大 ~40mm
    for fid in (finger_ids or [9, 10]):
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL,
                                targetPosition=half, force=120, maxVelocity=0.6)
    _step(int(0.20 * 480))

    # 先到 grasp XY，Z 在 start_z（物体上方），姿态朝下
    target_pos = [gx, gy, start_z]
    move_to(target_pos, orn=quat_target, steps=180)
    
    # 在首次就位后：
    ang = log_axis_alignment(quat_target)
    if ang > 12.0:
        # 先做小角度滚转修正再继续（例如 ±8° 内试两次）
        for dyaw in [math.radians(+6.0), math.radians(-6.0)]:
            q = p.getQuaternionFromEuler([math.pi, 0.0, yaw + dyaw])
            move_to([gx, gy, start_z], orn=q, steps=80)
            if log_axis_alignment(q) <= 12.0:
                quat_target = q
                yaw += float(dyaw)
                break  
    # ---------- 渐降直到“手指/手掌”触物 ----------
    finger_or_palm = set(contact_links)# 8 通常是 hand/palm link
    contact = False
    z = start_z

    print(
        f"[DEBUG] shape={'cyl' if is_cyl_like_xy else 'other'} table_z={table_top_z:.3f}, top={top_z:.3f}, mid={mid_z:.3f}, "
        f"dz={top_z-mid_z:.3f}, approach={start_z:.3f}, from={target_pos[2]:.3f}, min_target_z={min_target_z:.3f}"
    )
    print(f"[DEBUG] descend z: start={start_z:.3f} -> min_target_z={min_target_z:.3f} (step={DESCENT_STEP:.4f})")

    while z > min_target_z:
        z = max(min_target_z, z - DESCENT_STEP)
        move_to([gx, gy, z], orn=quat_target, steps=12)
        cps = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        if any(c[3] in finger_or_palm for c in cps):
            contact = True
            break

    if not contact:
        # 在最低点再探一次
        move_to([gx, gy, min_target_z], orn=quat_target, steps=90)
        contact = _has_touch_or_near(panda_id, obj_id, contact_links, near_th=0.006)

    # ---------- 速度找物 -> 位置夹紧(留缝) ----------
    # 先速度闭合一小段时间，推动物体进入指尖“槽”
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.VELOCITY_CONTROL, targetVelocity=-0.25, force=40)
    _step(int(float(vel_close) * 480))

    # 位置夹紧但保留 ~4mm 间隙，避免一次闭死
    gap = 0.004
    for fid in finger_ids:
        p.setJointMotorControl2(
            panda_id, fid, p.POSITION_CONTROL, targetPosition=0.5 * gap, force=float(pos_close)
        )
    _step(int(0.20 * 480))

    # 更新接触判定
    contact = _has_touch_or_near(panda_id, obj_id, contact_links, near_th=0.006)
       
    if not contact:
        # —— 螺旋微搜索：直接用固定半径，最大搜到 6cm ——
        offset_rs = [0.0, 0.015, 0.03, 0.045, 0.06]  # 单位: m
        angles = (0.0, 0.785398, 1.570796, 2.356194, 3.141593, 3.926991, 4.712389, 5.497787)
        yaw_deltas = [0.0, math.radians(+3), math.radians(-3), math.radians(+6), math.radians(-6)]

        found = False
        for r in offset_rs:
            for ang in angles:
                dxs, dys = r*math.cos(ang), r*math.sin(ang)
                for dyaw in yaw_deltas:
                    q = p.getQuaternionFromEuler([math.pi, 0.0, yaw + dyaw])
                    move_to([gx + dxs, gy + dys, min_target_z], orn=q, steps=70)

                    # 轻推（速度闭合）
                    for fid in finger_ids:
                        p.setJointMotorControl2(panda_id, fid, p.VELOCITY_CONTROL, targetVelocity=-0.20, force=32)
                    _step(int(0.10 * 480))

                    # 回到留缝（1~2mm）
                    for fid in finger_ids:
                        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL,
                                                targetPosition=0.002, force=float(pos_close))
                    _step(int(0.06 * 480))

                    if _has_touch_or_near(panda_id, obj_id, contact_links, near_th=0.010, min_nforce=0.2):
                        yaw += float(dyaw)
                        quat_target = q
                        gx, gy = gx + dxs, gy + dys
                        print(f"[DEBUG] microsearch hit at r={r:.3f}, ang={ang:.2f}, dyaw={math.degrees(dyaw):.1f}°")
                        found = True
                        break
                if found: break
            if found: break


        if not found:
            # 兜底：对齐几何中心再试一次
            cx = 0.5 * (aabb_obj[0][0] + aabb_obj[1][0])
            cy = 0.5 * (aabb_obj[0][1] + aabb_obj[1][1])
            move_to([cx, cy, min_target_z], orn=quat_target, steps=120)
            for fid in finger_ids:
                p.setJointMotorControl2(panda_id, fid, p.VELOCITY_CONTROL, targetVelocity=-0.22, force=40)
            _step(int(0.15 * 480))
            for fid in finger_ids:
                p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.002, force=float(pos_close))
            _step(int(0.08 * 480))
            if _has_touch_or_near(panda_id, obj_id, contact_links, near_th=0.010, min_nforce=0.2):
                gx, gy = cx, cy
                found = True
                print("[DEBUG] center-snap fallback hit")

        if not found:
            j9 = p.getJointState(panda_id, finger_ids[0])[0]
            j10 = p.getJointState(panda_id, finger_ids[1])[0]
            print(f"[DEBUG] fingers after close: j9={j9:.3f}, j10={j10:.3f}")
            print("[DEBUG] no contact -> early fail")
        else:
            contact = True
    # 命中后、seat 之前，切换到“锁定相”高摩擦
    for fid in finger_ids:
        p.changeDynamics(panda_id, fid,
                        lateralFriction=3.2, rollingFriction=0.035, spinningFriction=0.05,
                        contactStiffness=2400, contactDamping=70)

    # === 已确认 contact 后 ===

    # 1) 座位(下压 ~1.5mm) + 微摆动 + 再挤压（锁定相）
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = list(ee[0])
    seat_z = max(min_target_z + 0.001, ee_pos[2] - 0.0015)
    move_to([ee_pos[0], ee_pos[1], seat_z], orn=quat_target, steps=90)
    for deg in (+2.0, -2.0):
        q = p.getQuaternionFromEuler([math.pi, 0.0, yaw + math.radians(deg)])
        move_to([ee_pos[0], ee_pos[1], seat_z], orn=q, steps=70)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0,  force=float(pos_close)*1.10)
    _step(int(0.30 * 480))

    # 2) 预抬 2cm 做一次保持力刷新（抗滑）
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = list(ee[0])
    pre_lift = min(0.02, LIFT_UP * 0.2)
    move_to([ee_pos[0], ee_pos[1], ee_pos[2] + pre_lift], orn=quat_target, steps=120)
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0,  force=float(pos_close)*1.10)
    _step(int(0.20 * 480))

    # ---------- 抬升并判定 ----------
    if init_base_z is None:
        base_z = p.getBasePositionAndOrientation(obj_id)[0][2]
    else:
        base_z = float(init_base_z)

    # 倾斜抬升（先抬，再测 Δz）
    ee = p.getLinkState(panda_id, end_effector_index, computeForwardKinematics=True)
    ee_pos = list(ee[0])
    LIFT_UP_LOCAL = LIFT_UP
    pitch = math.radians(2.5)
    q_tilt = p.getQuaternionFromEuler([math.pi - pitch, 0.0, yaw])
    move_to([ee_pos[0], ee_pos[1], ee_pos[2] + LIFT_UP_LOCAL], orn=q_tilt, steps=300)

    # === 先计算 dz_lift ===
    now_z = p.getBasePositionAndOrientation(obj_id)[0][2]
    dz_lift = now_z - base_z

    # 物体高度 & 成功阈值（至少抬起 0.2H 或 2cm）
    aabb_now = p.getAABB(obj_id, -1)
    H = aabb_now[1][2] - aabb_now[0][2]
    need_dz = max(0.2 * H, 0.020)

    # 先给 still_touch 一个默认值，避免某些分支没赋值
    still_touch = False

    # 极端值保护（仿真数值爆掉的情况直接判失败）
    if (dz_lift < -0.05) or (abs(dz_lift) > 0.4):
        print(f"[WARN] abnormal Δz={dz_lift:.3f}, treat as fail.")
        lifted = False
    else:
        lifted = (dz_lift > need_dz)

        # 仍需“保持接触”才算真成功（避免抬起即滑脱）
        # 注意这里统一用 contact_links，而不是只看 finger_ids
        still_touch = _has_touch_or_near(
            panda_id, obj_id, contact_links,
            near_th=0.012, min_nforce=0.3
        )

        # 如果抬得特别高（比阈值多 3cm），可以放宽 still_touch 限制
        margin = 0.03  # 3cm
        if lifted and dz_lift > (need_dz + margin):
            # 抬得很多，哪怕接触力有点变小也算成功
            lifted = True
        else:
            lifted = bool(lifted and still_touch)

    print(f"[DEBUG] contact={contact}, Δz={dz_lift:.3f}, need>{need_dz:.3f},    still_touch={still_touch}, success={lifted}")

    # === 记录质量指标到 grasp_pose，后面 JSON 会一起写出去 ===
    try:
        metrics = grasp_pose.setdefault("_metrics", {})
        metrics["dz_lift"] = float(dz_lift)     # 实际抬起高度
        metrics["need_dz"] = float(need_dz)     # 判定阈值
        metrics["H"] = float(H)                 # 物体高度
        metrics["still_touch"] = bool(still_touch)
        metrics["contact_initial"] = bool(contact)
    except Exception:
        pass

    # ---------- 松手复位 ----------
    for fid in finger_ids:
        p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
    _step(int(0.20 * 480))
    return bool(lifted)

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
    parser.add_argument("--descent-step", type=float, default=0.0006,
                        dest="descent_step", help="渐降步长(m)，默认 0.0006")
    parser.add_argument("--descend-clear", type=float, default=0.020,
                        dest="descend_clear", help="从顶面上方多少米开始渐降，默认 0.020")
    parser.add_argument("--vel-close", type=float, default=0.8,
                        dest="vel_close", help="速度合爪阶段时长(s)，默认 0.8")
    parser.add_argument("--pos-close", type=float, default=700,
                        dest="pos_close", help="位置夹紧的力(牛)，默认 700")
    parser.add_argument("--squeeze", type=float, default=1.0,
                        dest="squeeze", help="二次挤压时长(s)，默认 0.35")
    parser.add_argument("--summary-csv", type=str, default="grasp_6dof/out/summary.csv",
                        help="将本次实验的配置与结果附加写入该 CSV")

    args = parser.parse_args()
    set_global_seed(args.seed)

    def append_summary_row(path, fields, values):
        os.makedirs(os.path.dirname(path), exist_ok=True)
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

    # —— 物理稳定性/确定性参数 ——
    p.setPhysicsEngineParameter(
        deterministicOverlappingPairs=1,
        collisionFilterMode=1,
        contactSlop=1e-3,
        enableConeFriction=1,
        solverResidualThreshold=1e-7,
        numSolverIterations=200,
        erp=0.2, contactERP=0.2, frictionERP=0.2,
        useSplitImpulse=1, splitImpulsePenetrationThreshold=-0.01,
        enableFileCaching=0,  # 纯内存更稳定
        allowedCcdPenetration=0.0005
    )
    p.setTimeStep(1.0/480.0)
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

    p.changeDynamics(table_id, -1,
                     lateralFriction=1.6, rollingFriction=0.05, spinningFriction=0.02,
                     restitution=0.0)

    # 物体：静置在桌面
    CUBE_SCALE = float(args.cube_scale)
    CUBE_XY = (0.38, 0.00)
    obj_id, sf, obj_z, TARGET_H = load_obj_with_target_height(args.obj, args.cube_scale, TABLE_TOP_Z, xy=CUBE_XY)
    print(f"[INFO] 对象放置：target_h={TARGET_H:.3f}m, auto_scale={sf:.3f}, z={obj_z:.3f}")

    p.changeDynamics(obj_id, -1,
        lateralFriction=3.0, rollingFriction=0.08, spinningFriction=0.15, restitution=0.0)
    print(f"[INFO] 方块已放在桌面上 (x={CUBE_XY[0]:.2f}, y={CUBE_XY[1]:.2f}, z={obj_z:.3f}, scale={CUBE_SCALE}).")

    init_obj_pos, init_obj_orn = p.getBasePositionAndOrientation(obj_id)
    print(f"[DEBUG] obj center (world): ({init_obj_pos[0]:.3f}, {init_obj_pos[1]:.3f}, {init_obj_pos[2]:.3f})")

    # 先稳定
    step(int(0.5 * 480))

    # 读取 grasps；为空则兜底
    try:
        with open(args.grasps, "r") as f:
            grasps = json.load(f)
            if not isinstance(grasps, list):
                grasps = []
    except Exception:
        grasps = []

    # 根据 score 做 topk
    if args.topk is not None and len(grasps) > 0:
        if "score" in grasps[0]:
            grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)
        grasps = grasps[:args.topk]

    # === 用 grasps 的平均 XY 对齐当前场景中的物体位置 ===
    if len(grasps) > 0:
        xs, ys = [], []
        for g in grasps:
            pos = g.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                try:
                    xs.append(float(pos[0]))
                    ys.append(float(pos[1]))
                except (TypeError, ValueError):
                    continue
        if xs and ys:
            gx_mean = float(np.mean(xs))
            gy_mean = float(np.mean(ys))
            cur_pos, cur_orn = p.getBasePositionAndOrientation(obj_id)
            new_pos = [gx_mean, gy_mean, cur_pos[2]]
            p.resetBasePositionAndOrientation(obj_id, new_pos, cur_orn)
            init_obj_pos = new_pos  # 后面 reset 物体要用这个
            print(f"[INFO] Realign object XY to grasp mean: x={gx_mean:.3f}, y={gy_mean:.3f}")
            # 顺便打印前几个 grasp，看一下大概位置
            for i, g in enumerate(grasps[:3]):
                print(f"[DEBUG] grasp[{i}] position={g.get('position')} rpy={g.get('rpy')}")
        else:
            print("[WARN] Grasps JSON has no valid 'position'; skip XY realign.")

    # 如果 JSON 根本没有 grasps，就自动生成一圈 top-down 抓取
    if len(grasps) == 0:
        # 用 AABB 计算真实几何中心，而不是 basePosition
        aabb = p.getAABB(obj_id)
        cx = 0.5 * (aabb[0][0] + aabb[1][0])
        cy = 0.5 * (aabb[0][1] + aabb[1][1])
        cz = 0.5 * (aabb[0][2] + aabb[1][2])

        z_above = cz + 0.12   # 从物体中心上方 12cm 开始
        yaw_list = np.linspace(-np.pi, np.pi, 12, endpoint=False)
        grasps = [{
            "position": [float(cx), float(cy), float(z_above)],
            "rpy": [float(np.pi), 0.0, float(yaw)]
        } for yaw in yaw_list]

        print(f"[WARN] No grasps in JSON. Generated {len(grasps)} top-down fallback grasps.")
        print(f"[DEBUG] fallback center from AABB: ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    print(f"[INFO] Loaded {len(grasps)} grasps for validation.")

    # ---- Panda 加载在使用前，避免未定义 finger_ids ----
    panda_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    finger_ids = [9, 10]

    # 左右手指齿轮约束（对称相向运动）
    cid = p.createConstraint(
        parentBodyUniqueId=panda_id, parentLinkIndex=9,
        childBodyUniqueId=panda_id,  childLinkIndex=10,
        jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
        parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
    )
    p.changeConstraint(cid, gearRatio=-1, maxForce=200, erp=0.5)

    END_EFFECTOR_INDEX = int(args.ee_index)

    # 物体与手指的动力学细化
    p.changeDynamics(obj_id, -1, lateralFriction=3.0, rollingFriction=0.08, spinningFriction=0.15, restitution=0.0)
    for fid in finger_ids:
        p.changeDynamics(
            panda_id, fid,
            lateralFriction=4.0,
            rollingFriction=0.03,
            spinningFriction=0.05
        )

    # 统计
    validated, success_count = [], 0

    for i, grasp in enumerate(grasps):
        # 每轮重置：臂+手指+物体
        if args.reset_each_trial:
            for fid in finger_ids:
                p.changeDynamics(panda_id, fid, lateralFriction=3.2, rollingFriction=0.035,
                                spinningFriction=0.05, contactStiffness=2400, contactDamping=70)
                p.resetJointState(panda_id, fid, 0.04)
            p.resetBasePositionAndOrientation(obj_id, init_obj_pos, init_obj_orn)
            step(int(0.15 * 480))

        init_base_z = p.getBasePositionAndOrientation(obj_id)[0][2]

        # --- 以 AABB 估算该物体在夹爪方向所需最小开口（+ 安全余量）---
        aabb = p.getAABB(obj_id, -1)
        dx = aabb[1][0] - aabb[0][0]
        dy = aabb[1][1] - aabb[0][1]
        need_open = float(max(dx, dy)) + 0.008  # 直径 + 8mm 余量（略加大）
        suggest_open = float(grasp.get("width", 0.04)) + 0.004
        open_width_m = min(max(need_open, suggest_open), 0.080)

        ok = grasp_with_panda(
            obj_id, grasp, panda_id,
            end_effector_index=END_EFFECTOR_INDEX,
            finger_ids=finger_ids,
            table_top_z=TABLE_TOP_Z,
            init_base_z=init_base_z,
            open_width_m=open_width_m,
            descent_step=args.descent_step,
            descend_clear=args.descend_clear,
            vel_close=args.vel_close,
            pos_close=args.pos_close,
            squeeze=args.squeeze,
            ik_iters=args.ik_iters,
            ik_attempts=args.ik_attempts,
            joint_force=args.joint_force,
            step_fn=step if (args.fast or not args.vis) else None,
        )

        out_g = dict(grasp)
        out_g["success"] = bool(ok)
        # 附加观测字段
        cps_now = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        out_g["contact_seen"] = any(c[3] in finger_ids for c in cps_now)
        now_z = p.getBasePositionAndOrientation(obj_id)[0][2]
        out_g["dz"] = round(float(now_z - init_base_z), 4)
        out_g["fell_off"] = bool((out_g["dz"] < -0.05) or (abs(out_g["dz"]) > 0.5))
        validated.append(out_g)
        if ok:
            success_count += 1
        print(f"[{i+1}/{len(grasps)}] Grasp success = {ok}")

    # 结果
    total = max(1, len(grasps))
    print(f"[INFO] Success rate = {success_count / total:.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(validated, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved validated grasps → {args.out}")

    # 物体形状标签（记录到 CSV）
    obj_aabb = p.getAABB(obj_id, -1)
    dx = obj_aabb[1][0] - obj_aabb[0][0]
    dy = obj_aabb[1][1] - obj_aabb[0][1]
    dz_ = obj_aabb[1][2] - obj_aabb[0][2]
    rx = abs(dx - dy) / max(1e-6, max(dx, dy))
    ry = abs(dx - dz_) / max(1e-6, max(dx, dz_))
    rz = abs(dy - dz_) / max(1e-6, max(dy, dz_))
    obj_shape = "sphere_like" if (rx < 0.15 and ry < 0.15 and rz < 0.15) else "other"

    fields = ["time","obj","cube_scale","topk","seed",
              "ee_index","ik_iters","ik_attempts","joint_force",
              "descent_step","descend_clear","vel_close","pos_close","squeeze",
              "fast","fast_scale",
              "n_trials","success_count","success_rate",
              "obj_shape","grasps_path","tag"]
    values = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.obj, args.cube_scale, args.topk, args.seed,
        args.ee_index, args.ik_iters, args.ik_attempts, args.joint_force,
        args.descent_step, args.descend_clear, args.vel_close, args.pos_close, args.squeeze,
        int(args.fast), args.fast_scale,
        len(grasps), success_count, round(success_count/max(1,len(grasps)), 4),
        obj_shape, args.grasps,
        f"{pathlib.Path(args.obj).name}_pc{int(args.pos_close)}_sq{args.squeeze}_topk{args.topk}_{pathlib.Path(args.grasps).stem if args.grasps else 'nograsps'}"
    ]
    append_summary_row(args.summary_csv, fields, values)
    print(f"[INFO] Appended summary → {args.summary_csv}")
    p.disconnect()

if __name__ == "__main__":
    main()

