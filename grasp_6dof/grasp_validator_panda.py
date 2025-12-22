import pybullet as p
import pybullet_data
import numpy as np
import json, os, time

def validate_grasps(cfg, sample_file, out_file="grasp_6dof/dataset/validated_grasps.json"):
    """在 Panda 环境中验证抓取姿态"""
    with open(sample_file, "r") as f:
        grasps = json.load(f)

    p.connect(p.GUI if cfg.policy.vis else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # 场景搭建
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
    panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # 加载待抓物体（放在桌面中央）
    obj_path = os.path.join(pybullet_data.getDataPath(), "sphere_smooth.obj")
    vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.1]*3)
    col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[0.1]*3)
    obj_id = p.createMultiBody(0.05, col_shape, vis_shape, basePosition=[0.5, 0, 0.02])

    validated = []
    for i, g in enumerate(grasps):
        pos, rpy = np.array(g["position"]), np.array(g["rpy"])
        quat = p.getQuaternionFromEuler(rpy)

        # 计算 IK 并移动手臂
        jpos = p.calculateInverseKinematics(panda_id, 11, pos, quat)
        for j in range(7):
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, jpos[j])
        for _ in range(240): p.stepSimulation()

        # 模拟夹爪闭合
        for fid in [9, 10]:
            p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.0, force=200)
        for _ in range(240): p.stepSimulation()

        # 尝试提起物体
        pos[2] += 0.1
        jpos = p.calculateInverseKinematics(panda_id, 11, pos, quat)
        for j in range(7):
            p.setJointMotorControl2(panda_id, j, p.POSITION_CONTROL, jpos[j])
        for _ in range(240): p.stepSimulation()

        # 判断是否抓起
        obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
        success = obj_pos[2] > 0.04
        validated.append({**g, "success": success})
        print(f"[{i+1}/{len(grasps)}] Grasp success = {success}")

    with open(out_file, "w") as f:
        json.dump(validated, f, indent=2)
    print(f"[INFO] Saved validated grasps → {out_file}")

    if cfg.policy.vis:
        input("Press Enter to exit simulation...")

    p.disconnect()
    return out_file

