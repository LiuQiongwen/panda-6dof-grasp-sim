import pybullet as p
import pybullet_data
import json
import time
import numpy as np
import argparse
import os


def euler_to_matrix(rpy):
    """Convert roll, pitch, yaw to rotation matrix."""
    roll, pitch, yaw = rpy
    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw]))).reshape(3, 3)
    return R


def check_grasp_reachability(obj_id, grasp_pose, gripper_offset=0.1):
    """
    Pseudo grasp:
    - Move gripper to pose
    - If within small distance -> attach constraint (simulate suction)
    - Lift, then release
    """
    pos, rpy = np.array(grasp_pose["position"]), np.array(grasp_pose["rpy"])
    quat = p.getQuaternionFromEuler(rpy)

    # Record initial height
    start_pos, _ = p.getBasePositionAndOrientation(obj_id)
    start_height = start_pos[2]

    # Move to grasp pose
    p.resetBasePositionAndOrientation(gripper_id, pos, quat)
    p.stepSimulation()

    # If close enough, create a temporary constraint (吸附)
    obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
    dist = np.linalg.norm(np.array(obj_pos) - np.array(pos))
    attached = False
    if dist < 0.08:  # 小于8cm则认为吸附成功
        cid = p.createConstraint(
            parentBodyUniqueId=gripper_id,
            parentLinkIndex=-1,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )
        attached = True

    # Lift
    pos[2] += 0.15
    for _ in range(240):
        p.resetBasePositionAndOrientation(gripper_id, pos, quat)
        p.stepSimulation()
        time.sleep(1. / 240.)

    # Check if object lifted
    end_pos, _ = p.getBasePositionAndOrientation(obj_id)
    lifted = (end_pos[2] - start_height) > 0.03

    # Release
    if attached:
        p.removeConstraint(cid)

    return lifted




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True, help="Path to .obj file")
    parser.add_argument("--grasps", type=str, default="grasp_6dof/dataset/sample_grasps.json")
    parser.add_argument("--out", type=str, default="grasp_6dof/dataset/validated_grasps.json")
    parser.add_argument("--vis", type=int, default=1)
    args = parser.parse_args()

    # Load grasp candidates
    with open(args.grasps, "r") as f:
        grasps = json.load(f)

    physicsClient = p.connect(p.GUI if args.vis else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # ---- Scene Setup ----
    p.loadURDF("plane.urdf")

    # Adjust camera for good view
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )

    # ---- Load Object ----
    scale = 0.3  # object scale
    try:
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=args.obj,
            rgbaColor=[0.8, 0.8, 0.8, 1],
            meshScale=[scale, scale, scale]
        )
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=args.obj,
            meshScale=[scale, scale, scale]
        )
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[0, 0, 0.05]
        )
        print(f"[INFO] Successfully loaded mesh ({scale}×):", args.obj)
    except Exception as e:
        print("[WARN] Mesh load failed, using cube_small.urdf:", e)
        obj_id = p.loadURDF("cube_small.urdf", basePosition=[0, 0, 0.05])

    # ---- Create Gripper ----
    global gripper_id
    gripper_scale = 1.5 / scale  # 自动调整比例
    gripper_id = p.loadURDF(
        os.path.join(pybullet_data.getDataPath(), "cube_small.urdf"),
        basePosition=[0, 0, 0.1],
        globalScaling=gripper_scale
    )
    p.changeVisualShape(gripper_id, -1, rgbaColor=[1, 0, 0, 1])  # 红色
    print(f"[INFO] Gripper scaling: {gripper_scale:.2f}")

    # ---- Validation Loop ----
    validated = []
    for i, grasp in enumerate(grasps):
        success = check_grasp_reachability(obj_id, grasp)
        grasp["success"] = success
        validated.append(grasp)
        print(f"[{i+1}/{len(grasps)}] Grasp success = {success}")
        if args.vis:
            time.sleep(0.5)

    # ---- Save Results ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(validated, f, indent=2)
    print(f"[INFO] Saved validated grasps → {args.out}")

    if args.vis:
        input("Press Enter to exit simulation...")
    p.disconnect()


if __name__ == "__main__":
    main()

