# tools/patch_validate.py
# -*- coding: utf-8 -*-
import re, sys, pathlib

path = pathlib.Path("grasp_6dof/validate_grasps_panda.py")
txt = path.read_text(encoding="utf-8")

def ins_after(pattern, addition, flags=re.DOTALL):
    m = re.search(pattern, txt, flags)
    if not m: 
        print(f"[WARN] pattern not found, skip:\n{pattern[:80]}...")
        return None
    return txt[:m.end()] + addition + txt[m.end():]

def replace_block(pattern, repl, flags=re.DOTALL):
    return re.sub(pattern, repl, txt, flags=flags)

orig = txt

# 1) import math
if "import math" not in txt:
    txt = re.sub(r"(import time\s*\n)", r"\1import math\n", txt)

# 2) CONTACT_NEG_MARGIN 紧跟 IK_ATTEMPTS
txt = re.sub(
    r"(IK_ATTEMPTS\s*=\s*max\(1,\s*int\(ik_attempts\)\)\s*\n)",
    r"\1    CONTACT_NEG_MARGIN = 0.05  # 掉桌/数值爆掉的下界阈值\n",
    txt
)

# 3) 未接触直接失败兜底（插在第二次 contact 计算之后）
txt = re.sub(
    r"(if not contact:\s*\n\s*cps = p\.getContactPoints\(bodyA=panda_id, bodyB=obj_id\)\s*\n\s*contact = any\(c\[3\] in finger_ids for c in cps\)\s*\n)",
    r"""\1    # 没接触就直接失败，避免盲抬横扫
    if not contact:
        for fid in finger_ids:
            p.setJointMotorControl2(panda_id, fid, p.POSITION_CONTROL, targetPosition=0.04, force=200)
        step_fn(int(0.15 * 480))
        return False
""",
    txt
)

# 4) 抬升判定：计算 dz / fell_off / success，并更新 DEBUG 打印
txt = re.sub(
    r"""now_z\s*=\s*p\.getBasePositionAndOrientation\(obj_id\)\[0\]\[2\]\s*\n\s*
base_z\s*=\s*\(init_base_z if init_base_z is not None else obj_pos0\[2\]\)\s*\n\s*
lifted\s*=\s*\(now_z\s*-\s*base_z\)\s*>\s*LIFT_SUCCESS_DZ\s*\n\s*
print\(f"\[DEBUG] contact=\{contact\}, Δz=\{now_z\s*-\s*base_z:\.3f\}, success=\{lifted\}"\)\s*""",
    r"""now_z = p.getBasePositionAndOrientation(obj_id)[0][2]
base_z = (init_base_z if init_base_z is not None else obj_pos0[2])
dz = float(now_z - base_z)
fell_off = (dz < -CONTACT_NEG_MARGIN) or (abs(dz) > 0.5)
lifted = (dz > LIFT_SUCCESS_DZ) and (not fell_off)
print(f"[DEBUG] contact={contact}, Δz={dz:.3f}, fell_off={fell_off}, success={lifted}")
""",
    txt
)

# 5) 物理参数强化 + 固定步长（在 setGravity 之后注入）
txt = re.sub(
    r"(p\.setGravity\(0,\s*0,\s*-9\.8\)\s*\n)",
    r"""\1    # —— 物理稳定性/确定性参数 ——
    p.setPhysicsEngineParameter(
        deterministicOverlappingPairs=1,
        collisionFilterMode=1,
        contactSlop=1e-3,
        enableConeFriction=1,
        solverResidualThreshold=1e-7,
        numSolverIterations=200,
        erp=0.2, contactERP=0.2, frictionERP=0.2,
        useSplitImpulse=1, splitImpulsePenetrationThreshold=-0.01
    )
    p.setTimeStep(1.0/480.0)
""",
    txt
)

# 6) 桌面动力学调优（在 Detected table 打印后注入）
txt = re.sub(
    r"""(print\(f"\[INFO] Detected table top z = \{TABLE_TOP_Z:\.3f\}"\)\s*\n)""",
    r"""\1    p.changeDynamics(table_id, -1,
                     lateralFriction=1.6, rollingFriction=0.05, spinningFriction=0.02,
                     restitution=0.0)
""",
    txt
)

# 7) grasps 按 score 排序兜底健壮
txt = re.sub(
    r"""if\s+args\.topk\s+is\s+not\s+None\s+and\s+len\(grasps\)\s*>\s*0:\s*\n\s*
if\s*\"score\"\s*in\s*grasps\[0\]:\s*\n\s*
\s*grasps\s*=\s*sorted\(grasps,\s*key=lambda g:\s*g\.get\(\"score\",\s*0\.0\),\s*reverse=True\)\s*\n\s*
\s*grasps\s*=\s*grasps\[:args\.topk\]\s*""",
    r"""if args.topk is not None and len(grasps) > 0:
        try:
            grasps = sorted(grasps, key=lambda g: float(g.get("score", 0.0)), reverse=True)
        except Exception:
            pass
        grasps = grasps[:args.topk]
""",
    txt
)

# 8) 每次结果增加 contact_seen / dz / fell_off 字段（在 out_g["success"]=... 之后）
txt = re.sub(
    r"""(out_g\s*=\s*dict\(grasp\);\s*out_g\["success"\]\s*=\s*bool\(ok\)\s*\n)""",
    r"""\1        # 附加观测字段
        cps_now = p.getContactPoints(bodyA=panda_id, bodyB=obj_id)
        out_g["contact_seen"] = any(c[3] in finger_ids for c in cps_now)
        now_z = p.getBasePositionAndOrientation(obj_id)[0][2]
        out_g["dz"] = round(float(now_z - init_base_z), 4)
        out_g["fell_off"] = bool((out_g["dz"] < -0.05) or (abs(out_g["dz"]) > 0.5))
""",
    txt
)

# 9) CSV：增加 obj_shape / grasps_path / tag 三列（先计算，再扩展 fields 与 values）
# 在保存 json 后构造 fields/values 之前插入 obj_shape 计算
txt = re.sub(
    r"""(print\(f"\[INFO] Saved validated grasps → \{args\.out\}"\)\s*\n\s*)fields\s*=\s*\[\s*""",
    r"""\1# 物体形状标签（记录到 CSV）
    obj_aabb = p.getAABB(obj_id, -1)
    dx = obj_aabb[1][0] - obj_aabb[0][0]
    dy = obj_aabb[1][1] - obj_aabb[0][1]
    dz_ = obj_aabb[1][2] - obj_aabb[0][2]
    rx = abs(dx - dy) / max(1e-6, max(dx, dy))
    ry = abs(dx - dz_) / max(1e-6, max(dx, dz_))
    rz = abs(dy - dz_) / max(1e-6, max(dy, dz_))
    obj_shape = "sphere_like" if (rx < 0.15 and ry < 0.15 and rz < 0.15) else "other"

    fields = [""",
    txt
)

# 扩展 fields 末尾
txt = re.sub(
    r"""("n_trials","success_count","success_rate"\s*\])""",
    r'''\1 + ["obj_shape","grasps_path","tag"]''',
    txt
)

# 扩展 values 末尾
txt = re.sub(
    r"""(len\(grasps\),\s*success_count,\s*round\(success_count/max\(1,len\(grasps\)\),\s*4\)\s*\])""",
    r'''\1 + [obj_shape, args.grasps, f"{pathlib.Path(args.obj).name}_pc{int(args.pos_close)}_sq{args.squeeze}_topk{args.topk}_{pathlib.Path(args.grasps).stem if args.grasps else "fallback"}"]''',
    txt
)

# 写回
if txt != orig:
    path.write_text(txt, encoding="utf-8")
    print("[OK] validate_grasps_panda.py patched.")
else:
    print("[NOTE] Nothing changed (maybe already patched).")

