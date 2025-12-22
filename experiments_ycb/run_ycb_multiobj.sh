#!/bin/bash
set -e

OBJ_LIST=objects_ycb.txt
SUMMARY=results/summary_ycb.csv

mkdir -p data/pc data/grasps_raw data/grasps_val results

# 可选：清空旧 summary
rm -f "$SUMMARY"

while read oid urdf_path target_h; do
  # 跳过注释行或空行
  [[ "$oid" =~ ^#.*$ ]] && continue
  [[ -z "$oid" ]] && continue

  echo "==== Object: $oid  ($urdf_path, target_h=$target_h) ===="

  pc_path=data/pc/${oid}_$(echo $target_h | sed 's/0\.//').ply
  grasps_raw=data/grasps_raw/${oid}_geom_K256.json
  grasps_val=data/grasps_val/${oid}_geom_val.json

  # 1) URDF -> 点云
  python grasp_6dof/tools/make_pc_from_urdf.py \
    --urdf "$urdf_path" \
    --out "$pc_path" \
    --target-h "$target_h" \
    --table-z -0.004

  # 2) 点云 -> 几何 grasp 候选
  python grasp_6dof/grasp_gen_open3d.py \
    --pc "$pc_path" \
    --out "$grasps_raw" \
    --topk 256 \
    --yaw-bins 12 \
    --offset-mm 2 8 \
    --table-z -0.004 \
    --seed 1

  # 3) Panda 仿真验证 + 写 summary
  python grasp_6dof/validate_grasps_panda.py \
    --obj "$urdf_path" \
    --cube-scale "$target_h" \
    --grasps "$grasps_raw" \
    --out "$grasps_val" \
    --topk 64 \
    --fast \
    --fast-scale 0.9 \
    --reset-each-trial 1 \
    --seed 123 \
    --descent-step 0.0015 \
    --descend-clear 0.020 \
    --vel-close 0.8 \
    --pos-close 700 \
    --squeeze 0.8 \
    --summary-csv "$SUMMARY"

done < "$OBJ_LIST"

