#!/bin/bash
set -e

OBJ_LIST=objects_ycb.txt
SUMMARY=results/summary_ycb_random.csv

mkdir -p data/pc data/grasps_raw data/grasps_val results

# 可选：清空旧 summary
rm -f "$SUMMARY"

while read oid urdf_path target_h; do
  # 跳过注释和空行
  [[ "$oid" =~ ^#.*$ ]] && continue
  [[ -z "$oid" ]] && continue

  echo "==== [RANDOM] Object: $oid  ($urdf_path, target_h=$target_h) ===="

  grasps_random=data/grasps_raw/${oid}_random_K256.json
  grasps_val_rand=data/grasps_val/${oid}_random_val.json

  if [ ! -f "$grasps_random" ]; then
    echo "[WARN] Random grasps file not found: $grasps_random, skip."
    continue
  fi

  python grasp_6dof/validate_grasps_panda.py \
    --obj "$urdf_path" \
    --cube-scale "$target_h" \
    --grasps "$grasps_random" \
    --out "$grasps_val_rand" \
    --topk 64 \
    --fast \
    --fast-scale 0.9 \
    --reset-each-trial 1 \
    --seed 321 \
    --descent-step 0.0015 \
    --descend-clear 0.020 \
    --vel-close 0.8 \
    --pos-close 700 \
    --squeeze 0.8 \
    --summary-csv "$SUMMARY"

done < "$OBJ_LIST"

