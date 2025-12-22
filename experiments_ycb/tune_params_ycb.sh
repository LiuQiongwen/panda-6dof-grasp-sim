#!/bin/bash
set -e

OBJ_LIST=objects_ycb.txt
SUMMARY=results/summary_ycb_tune.csv

mkdir -p results

# 可选：先清空旧的 tune summary
rm -f "$SUMMARY"

# 参数网格：每个字符串是 "descent_step vel_close squeeze"
PARAMS=(
  "0.0010 0.6 0.6"
  "0.0010 0.6 0.8"
  "0.0010 0.8 0.6"
  "0.0010 0.8 0.8"
  "0.0015 0.6 0.6"
  "0.0015 0.6 0.8"
  "0.0015 0.8 0.6"
  "0.0015 0.8 0.8"   # 这一组就是当前默认参数
)

while read oid urdf_path target_h; do
  # 跳过注释和空行
  [[ "$oid" =~ ^#.*$ ]] && continue
  [[ -z "$oid" ]] && continue

  echo "==== [TUNE] Object: $oid  ($urdf_path, target_h=$target_h) ===="

  grasps_raw=data/grasps_raw/${oid}_geom_K256.json
  if [ ! -f "$grasps_raw" ]; then
    echo "[WARN] Geometric grasps not found: $grasps_raw, skip."
    continue
  fi

  for p in "${PARAMS[@]}"; do
    # 拆出三个参数
    read ds vc sq <<< "$p"
    echo "  -> ds=$ds, vel_close=$vc, squeeze=$sq"

    # 为避免覆盖结果，out 文件名带上参数
    grasps_val=data/grasps_val/${oid}_ds${ds}_vc${vc}_sq${sq}_val.json

    python grasp_6dof/validate_grasps_panda.py \
      --obj "$urdf_path" \
      --cube-scale "$target_h" \
      --grasps "$grasps_raw" \
      --out "$grasps_val" \
      --topk 32 \
      --fast \
      --fast-scale 0.9 \
      --reset-each-trial 1 \
      --seed 456 \
      --descent-step "$ds" \
      --descend-clear 0.020 \
      --vel-close "$vc" \
      --pos-close 700 \
      --squeeze "$sq" \
      --summary-csv "$SUMMARY"

  done

done < "$OBJ_LIST"

