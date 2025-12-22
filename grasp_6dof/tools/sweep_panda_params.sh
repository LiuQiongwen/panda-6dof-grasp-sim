#!/usr/bin/env bash
set -e

OBJ="grasp_6dof/assets/cylinder.urdf"
GRASPS="grasp_6dof/dataset/gen_cylinder_patched_pruned_FIXED.json"
OUT_DIR="grasp_6dof/dataset"
SUMMARY="grasp_6dof/out/summary_tune.csv"

SEED=1
TOPK=24

# 参数网格，你可以根据需要自己加/删
DESCEND_CLEAR_LIST=(0.014 0.016 0.020)
DESCENT_STEP_LIST=(0.0004 0.0006)
VEL_CLOSE_LIST=(0.6 0.8 1.0)
POS_CLOSE_LIST=(700 900 1100)
SQUEEZE_LIST=(1.0 1.5)

mkdir -p "$(dirname "$SUMMARY")"
echo "[INFO] sweep started, results → $SUMMARY"

for dc in "${DESCEND_CLEAR_LIST[@]}"; do
  for ds in "${DESCENT_STEP_LIST[@]}"; do
    for vc in "${VEL_CLOSE_LIST[@]}"; do
      for pc in "${POS_CLOSE_LIST[@]}"; do
        for sq in "${SQUEEZE_LIST[@]}"; do

          TAG="dc${dc}_ds${ds}_vc${vc}_pc${pc}_sq${sq}"
          OUT_JSON="${OUT_DIR}/gen_cylinder_validated_${TAG}.json"

          echo
          echo "=== Running $TAG ==="

          python grasp_6dof/validate_grasps_panda.py \
            --obj "$OBJ" \
            --cube-scale 0.08 \
            --descend-clear "$dc" --descent-step "$ds" \
            --vel-close "$vc" --pos-close "$pc" --squeeze "$sq" \
            --ee-index 11 --topk "$TOPK" --seed "$SEED" \
            --vis 0 --fast \
            --grasps "$GRASPS" \
            --out "$OUT_JSON" \
            --summary-csv "$SUMMARY"
        done
      done
    done
  done
done

echo "[INFO] sweep finished."
