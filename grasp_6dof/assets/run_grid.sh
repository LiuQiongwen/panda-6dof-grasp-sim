#!/usr/bin/env bash
set -e

OBJ_LIST=("cube.urdf" "grasp_6dof/assets/sphere_small.urdf" "grasp_6dof/assets/cylinder.urdf")
YAWS=(8 12 16)
VOXELS=(0.004 0.005 0.006)
ZMARGINS=(0.003 0.004)
SEEDS=(1 2 3)

OUTDIR="grasp_6dof/dataset"
SUMCSV="grasp_6dof/out/summary.csv"
mkdir -p "$OUTDIR" "$(dirname $SUMCSV)"

echo "== Pipeline start (multi-object) =="
echo "Summary CSV: $SUMCSV"

for OBJ in "${OBJ_LIST[@]}"; do
  ONAME=$(basename "$OBJ" .urdf)
  for Y in "${YAWS[@]}"; do
    for V in "${VOXELS[@]}"; do
      for ZM in "${ZMARGINS[@]}"; do
        for S in "${SEEDS[@]}"; do
          GEN="$OUTDIR/${ONAME}_y${Y}_v${V}_zm${ZM}_s${S}.json"
          VAL="$OUTDIR/${ONAME}_y${Y}_v${V}_zm${ZM}_s${S}_validated.json"

          echo "--- GEN [$GEN] ---"
          if ! PYBULLET_EGL=1 python grasp_6dof/generate_grasps_open3d.py \
              --obj "$OBJ" --cube-scale 0.08 \
              --n-cand 1000 --yaw-samples $Y --voxel $V \
              --z-margin $ZM \
              --topk 50 --topk-bullet 150 \
              --renderer opengl --vis 0 --img 256 256 \
              --seed $S --out "$GEN" ; then
            echo "[RETRY] GEN once: $GEN"
            PYBULLET_EGL=1 python grasp_6dof/generate_grasps_open3d.py \
              --obj "$OBJ" --cube-scale 0.08 \
              --n-cand 1000 --yaw-samples $Y --voxel $V \
              --z-margin $ZM \
              --topk 50 --topk-bullet 150 \
              --renderer opengl --vis 0 --img 256 256 \
              --seed $S --out "$GEN"
          fi

          echo "--- VAL [$VAL] ---"
          if ! python grasp_6dof/validate_grasps_panda.py \
              --obj "$OBJ" --cube-scale 0.08 --vis 0 \
              --fast --fast-scale 0.85 \
              --descent-step 0.002 --descend-clear 0.020 \
              --vel-close 0.30 --pos-close 950 --squeeze 0.40 \
              --z-margin $ZM \
              --topk 12 --seed $S \
              --grasps "$GEN" --out "$VAL" \
              --summary-csv "$SUMCSV" ; then
            echo "[RETRY] VAL once: $VAL"
            python grasp_6dof/validate_grasps_panda.py \
              --obj "$OBJ" --cube-scale 0.08 --vis 0 \
              --fast --fast-scale 0.85 \
              --descent-step 0.002 --descend-clear 0.020 \
              --vel-close 0.30 --pos-close 950 --squeeze 0.40 \
              --z-margin $ZM \
              --topk 12 --seed $S \
              --grasps "$GEN" --out "$VAL" \
              --summary-csv "$SUMCSV"
          fi
        done
      done
    done
  done
done

echo "== Done =="

