# panda-6dof-grasp-sim

Simulation-first 6-DoF grasp pipeline for tabletop objects with a Franka Emika Panda arm, built on top of PyBullet and Open3D.

Starting from object URDF models, this project:

1. Generates multi-view point clouds via ray casting in PyBullet.  
2. Samples geometric 6-DoF grasp candidates in Open3D.  
3. Validates grasps in PyBullet with a physically realistic Panda controller  
   (approach → descent → finger closing → compliant squeezing → lifting).  
4. Builds per-object grasp libraries and performs controller parameter tuning.  

The code is designed as a **strong geometric baseline** and a **reusable simulation framework** for future work on:

- learned grasp scoring networks (Geometric LG-GSN), and  
- language-conditioned grasp scoring (Language-Guided LG-GSN) on top of OWG.

> **Status:** research code used for an in-preparation journal paper.  
> APIs may evolve as experiments continue.

---

## 1. Features

- **URDF → point cloud**
  - Multi-view depth rendering and ray casting in PyBullet.
  - Object normalization (scale to target height) and consistent table frame.

- **Geometric 6-DoF grasp generation**
  - Sampling in Open3D on the object point cloud.
  - Local normals, antipodal-ish contact pairs, collision checks with the table.
  - Geometric scoring (clearance, alignment, edge distance, etc.).

- **Panda controller in PyBullet**
  - Cartesian approach and vertical descent with step size control.
  - Finger closing with velocity and compliant squeezing.
  - Lifting and hold with success / failure detection.

- **Grasp validation and ranking**
  - Save per-object validated grasps with rich metrics (JSON/CSV).
  - Compute SR@k (success rate at top-k) for k ∈ {1, 3, 5, 10, 20, 32, 64}.
  - Compare geometric vs random 6-DoF baselines.
  - Analyze effect of geometric ranking vs random ordering.

- **Controller tuning**
  - Small grid search over descent step size, closing velocity, squeezing magnitude.
  - Per-object tuned controller configuration to maximize grasp success.

- **YCB experiments**
  - 8 representative YCB objects (bleach, bowl, cracker box, mug, mustard, potted meat, sugar box, tomato soup).
  - Per-object SR@k curves & geom vs random plots.

---

## 2. Dependencies

Tested with:

- Python 3.9+  
- [PyBullet](https://github.com/bulletphysics/bullet3)  
- [Open3D](http://www.open3d.org/)  
- NumPy, SciPy, pandas, matplotlib  
- (optional) tqdm, seaborn, etc. for plotting

Recommended environment setup:

```bash
# Option 1: use conda (for new users)
conda create -n panda-6dof python=3.9
conda activate panda-6dof
pip install -r requirements.txt

# Option 2: use your own venv
python -m venv owg_env
source owg_env/bin/activate
pip install -r requirements.txt
